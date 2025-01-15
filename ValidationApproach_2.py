import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer
from typing import List, Dict, Set
from dataclasses import dataclass
from sklearn.metrics import ndcg_score
import numpy as np
from data_preparing_rerank_bert import get_elements
import pytrec_eval

@dataclass
class ValidationQueryDocument:
    query_id: int
    query_text: str
    positive_docs: List[str]
    positive_ids: Set[int]
    candidate_docs: List[str]  # Mix of positive and negative documents for evaluation

class ValidationDataset(Dataset):
    def __init__(self, query_documents: List[ValidationQueryDocument], 
                 tokenizer: BertTokenizer, max_length_query: int = 128, 
                 max_length_doc: int = 512):
        self.query_documents = query_documents
        self.tokenizer = tokenizer
        self.max_length_query = max_length_query
        self.max_length_doc = max_length_doc

    def __len__(self) -> int:
        return len(self.query_documents)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query_doc = self.query_documents[idx]
        
        # Tokenize query
        query_encoding = self.tokenizer(
            query_doc.query_text,
            max_length=self.max_length_query,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize all candidate documents
        candidate_encodings = [
            self.tokenizer(
                doc,
                max_length=self.max_length_doc,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for doc in query_doc.candidate_docs
        ]

        # Create label tensor (1 for positive documents, 0 for negative)
        labels = torch.tensor([
            1 if i in query_doc.positive_ids else 0
            for i in range(len(query_doc.candidate_docs))
        ])

        return {
            'query_id': query_doc.query_id,
            'query_ids': query_encoding['input_ids'].squeeze(),
            'query_mask': query_encoding['attention_mask'].squeeze(),
            'candidate_ids': torch.stack([enc['input_ids'].squeeze() for enc in candidate_encodings]),
            'candidate_mask': torch.stack([enc['attention_mask'].squeeze() for enc in candidate_encodings]),
            'labels': labels
        }

class DenseRetrieverBERT(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", pooling: str = "mean"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == "mean":
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        elif self.pooling == "cls":
            embeddings = outputs.last_hidden_state[:, 0]

        return F.normalize(embeddings, p=2, dim=1)

class CrossAttentionReranker(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.score_layer = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, query_ids: torch.Tensor, query_mask: torch.Tensor,
                passage_ids: torch.Tensor, passage_mask: torch.Tensor) -> torch.Tensor:
        sep_token = torch.full((query_ids.size(0), 1), self.bert.config.sep_token_id, device=query_ids.device)
        input_ids = torch.cat([query_ids, sep_token, passage_ids], dim=1)
        attention_mask = torch.cat([query_mask, torch.ones_like(sep_token), passage_mask], dim=1)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        score = self.score_layer(cls_output)
        return score

def compute_metrics(scores: torch.Tensor, labels: torch.Tensor):
    # Convert to numpy for metric calculation
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()
    
    # Calculate metrics
    metrics = {}
    
    # MRR
    rankings = (-scores_np).argsort(axis=1)
    reciprocal_ranks = []
    for i, ranking in enumerate(rankings):
        positive_positions = np.where(labels_np[i][ranking] == 1)[0]
        if len(positive_positions) > 0:
            reciprocal_ranks.append(1.0 / (positive_positions[0] + 1))
        else:
            reciprocal_ranks.append(0.0)
    metrics['mrr'] = np.mean(reciprocal_ranks)

    # Recall@k
    for k in [5, 10]:
        recall_at_k = []
        for i, ranking in enumerate(rankings):
            top_k_docs = ranking[:k]
            relevant_in_top_k = np.sum(labels_np[i][top_k_docs])
            total_relevant = np.sum(labels_np[i])
            if total_relevant > 0:
                recall_at_k.append(relevant_in_top_k / total_relevant)
        metrics[f'recall@{k}'] = np.mean(recall_at_k)

    # NDCG@k
    for k in [5, 10]:
        ndcg_scores = []
        for i in range(len(scores_np)):
            ndcg_scores.append(ndcg_score([labels_np[i]], [scores_np[i]], k=k))
        metrics[f'ndcg@{k}'] = np.mean(ndcg_scores)
    evaluator = pytrec_eval.RelevanceEvaluator(
        {0: {0: int(labels_np[0][0])}},  # Dummy initialization
        {'map', 'ndcg', 'P_5', 'P_10', 'recall_5', 'recall_10'}
    )

    qrels = {}
    run = {}
    
    # Prepare data in pytrec_eval format
    for query_idx in range(len(scores_np)):
        qrels[str(query_idx)] = {
            str(doc_idx): int(label)
            for doc_idx, label in enumerate(labels_np[query_idx])
        }
        
        run[str(query_idx)] = {
            str(doc_idx): float(score)
            for doc_idx, score in enumerate(scores_np[query_idx])
        }

    # Create new evaluator with actual qrels
    evaluator = pytrec_eval.RelevanceEvaluator(
        qrels,
        {'map', 'ndcg', 'P_5', 'P_10', 'recall_5', 'recall_10'}
    )
    
    # Compute pytrec_eval metrics
    results = evaluator.evaluate(run)
    
    # Average the metrics across queries
    mean_metrics = {}
    for measure in ['map', 'ndcg', 'P_5', 'P_10', 'recall_5', 'recall_10']:
        mean_metrics[measure] = np.mean([
            query_measures[measure]
            for query_measures in results.values()
        ])
    
    # Add pytrec_eval metrics to the existing metrics
    metrics.update({
        'map': mean_metrics['map'],
        'pytrec_ndcg': mean_metrics['ndcg'],
        'P@5': mean_metrics['P_5'],
        'P@10': mean_metrics['P_10'],
        'pytrec_recall@5': mean_metrics['recall_5'],
        'pytrec_recall@10': mean_metrics['recall_10']
    })

    return metrics
    return metrics

def validate_models(retriever_path: str, reranker_path: str, validation_dataloader: DataLoader, 
                   device: torch.device):
    # Initialize models
    retriever = DenseRetrieverBERT().to(device)
    reranker = CrossAttentionReranker().to(device)
    
    # Load saved weights
    retriever.load_state_dict(torch.load(retriever_path, map_location=device))
    reranker.load_state_dict(torch.load(reranker_path, map_location=device))
    
    # Set to evaluation mode
    retriever.eval()
    reranker.eval()
    
    all_retriever_scores = []
    all_reranker_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in validation_dataloader:
            # Move batch to device
            query_ids = batch['query_ids'].to(device)
            query_mask = batch['query_mask'].to(device)
            candidate_ids = batch['candidate_ids'].to(device)
            candidate_mask = batch['candidate_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Retriever forward pass
            query_embeddings = retriever(query_ids, query_mask)
            candidate_embeddings = retriever(
                candidate_ids.view(-1, candidate_ids.size(-1)),
                candidate_mask.view(-1, candidate_mask.size(-1))
            )
            candidate_embeddings = candidate_embeddings.view(
                candidate_ids.size(0), candidate_ids.size(1), -1
            )
            
            # Compute retriever scores
            retriever_scores = torch.matmul(
                query_embeddings.unsqueeze(1),
                candidate_embeddings.transpose(1, 2)
            ).squeeze(1)
            
            # Reranker forward pass
            reranker_scores = torch.zeros_like(retriever_scores)
            for i in range(candidate_ids.size(1)):
                reranker_scores[:, i] = reranker(
                    query_ids,
                    query_mask,
                    candidate_ids[:, i],
                    candidate_mask[:, i]
                ).squeeze()
            
            # Collect scores and labels
            all_retriever_scores.append(retriever_scores)
            all_reranker_scores.append(reranker_scores)
            all_labels.append(labels)
    
    # Concatenate all batches
    all_retriever_scores = torch.cat(all_retriever_scores, dim=0)
    all_reranker_scores = torch.cat(all_reranker_scores, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Compute and print metrics
    retriever_metrics = compute_metrics(all_retriever_scores, all_labels)
    reranker_metrics = compute_metrics(all_reranker_scores, all_labels)
    
    print("\nRetriever Metrics:")
    for metric, value in retriever_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\nReranker Metrics:")
    for metric, value in reranker_metrics.items():
        print(f"{metric}: {value:.4f}")

def prepare_validation_data():
    # Get data from your data preparation module
    queries, positive_docs, negative_docs = get_elements()
    # TODO GET SOME PART OF THE DATA
    validation_documents = []
    q_num = 0
    pos_ctr = 0
    
    for query in queries:
        if query != 'No relevant document found.':
            doc_count = len(positive_docs[q_num])
            
            # Combine positive and negative documents as candidates
            candidate_docs = positive_docs[q_num] + negative_docs[q_num]
            
            validation_documents.append(
                ValidationQueryDocument(
                    query_id=q_num,
                    query_text=query,
                    positive_docs=positive_docs[q_num],
                    positive_ids=set(range(pos_ctr, pos_ctr + doc_count)),
                    candidate_docs=candidate_docs
                )
            )
            pos_ctr += doc_count
            q_num += 1
    
    return validation_documents

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Initialize tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Prepare validation data
    validation_documents = prepare_validation_data()
    validation_dataset = ValidationDataset(validation_documents, tokenizer)
    validation_dataloader = DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False
    )
    
    # Paths to saved models
    retriever_path = "multi_positive_retriever.pt"
    reranker_path = "multi_positive_reranker.pt"
    
    # Run validation
    validate_models(retriever_path, reranker_path, validation_dataloader, device)

if __name__ == "__main__":
    main()