import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer
import numpy as np
from typing import List, Dict, Set
from sklearn.metrics import ndcg_score

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

        return embeddings

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

def compute_mrr(scores: torch.Tensor, labels: torch.Tensor) -> float:
    sorted_indices = torch.argsort(scores, dim=1, descending=True)
    ranks = (sorted_indices == labels.unsqueeze(1)).nonzero(as_tuple=True)[1] + 1
    mrr = torch.mean(1.0 / ranks.float()).item()
    return mrr

def compute_recall_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    top_k_indices = torch.argsort(scores, dim=1, descending=True)[:, :k]
    hits = torch.any(top_k_indices == labels.unsqueeze(1), dim=1).float()
    recall = torch.mean(hits).item()
    return recall

def compute_ndcg_at_k(scores: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    scores_np = scores.cpu().numpy()
    labels_np = labels.cpu().numpy()
    ndcg = 0.0
    for i in range(scores_np.shape[0]):
        y_true = np.zeros(scores_np.shape[1])
        y_true[labels_np[i]] = 1
        y_score = scores_np[i]
        ndcg += ndcg_score([y_true], [y_score], k=k)
    ndcg /= scores_np.shape[0]
    return ndcg

def validate_model(retriever_path: str, reranker_path: str, dataloader: DataLoader, device: torch.device):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    # Load models
    retriever = DenseRetrieverBERT().to(device)
    retriever.load_state_dict(torch.load(retriever_path, map_location=device))
    retriever.eval()

    reranker = CrossAttentionReranker().to(device)
    reranker.load_state_dict(torch.load(reranker_path, map_location=device))
    reranker.eval()

    total_mrr = 0.0
    total_recall_at_k = {5: 0.0, 10: 0.0}
    total_ndcg_at_k = {5: 0.0, 10: 0.0}
    num_samples = 0

    with torch.no_grad():
        for batch in dataloader:
            query_ids = batch['query_ids'].to(device)
            query_mask = batch['query_mask'].to(device)
            candidate_ids = batch['candidate_ids'].to(device)
            candidate_mask = batch['candidate_mask'].to(device)
            labels = batch['labels'].to(device)

            # Retriever step
            query_embeddings = retriever(query_ids, query_mask)
            candidate_embeddings = retriever(candidate_ids.view(-1, candidate_ids.size(-1)), candidate_mask.view(-1, candidate_mask.size(-1)))
            candidate_embeddings = candidate_embeddings.view(candidate_ids.size(0), candidate_ids.size(1), -1)

            retriever_scores = torch.bmm(query_embeddings.unsqueeze(1), candidate_embeddings.transpose(1, 2)).squeeze(1)

            # Reranker step
            reranker_scores = torch.zeros_like(retriever_scores)
            for i in range(candidate_ids.size(1)):
                reranker_scores[:, i] = reranker(query_ids, query_mask, candidate_ids[:, i], candidate_mask[:, i]).squeeze()

            # Metrics calculation
            mrr = compute_mrr(reranker_scores, labels)
            recall_at_5 = compute_recall_at_k(reranker_scores, labels, k=5)
            recall_at_10 = compute_recall_at_k(reranker_scores, labels, k=10)
            ndcg_at_5 = compute_ndcg_at_k(reranker_scores, labels, k=5)
            ndcg_at_10 = compute_ndcg_at_k(reranker_scores, labels, k=10)

            total_mrr += mrr * query_ids.size(0)
            total_recall_at_k[5] += recall_at_5 * query_ids.size(0)
            total_recall_at_k[10] += recall_at_10 * query_ids.size(0)
            total_ndcg_at_k[5] += ndcg_at_5 * query_ids.size(0)
            total_ndcg_at_k[10] += ndcg_at_10 * query_ids.size(0)
            num_samples += query_ids.size(0)

    # Average metrics
    avg_mrr = total_mrr / num_samples
    avg_recall_at_5 = total_recall_at_k[5] / num_samples
    avg_recall_at_10 = total_recall_at_k[10] / num_samples
    avg_ndcg_at_5 = total_ndcg_at_k[5] / num_samples
    avg_ndcg_at_10 = total_ndcg_at_k[10] / num_samples

    print(f"MRR: {avg_mrr:.4f}")
    print(f"Recall@5: {avg_recall_at_5:.4f}")
    print(f"Recall@10: {avg_recall_at_10:.4f}")
    print(f"NDCG@5: {avg_ndcg_at_5:.4f}")
    print(f"NDCG@10: {avg_ndcg_at_10:.4f}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dummy dataset for demonstration (replace with actual DataLoader)
    class DummyDataset:
        def __len__(self):
            return 100

        def __getitem__(self, idx):
            return {
                'query_ids': torch.randint(0, 30522, (128,)),
                'query_mask': torch.ones(128),
                'candidate_ids': torch.randint(0, 30522, (10, 128)),
                'candidate_mask': torch.ones(10, 128),
                'labels': torch.tensor(idx % 10)
            }

    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=8)

    # Paths to saved models
    retriever_path = "multi_positive_retriever.pt"
    reranker_path = "multi_positive_reranker.pt"

    # Validate models
    validate_model(retriever_path, reranker_path, dataloader, device)

if __name__ == "__main__":
    main()