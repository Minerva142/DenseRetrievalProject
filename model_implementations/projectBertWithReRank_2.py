import os.path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from typing import List, Tuple, Dict, Set
import logging
import random
from dataclasses import dataclass
from torch.nn import functional as F
from data_prepare_files.data_preparing_rerank_bert import get_elements_train


@dataclass
class QueryDocument:
    query_id: int
    query_text: str
    positive_docs: List[str]  # Multiple positive documents
    positive_ids: Set[int]  # IDs of positive documents for efficient lookup
    negative_docs: List[str]  # Known negative documents


class CrossAttentionReranker(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.score_layer = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, query_ids: torch.Tensor, query_mask: torch.Tensor,
                pos_ids: torch.Tensor, pos_mask: torch.Tensor) -> torch.Tensor:
        # Use explicit SEP token ID (102 for BERT)
        sep_token = torch.full((query_ids.size(0), 1), 102,
                               device=query_ids.device)
        # Ensure pos_ids is reshaped to match dimensions
        pos_ids = pos_ids.view(pos_ids.size(0), -1, pos_ids.size(-1))  # Reshape pos_ids to [batch_size, num_positives, seq_len]
        input_ids = torch.cat([query_ids.unsqueeze(1), sep_token, pos_ids], dim=1)  # Unsqueeze query_ids to match dimensions
        attention_mask = torch.cat([query_mask.unsqueeze(1),
                                    torch.ones_like(sep_token),
                                    pos_mask], dim=1)

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        score = self.score_layer(cls_output)
        return score

class MultiPositiveRetrieverDataset(Dataset):
    def __init__(self, query_documents: List[QueryDocument],
                 tokenizer: BertTokenizer, max_positives: int = 5):
        self.query_documents = query_documents
        self.tokenizer = tokenizer
        self.max_positives = max_positives

    def __len__(self) -> int:
        return len(self.query_documents)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query_doc = self.query_documents[idx]

        # Tokenize query
        query_encoding = self.tokenizer(
            query_doc.query_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Sample positive documents if there are more than max_positives
        pos_docs = query_doc.positive_docs
        if len(pos_docs) > self.max_positives:
            pos_docs = random.sample(pos_docs, self.max_positives)

        # Pad positive documents to max_positives if necessary
        while len(pos_docs) < self.max_positives:
            pos_docs.append(pos_docs[0])  # Duplicate first positive if needed

        # Tokenize positive documents
        pos_encodings = [
            self.tokenizer(
                doc,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for doc in pos_docs
        ]

        # Tokenize negative documents
        neg_encodings = [
            self.tokenizer(
                neg,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for neg in query_doc.negative_docs
        ]

        return {
            'query_id': query_doc.query_id,
            'query_ids': query_encoding['input_ids'].squeeze(),
            'query_mask': query_encoding['attention_mask'].squeeze(),
            'pos_ids': torch.stack([enc['input_ids'].squeeze() for enc in pos_encodings]),
            'pos_mask': torch.stack([enc['attention_mask'].squeeze() for enc in pos_encodings]),
            'neg_ids': torch.stack([enc['input_ids'].squeeze() for enc in neg_encodings]),
            'neg_mask': torch.stack([enc['attention_mask'].squeeze() for enc in neg_encodings]),
            'positive_ids': torch.tensor(list(query_doc.positive_ids))
        }


class DenseRetrieverBERT(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased", pooling: str = "mean"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == "mean":
            # Mean pooling of token embeddings
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                            min=1e-9)
        elif self.pooling == "cls":
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0]

        return F.normalize(embeddings, p=2, dim=1)  # L2 normalize embeddings

class MultiPositiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, query_emb: torch.Tensor, pos_emb: torch.Tensor,
                neg_emb: torch.Tensor) -> torch.Tensor:
        # Shape: [batch_size, 1, embed_dim]
        query_emb = query_emb.unsqueeze(1)

        # Shape: [batch_size, num_positives, embed_dim]
        pos_emb = pos_emb.view(query_emb.size(0), -1, query_emb.size(-1))

        # Shape: [batch_size, num_negatives, embed_dim]
        neg_emb = neg_emb.view(query_emb.size(0), -1, query_emb.size(-1))

        # Compute similarities with all positives
        pos_sim = torch.bmm(query_emb, pos_emb.transpose(1, 2)) / self.temperature

        # Compute similarities with all negatives
        neg_sim = torch.bmm(query_emb, neg_emb.transpose(1, 2)) / self.temperature

        # Concatenate positive and negative similarities
        all_sim = torch.cat([pos_sim, neg_sim], dim=2)

        # Create labels (positives are targets)
        labels = torch.zeros(all_sim.size(0), dtype=torch.long, device=all_sim.device)

        # Compute loss (takes max similarity among positives)
        loss = F.cross_entropy(all_sim.squeeze(1), labels)

        return loss


class MultiPositiveRerankerLoss(nn.Module):
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin

    def forward(self, query_emb: torch.Tensor, pos_emb: torch.Tensor,
                neg_emb: torch.Tensor) -> torch.Tensor:
        # Compute similarities with all positives
        pos_scores = torch.matmul(query_emb.unsqueeze(1), pos_emb.transpose(1, 2)).squeeze()

        # Compute similarities with all negatives
        neg_scores = torch.matmul(query_emb.unsqueeze(1), neg_emb.transpose(1, 2)).squeeze()

        # Compute loss: ensure all positives score higher than all negatives
        loss = torch.tensor(0.0, device=query_emb.device)
        for i in range(pos_scores.size(1)):
            for j in range(neg_scores.size(1)):
                current_loss = torch.relu(self.margin - pos_scores[:, i] + neg_scores[:, j])
                loss += current_loss.mean()

        return loss / (pos_scores.size(1) * neg_scores.size(1))


def train_enhanced_retriever(
        retriever: DenseRetrieverBERT,
        reranker: CrossAttentionReranker,
        train_dataloader: DataLoader,
        num_epochs: int,
        learning_rate: float,
        device: torch.device
) -> Tuple[DenseRetrieverBERT, CrossAttentionReranker]:
    retriever = retriever.to(device)
    reranker = reranker.to(device)

    retriever_optimizer = AdamW(retriever.parameters(), lr=learning_rate)
    reranker_optimizer = AdamW(reranker.parameters(), lr=learning_rate)

    retriever_criterion = MultiPositiveLoss()
    reranker_criterion = MultiPositiveRerankerLoss()

    for epoch in range(num_epochs):
        retriever.train()
        reranker.train()
        total_retriever_loss = 0
        total_reranker_loss = 0

        for batch in train_dataloader:
            # Move batch to device
            query_ids = batch['query_ids'].to(device)
            query_mask = batch['query_mask'].to(device)
            pos_ids = batch['pos_ids'].to(device)
            pos_mask = batch['pos_mask'].to(device)
            neg_ids = batch['neg_ids'].to(device)
            neg_mask = batch['neg_mask'].to(device)

            # First stage: Train retriever
            query_emb = retriever(query_ids, query_mask)
            pos_emb = retriever(pos_ids.view(-1, pos_ids.size(-1)),
                                pos_mask.view(-1, pos_mask.size(-1)))
            neg_emb = retriever(neg_ids.view(-1, neg_ids.size(-1)),
                                neg_mask.view(-1, neg_mask.size(-1)))

            retriever_loss = retriever_criterion(query_emb, pos_emb, neg_emb)

            retriever_optimizer.zero_grad()
            retriever_loss.backward()
            retriever_optimizer.step()

            # Second stage: Train reranker
            reranker_scores_pos = reranker(query_ids.unsqueeze(1), query_mask.unsqueeze(1), pos_ids, pos_mask)  # Unsqueeze query_ids and query_mask
            reranker_scores_neg = reranker(query_ids.unsqueeze(1), query_mask.unsqueeze(1), neg_ids, neg_mask)  # Unsqueeze query_ids and query_mask

            reranker_loss = reranker_criterion(
                query_emb,
                pos_emb.view(query_emb.size(0), -1, query_emb.size(-1)),
                neg_emb.view(query_emb.size(0), -1, query_emb.size(-1))
            )

            reranker_optimizer.zero_grad()
            reranker_loss.backward()
            reranker_optimizer.step()

            total_retriever_loss += retriever_loss.item()
            total_reranker_loss += reranker_loss.item()

        avg_retriever_loss = total_retriever_loss / len(train_dataloader.dataset)
        avg_reranker_loss = total_reranker_loss / len(train_dataloader.dataset)

        logging.info(f"Epoch {epoch + 1}/{num_epochs}")
        logging.info(f"Average Retriever Loss: {avg_retriever_loss:.4f}")
        logging.info(f"Average Reranker Loss: {avg_reranker_loss:.4f}")

    return retriever, reranker

def load_query_documents():

    queries, positive_docs, negative_docs = get_elements_train()

    q_num = 0
    pos_ctr = 0
    query_documents = []
    for query in queries:

        if query == 'No relevant document found.':
            continue

        doc_count = len(positive_docs[q_num])  # Change made here
        query_documents.append(
            QueryDocument(
                query_id=q_num,
                query_text=query,
                positive_docs=positive_docs[q_num],
                positive_ids=set(range(pos_ctr, pos_ctr + doc_count)),
                negative_docs=negative_docs[q_num]
            )
        )
        pos_ctr = pos_ctr + doc_count
        q_num += 1

    return query_documents


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    retriever = DenseRetrieverBERT(model_name="bert-base-uncased", pooling="mean") # Can be Cls also
    reranker = CrossAttentionReranker(model_name="bert-base-uncased")

    query_documents = load_query_documents()

    # Create dataset and dataloader
    dataset = MultiPositiveRetrieverDataset(query_documents, tokenizer)
    batch_size = 4  # Reduce from 8 to 4
    dataloader = DataLoader(dataset,
                            batch_size=batch_size,
                            num_workers=4,
                            pin_memory=True)

    # Train models
    trained_retriever, trained_reranker = train_enhanced_retriever(
        retriever=retriever,
        reranker=reranker,
        train_dataloader=dataloader,
        num_epochs=3,
        learning_rate=2e-5,
        device=device
    )
    model_path = 'saved_models_for_with_rerank'
    # Save models
    torch.save(trained_retriever.state_dict(), os.path.join(model_path, "multi_positive_rerank_retriever.pt"))
    torch.save(trained_reranker.state_dict(), os.path.join(model_path, "multi_positive_reranker.pt"))


if __name__ == "__main__":
    main()