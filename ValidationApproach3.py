import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
import numpy as np
from typing import List, Dict, Tuple
import logging
from collections import defaultdict
from tqdm import tqdm
from BertDenseRetrieverMultiWotReRank import BertDenseRetrieval

# [Previous DenseRetrievalDataset, BertDenseRetrieval, and other classes remain the same...]

class ValidationDataset(Dataset):
    def __init__(self, queries: List[str], pos_docs: List[List[str]],
                 neg_docs: List[List[str]], tokenizer: BertTokenizer, max_length: int = 128):
        """
        Dataset for validation with multiple positive documents per query

        Args:
            queries: List of query strings
            pos_docs: List of lists of positive document strings
            neg_docs: List of lists of negative document strings
            tokenizer: BERT tokenizer
            max_length: Maximum sequence length
        """
        self.queries = queries
        self.pos_docs = pos_docs
        self.neg_docs = neg_docs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query = self.queries[idx]
        pos_docs = self.pos_docs[idx]
        neg_docs = self.neg_docs[idx]

        # Tokenize query
        query_tokens = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize all documents (positive and negative)
        all_docs = pos_docs + neg_docs
        docs_tokens = [
            self.tokenizer(
                doc,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for doc in all_docs
        ]

        return {
            'query_input_ids': query_tokens['input_ids'].squeeze(0),
            'query_attention_mask': query_tokens['attention_mask'].squeeze(0),
            'docs_input_ids': torch.stack([t['input_ids'].squeeze(0) for t in docs_tokens]),
            'docs_attention_mask': torch.stack([t['attention_mask'].squeeze(0) for t in docs_tokens]),
            'num_pos_docs': len(pos_docs)
        }


class RetrievalMetrics:
    @staticmethod
    def mean_reciprocal_rank(rankings: List[List[int]], num_pos_docs: List[int]) -> float:
        """Calculate MRR for a batch of rankings"""
        mrr = 0.0
        for ranking, n_pos in zip(rankings, num_pos_docs):
            for rank, idx in enumerate(ranking, 1):
                if idx < n_pos:
                    mrr += 1.0 / rank
                    break
        return mrr / len(rankings)

    @staticmethod
    def recall_at_k(rankings: List[List[int]], num_pos_docs: List[int], k: int) -> float:
        """Calculate Recall@K for a batch of rankings"""
        recall = 0.0
        for ranking, n_pos in zip(rankings, num_pos_docs):
            hits = sum(1 for idx in ranking[:k] if idx < n_pos)
            recall += hits / n_pos
        return recall / len(rankings)

    @staticmethod
    def ndcg_at_k(rankings: List[List[int]], num_pos_docs: List[int], k: int) -> float:
        """Calculate NDCG@K for a batch of rankings"""

        def dcg(ranking: List[int], n_pos: int, k: int) -> float:
            score = 0.0
            for i, idx in enumerate(ranking[:k], 1):
                if idx < n_pos:
                    score += 1.0 / np.log2(i + 1)
            return score

        def idcg(n_pos: int, k: int) -> float:
            k = min(n_pos, k)
            return sum(1.0 / np.log2(i + 1) for i in range(1, k + 1))

        ndcg = 0.0
        for ranking, n_pos in zip(rankings, num_pos_docs):
            ideal_dcg = idcg(n_pos, k)
            if ideal_dcg > 0:
                ndcg += dcg(ranking, n_pos, k) / ideal_dcg
        return ndcg / len(rankings)


class DenseRetrievalValidator:
    def __init__(self, model: BertDenseRetrieval, device: torch.device):
        self.model = model
        self.device = device
        self.metrics = RetrievalMetrics()

    @torch.no_grad()
    def validate(self, dataloader: DataLoader, k_values: List[int] = [1, 5, 10]) -> Dict[str, float]:
        """
        Validate the model on a validation dataset

        Args:
            dataloader: Validation data loader
            k_values: List of K values for Recall@K and NDCG@K

        Returns:
            Dictionary of metric names and values
        """
        self.model.eval()
        all_rankings = []
        all_num_pos_docs = []

        for batch in tqdm(dataloader, desc="Validating"):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            num_pos_docs = batch.pop('num_pos_docs')

            # Get embeddings
            query_emb = self.model(batch['query_input_ids'], batch['query_attention_mask'])

            docs_shape = batch['docs_input_ids'].shape
            docs_emb = self.model(
                batch['docs_input_ids'].view(-1, docs_shape[-1]),
                batch['docs_attention_mask'].view(-1, docs_shape[-1])
            ).view(docs_shape[0], docs_shape[1], -1)

            # Compute similarities and rankings
            similarities = torch.matmul(query_emb.unsqueeze(1), docs_emb.transpose(1, 2)).squeeze(1)
            rankings = torch.argsort(similarities, dim=1, descending=True).cpu().numpy()

            all_rankings.extend(rankings.tolist())
            all_num_pos_docs.extend(num_pos_docs.tolist())

        # Calculate metrics
        metrics = {
            'mrr': self.metrics.mean_reciprocal_rank(all_rankings, all_num_pos_docs)
        }

        for k in k_values:
            metrics[f'recall@{k}'] = self.metrics.recall_at_k(all_rankings, all_num_pos_docs, k)
            metrics[f'ndcg@{k}'] = self.metrics.ndcg_at_k(all_rankings, all_num_pos_docs, k)

        return metrics


def validate_dense_retrieval(
        model: BertDenseRetrieval,
        tokenizer: BertTokenizer,
        val_queries: List[str],
        val_pos_docs: List[List[str]],
        val_neg_docs: List[List[str]],
        batch_size: int = 32,
        max_length: int = 128,
        k_values: List[int] = [1, 5, 10]
) -> Dict[str, float]:
    """
    Validate dense retrieval model

    Args:
        model: Trained BERT dense retrieval model
        tokenizer: BERT tokenizer
        val_queries: List of validation query strings
        val_pos_docs: List of lists of positive document strings for each query
        val_neg_docs: List of lists of negative document strings for each query
        batch_size: Batch size for validation
        max_length: Maximum sequence length
        k_values: List of K values for Recall@K and NDCG@K

    Returns:
        Dictionary of metric names and values
    """
    device = next(model.parameters()).device

    # Create validation dataset and dataloader
    val_dataset = ValidationDataset(val_queries, val_pos_docs, val_neg_docs, tokenizer, max_length)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Initialize validator
    validator = DenseRetrievalValidator(model, device)

    # Run validation
    metrics = validator.validate(val_dataloader, k_values)

    # Log results
    for metric_name, value in metrics.items():
        logging.info(f"{metric_name}: {value:.4f}")

    return metrics

# add main part