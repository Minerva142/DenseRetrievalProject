import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel,
    AutoTokenizer,
    AdamW,
    get_linear_schedule_with_warmup
)
from typing import List, Dict, Set, Tuple
import logging
import random
from dataclasses import dataclass
import numpy as np

from data_prepare_files.data_preparing_rerank_bert import get_elements_train

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryDocument:
    query_id: int
    query_text: str
    positive_docs: List[str]
    positive_ids: Set[int]
    negative_docs: List[str]


class DualEncoder(nn.Module):
    def __init__(self, model_name: str = "microsoft/mpnet-base"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.projection = nn.Linear(self.encoder.config.hidden_size, 768)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        return F.normalize(self.projection(embeddings), p=2, dim=1)

    @staticmethod
    def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class CrossEncoder(nn.Module):
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0]
        scores = self.classifier(cls_output)
        return scores


class RetrievalDataset(Dataset):
    def __init__(
            self,
            queries: List[str],
            positive_docs: List[List[str]],
            negative_docs: List[List[str]],
            tokenizer: AutoTokenizer,
            max_length: int = 128,
            max_positives: int = 3,
            max_negatives: int = 5
    ):
        self.queries = queries
        self.positive_docs = positive_docs
        self.negative_docs = negative_docs
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_positives = max_positives
        self.max_negatives = max_negatives

    def __len__(self) -> int:
        return len(self.queries)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        query = self.queries[idx]

        # Sample positive and negative documents
        pos_docs = random.sample(
            self.positive_docs[idx],
            min(len(self.positive_docs[idx]), self.max_positives)
        )
        neg_docs = random.sample(
            self.negative_docs[idx],
            min(len(self.negative_docs[idx]), self.max_negatives)
        )

        # Pad if necessary
        while len(pos_docs) < self.max_positives and pos_docs:
            pos_docs.append(pos_docs[0])
        while len(neg_docs) < self.max_negatives and neg_docs:
            neg_docs.append(neg_docs[0])

        # Tokenize query
        query_encodings = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize documents
        pos_encodings = [
            self.tokenizer(
                doc,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for doc in pos_docs
        ]

        neg_encodings = [
            self.tokenizer(
                doc,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for doc in neg_docs
        ]

        return {
            'query_ids': query_encodings['input_ids'].squeeze(0),
            'query_mask': query_encodings['attention_mask'].squeeze(0),
            'pos_ids': torch.stack([enc['input_ids'].squeeze(0) for enc in pos_encodings]),
            'pos_mask': torch.stack([enc['attention_mask'].squeeze(0) for enc in pos_encodings]),
            'neg_ids': torch.stack([enc['input_ids'].squeeze(0) for enc in neg_encodings]),
            'neg_mask': torch.stack([enc['attention_mask'].squeeze(0) for enc in neg_encodings]),
            'query_text': query,
            'pos_docs': pos_docs,
            'neg_docs': neg_docs
        }


class DualEncoderLoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super().__init__()
        self.temperature = temperature

    def forward(
            self,
            query_embeddings: torch.Tensor,
            positive_embeddings: torch.Tensor,
            negative_embeddings: torch.Tensor
    ) -> torch.Tensor:
        # Calculate similarities
        pos_similarities = torch.matmul(
            query_embeddings.unsqueeze(1),
            positive_embeddings.transpose(1, 2)
        ).squeeze(1) / self.temperature

        neg_similarities = torch.matmul(
            query_embeddings.unsqueeze(1),
            negative_embeddings.transpose(1, 2)
        ).squeeze(1) / self.temperature

        # Combine similarities and create labels
        similarities = torch.cat([pos_similarities, neg_similarities], dim=1)
        labels = torch.zeros(similarities.size(0), dtype=torch.long, device=similarities.device)

        # Calculate loss
        loss = F.cross_entropy(similarities, labels)
        return loss


def train_model(
        dual_encoder: DualEncoder,
        cross_encoder: CrossEncoder,
        train_dataloader: DataLoader,
        tokenizer: AutoTokenizer,  # Added tokenizer parameter
        device: torch.device,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        warmup_steps: int = 1000
) -> Tuple[DualEncoder, CrossEncoder]:
    # Optimizers and schedulers
    dual_optimizer = AdamW(dual_encoder.parameters(), lr=learning_rate)
    cross_optimizer = AdamW(cross_encoder.parameters(), lr=learning_rate)

    total_steps = len(train_dataloader) * num_epochs
    dual_scheduler = get_linear_schedule_with_warmup(
        dual_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    cross_scheduler = get_linear_schedule_with_warmup(
        cross_optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # Loss functions
    dual_criterion = DualEncoderLoss()
    cross_criterion = nn.BCEWithLogitsLoss()

    for epoch in range(num_epochs):
        dual_encoder.train()
        cross_encoder.train()
        total_loss = 0

        for step, batch in enumerate(train_dataloader):
            # Move batch to device
            query_ids = batch['query_ids'].to(device)
            query_mask = batch['query_mask'].to(device)
            pos_ids = batch['pos_ids'].to(device)
            pos_mask = batch['pos_mask'].to(device)
            neg_ids = batch['neg_ids'].to(device)
            neg_mask = batch['neg_mask'].to(device)

            # Dual encoder forward pass
            query_emb = dual_encoder.encode(query_ids, query_mask)
            pos_emb = dual_encoder.encode(
                pos_ids.view(-1, pos_ids.size(-1)),
                pos_mask.view(-1, pos_mask.size(-1))
            ).view(pos_ids.size(0), pos_ids.size(1), -1)
            neg_emb = dual_encoder.encode(
                neg_ids.view(-1, neg_ids.size(-1)),
                neg_mask.view(-1, neg_mask.size(-1))
            ).view(neg_ids.size(0), neg_ids.size(1), -1)

            # Dual encoder loss
            dual_loss = dual_criterion(query_emb, pos_emb, neg_emb)

            # Cross encoder forward pass
            # Prepare positive pairs
            pos_pairs = []
            for query, pos_doc in zip(batch['query_text'], batch['pos_docs']):
                pos_pairs.extend([(query, doc) for doc in pos_doc])

            # Prepare negative pairs
            neg_pairs = []
            for query, neg_doc in zip(batch['query_text'], batch['neg_docs']):
                neg_pairs.extend([(query, doc) for doc in neg_doc])

            # Tokenize all pairs
            pos_inputs = tokenizer(
                pos_pairs,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            neg_inputs = tokenizer(
                neg_pairs,
                max_length=256,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Move inputs to device
            pos_input_ids = pos_inputs['input_ids'].to(device)
            pos_attention_mask = pos_inputs['attention_mask'].to(device)
            neg_input_ids = neg_inputs['input_ids'].to(device)
            neg_attention_mask = neg_inputs['attention_mask'].to(device)

            # Get scores
            pos_scores = cross_encoder(pos_input_ids, pos_attention_mask)
            neg_scores = cross_encoder(neg_input_ids, neg_attention_mask)

            # Cross encoder loss
            pos_labels = torch.ones_like(pos_scores)
            neg_labels = torch.zeros_like(neg_scores)
            cross_loss = cross_criterion(
                torch.cat([pos_scores, neg_scores]),
                torch.cat([pos_labels, neg_labels])
            )

            # Combined loss and backward pass
            loss = dual_loss + cross_loss
            loss.backward()

            # Optimize
            dual_optimizer.step()
            cross_optimizer.step()
            dual_scheduler.step()
            cross_scheduler.step()

            # Zero gradients
            dual_optimizer.zero_grad()
            cross_optimizer.zero_grad()

            total_loss += loss.item()

            if step % 100 == 0:
                logger.info(
                    f"Epoch: {epoch}, Step: {step}, "
                    f"Loss: {loss.item():.4f}, "
                    f"Dual Loss: {dual_loss.item():.4f}, "
                    f"Cross Loss: {cross_loss.item():.4f}"
                )

        avg_loss = total_loss / len(train_dataloader)
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")

    return dual_encoder, cross_encoder


def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    queries, positive_docs, negative_docs = get_elements_train()

    # Initialize tokenizer and models
    tokenizer = AutoTokenizer.from_pretrained("microsoft/mpnet-base")
    dual_encoder = DualEncoder().to(device)
    cross_encoder = CrossEncoder().to(device)

    # Create dataset and dataloader
    dataset = RetrievalDataset(
        queries=queries,
        positive_docs=positive_docs,
        negative_docs=negative_docs,
        tokenizer=tokenizer
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # Train models
    trained_dual_encoder, trained_cross_encoder = train_model(
        dual_encoder=dual_encoder,
        cross_encoder=cross_encoder,
        train_dataloader=dataloader,
        tokenizer=tokenizer,  # Added tokenizer parameter
        device=device
    )

    # Save models
    #os.makedirs('saved_models', exist_ok=True)
    torch.save(
        trained_dual_encoder.state_dict(),
        '../saved_models_for_with_rerank/dual_encoder.pt'
    )
    torch.save(
        trained_cross_encoder.state_dict(),
        '../saved_models_for_with_rerank/cross_encoder.pt'
    )


if __name__ == "__main__":
    main()