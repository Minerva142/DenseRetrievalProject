import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel, BertTokenizer, AdamW
from typing import List, Dict, Tuple
import logging
from data_prepare_files.data_preparing_wot_rerank_multi_negative import prepare_training_data
import os

class DenseRetrievalDataset(Dataset):
    def __init__(self, queries: List[str], pos_docs: List[str], neg_docs: List[List[str]],
                 tokenizer: BertTokenizer, max_length: int = 128):
        """
        Dataset for dense retrieval training

        Args:
            queries: List of query strings
            pos_docs: List of positive document strings
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
        pos_doc = self.pos_docs[idx]
        neg_docs = self.neg_docs[idx]

        # Tokenize query
        query_tokens = self.tokenizer(
            query,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize positive document
        pos_doc_tokens = self.tokenizer(
            pos_doc,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Tokenize negative documents
        neg_docs_tokens = [
            self.tokenizer(
                neg_doc,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            ) for neg_doc in neg_docs
        ]

        return {
            'query_input_ids': query_tokens['input_ids'].squeeze(0),
            'query_attention_mask': query_tokens['attention_mask'].squeeze(0),
            'pos_doc_input_ids': pos_doc_tokens['input_ids'].squeeze(0),
            'pos_doc_attention_mask': pos_doc_tokens['attention_mask'].squeeze(0),
            'neg_docs_input_ids': torch.stack([t['input_ids'].squeeze(0) for t in neg_docs_tokens]),
            'neg_docs_attention_mask': torch.stack([t['attention_mask'].squeeze(0) for t in neg_docs_tokens])
        }


class BertDenseRetrieval(nn.Module):
    def __init__(self, model_name: str = 'bert-base-uncased', pooling: str = 'cls'):
        """
        BERT-based dense retrieval model

        Args:
            model_name: Name of the pretrained BERT model
            pooling: Pooling strategy ('cls' or 'mean')
        """
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.pooling = pooling

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        if self.pooling == 'cls':
            embeddings = outputs.last_hidden_state[:, 0]  # [CLS] token
        else:  # mean pooling
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1),
                                                                                            min=1e-9)

        return embeddings


class DenseRetrievalTrainer:
    def __init__(self,
                 model: BertDenseRetrieval,
                 tokenizer: BertTokenizer,
                 device: torch.device,
                 learning_rate: float = 2e-5,
                 temperature: float = 0.1):
        """
        Trainer for dense retrieval model

        Args:
            model: BERT dense retrieval model
            tokenizer: BERT tokenizer
            device: torch device
            learning_rate: Learning rate for optimization
            temperature: Temperature for similarity scaling
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)

    def compute_similarity(self, query_emb: torch.Tensor, doc_emb: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between query and document embeddings"""
        return torch.matmul(query_emb, doc_emb.transpose(0, 1)) / self.temperature

    def compute_loss(self, query_emb: torch.Tensor, pos_doc_emb: torch.Tensor,
                     neg_docs_emb: torch.Tensor) -> torch.Tensor:
        """Compute InfoNCE loss"""
        batch_size = query_emb.size(0)

        # Compute similarities
        pos_sim = self.compute_similarity(query_emb, pos_doc_emb)  # [batch_size, batch_size]
        neg_sim = self.compute_similarity(query_emb, neg_docs_emb.view(-1, neg_docs_emb.size(
            -1)))  # [batch_size, batch_size * num_neg]

        # Combine positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [batch_size, batch_size * (1 + num_neg)]

        # Create labels (positive pairs should have high similarity)
        labels = torch.arange(batch_size, device=self.device)

        # Compute cross entropy loss
        loss = nn.CrossEntropyLoss()(logits, labels)

        return loss

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Perform one training step"""
        self.model.train()

        # Move batch to device
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # Get embeddings
        query_emb = self.model(batch['query_input_ids'], batch['query_attention_mask'])
        pos_doc_emb = self.model(batch['pos_doc_input_ids'], batch['pos_doc_attention_mask'])

        neg_docs_shape = batch['neg_docs_input_ids'].shape
        neg_docs_emb = self.model(
            batch['neg_docs_input_ids'].view(-1, neg_docs_shape[-1]),
            batch['neg_docs_attention_mask'].view(-1, neg_docs_shape[-1])
        ).view(neg_docs_shape[0], neg_docs_shape[1], -1)

        # Compute loss
        loss = self.compute_loss(query_emb, pos_doc_emb, neg_docs_emb)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()


def train_dense_retrieval(
        queries: List[str],
        pos_docs: List[str],
        neg_docs: List[List[str]],
        model_name: str = 'bert-base-uncased',
        batch_size: int = 32,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        max_length: int = 128,
        temperature: float = 0.1
) -> Tuple[BertDenseRetrieval, BertTokenizer]:
    """
    Train BERT dense retrieval model

    Args:
        queries: List of query strings
        pos_docs: List of positive document strings
        neg_docs: List of lists of negative document strings
        model_name: Name of the pretrained BERT model
        batch_size: Batch size for training
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimization
        max_length: Maximum sequence length
        temperature: Temperature for similarity scaling

    Returns:
        Trained model and tokenizer
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertDenseRetrieval(model_name)

    # Create dataset and dataloader
    dataset = DenseRetrievalDataset(queries, pos_docs, neg_docs, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize trainer
    trainer = DenseRetrievalTrainer(
        model=model,
        tokenizer=tokenizer,
        device=device,
        learning_rate=learning_rate,
        temperature=temperature
    )

    # Training loop
    logging.info("Starting training...")
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            loss = trainer.train_step(batch)
            total_loss += loss

        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")

    return model, tokenizer

# add main part and model saving part

if __name__ == "__main__":
    queries, positive_docs, negative_docs_list = prepare_training_data()
    # Import prepared training data
    

    # Train the model
    model, tokenizer = train_dense_retrieval(
        queries=queries,
        pos_docs=positive_docs, 
        neg_docs=negative_docs_list,
        model_name='bert-base-uncased',
        batch_size=32,
        num_epochs=3,
        learning_rate=2e-5,
        max_length=128,
        temperature=0.1
    )

    model_path = '../saved_models_wot_with_rerank'
    # Save the trained model and tokenizer
    torch.save(model.state_dict(), os.path.join(model_path, 'dense_retrieval_model.pt'))
    tokenizer.save_pretrained(model_path)
