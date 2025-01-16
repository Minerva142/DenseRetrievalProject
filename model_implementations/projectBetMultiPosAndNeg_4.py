import torch
from torch import nn
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import Dataset, DataLoader
from typing import List
import os
from data_prepare_files.data_preparing_multi_pos_neg_bert import get_elements_train

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class DenseRetrievalDataset(Dataset):
    def __init__(self, queries: List[str], positive_docs: List[List[str]], negative_docs: List[List[str]], tokenizer,
                 max_length=512):
        self.queries = queries
        self.positive_docs = positive_docs
        self.negative_docs = negative_docs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        pos_docs = self.positive_docs[idx]
        neg_docs = self.negative_docs[idx]

        # Tokenize query
        query_encoding = self.tokenizer(
            query,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # Tokenize positive and negative documents
        pos_encodings = [self.tokenizer(
            doc,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ) for doc in pos_docs]

        neg_encodings = [self.tokenizer(
            doc,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        ) for doc in neg_docs]

        # Get counts for positive and negative documents
        num_pos = len(pos_docs)
        num_neg = len(neg_docs)

        return {
            'query_input_ids': query_encoding['input_ids'].squeeze(0),
            'query_attention_mask': query_encoding['attention_mask'].squeeze(0),
            'positive_input_ids': torch.cat([enc['input_ids'] for enc in pos_encodings], dim=0),
            'negative_input_ids': torch.cat([enc['input_ids'] for enc in neg_encodings], dim=0),
            'num_pos': num_pos,
            'num_neg': num_neg
        }


def collate_fn(batch):
    max_pos = max(item['num_pos'] for item in batch)
    max_neg = max(item['num_neg'] for item in batch)

    query_input_ids = []
    query_attention_masks = []
    positive_input_ids = []
    negative_input_ids = []

    for item in batch:
        # Repeat query for each positive and negative document
        num_repeats = max(item['num_pos'], item['num_neg'])
        query_input_ids.extend([item['query_input_ids']] * num_repeats)
        query_attention_masks.extend([item['query_attention_mask']] * num_repeats)

        # Add positive and negative documents
        positive_input_ids.append(item['positive_input_ids'])
        negative_input_ids.append(item['negative_input_ids'])

    return {
        'query_input_ids': torch.stack(query_input_ids).to(device),
        'query_attention_masks': torch.stack(query_attention_masks).to(device),
        'positive_input_ids': torch.cat(positive_input_ids).to(device),
        'negative_input_ids': torch.cat(negative_input_ids).to(device)
    }


class BertDenseRetriever(nn.Module):
    def __init__(self, model_name: str = "bert-base-uncased"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.bert.to(device)  # Move model to GPU if available

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]

    def get_embeddings(self, text: str):
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}  # Move inputs to GPU
        with torch.no_grad():
            return self.forward(inputs['input_ids'], inputs['attention_mask'])


def train_model(model, train_dataloader, epochs=3, learning_rate=2e-5):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.TripletMarginLoss(margin=1.0).to(device)  # Move loss function to GPU

    for epoch in range(epochs):
        total_loss = 0
        batch_count = 0

        for batch in train_dataloader:
            query_input_ids = batch['query_input_ids']
            query_attention_masks = batch['query_attention_masks']
            positive_input_ids = batch['positive_input_ids']
            negative_input_ids = batch['negative_input_ids']

            # Get embeddings (inputs are already on GPU from collate_fn)
            query_emb = model(query_input_ids, query_attention_masks)
            pos_emb = model(positive_input_ids, torch.ones_like(positive_input_ids))
            neg_emb = model(negative_input_ids, torch.ones_like(negative_input_ids))

            # Ensure all tensors have the same first dimension
            min_size = min(query_emb.size(0), pos_emb.size(0), neg_emb.size(0))
            query_emb = query_emb[:min_size]
            pos_emb = pos_emb[:min_size]
            neg_emb = neg_emb[:min_size]

            loss = criterion(query_emb, pos_emb, neg_emb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            batch_count += 1

        avg_loss = total_loss / batch_count
        print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    # Enable CUDA optimizations
    torch.backends.cudnn.benchmark = True

    use_desc = False
    queries, positive_docs, negative_docs_list = get_elements_train(use_desc)

    model = BertDenseRetriever()
    model.to(device)  # Ensure model is on GPU

    dataset = DenseRetrievalDataset(queries, positive_docs, negative_docs_list, model.tokenizer, max_length=512)
    train_dataloader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    train_model(model, train_dataloader)

    output_dir = "../multi_pos_and_neg"
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "final_dense_retriever.pt"))
    model.tokenizer.save_pretrained(output_dir)