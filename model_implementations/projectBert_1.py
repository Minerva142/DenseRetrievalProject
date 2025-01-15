import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, AdamW
from tqdm import tqdm
from data_prepare_files.data_preparing_normal_bert import get_elements_train
import os

# validation already added
class DenseRetrievalDataset(Dataset):
    def __init__(self, queries, positive_docs, negative_docs, tokenizer, max_length=128):
        self.queries = queries
        self.positive_docs = positive_docs
        self.negative_docs = negative_docs
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        query = self.queries[idx]
        pos_doc = self.positive_docs[idx]
        neg_doc = self.negative_docs[idx] if self.negative_docs else None

        query_enc = self.tokenizer(
            query, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length
        )
        pos_enc = self.tokenizer(
            pos_doc, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length
        )
        if neg_doc:
            neg_enc = self.tokenizer(
                neg_doc, return_tensors="pt", truncation=True, padding="max_length", max_length=self.max_length
            )
        else:
            neg_enc = None

        return query_enc, pos_enc, neg_enc


class DenseRetriever(nn.Module):
    def __init__(self, model_name="bert-base-uncased"):
        super(DenseRetriever, self).__init__()
        self.encoder = AutoModel.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]


def train_dense_retriever(model, dataloader, optimizer, device, epochs=3, temperature=0.05):
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()

            query_enc, pos_enc, neg_enc = batch
            query_input_ids = query_enc["input_ids"].squeeze(1).to(device)
            query_attention_mask = query_enc["attention_mask"].squeeze(1).to(device)

            pos_input_ids = pos_enc["input_ids"].squeeze(1).to(device)
            pos_attention_mask = pos_enc["attention_mask"].squeeze(1).to(device)

            query_emb = model(query_input_ids, query_attention_mask)
            pos_emb = model(pos_input_ids, pos_attention_mask)

            if neg_enc is not None:
                neg_input_ids = neg_enc["input_ids"].squeeze(1).to(device)
                neg_attention_mask = neg_enc["attention_mask"].squeeze(1).to(device)
                neg_emb = model(neg_input_ids, neg_attention_mask)
                embeddings = torch.cat([pos_emb, neg_emb], dim=0)
            else:
                embeddings = pos_emb

            similarities = torch.matmul(query_emb, embeddings.T) / temperature
            labels = torch.arange(len(query_emb)).to(device)

            loss = criterion(similarities, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1} Loss: {epoch_loss / len(dataloader):.4f}")
    
    return epoch_loss / len(dataloader)

if __name__ == "__main__":
    queries, positive_docs, negative_docs = get_elements_train()

    model_name = "bert-base-uncased"
    max_length = 128
    batch_size = 2
    learning_rate = 2e-5
    epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    dataset = DenseRetrievalDataset(queries, positive_docs, negative_docs, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = DenseRetriever(model_name).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)

    output_dir = "../saved_model_normal_bert"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    best_loss = float('inf')
    for epoch in range(epochs):
        avg_loss = train_dense_retriever(model, dataloader, optimizer, device, epochs=1)
        
        # Save model if it achieves better loss
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "best_dense_retriever.pt"))
            tokenizer.save_pretrained(output_dir)
            print(f"Saved model and tokenizer with loss: {best_loss:.4f}")

    # Save final model and tokenizer
    torch.save(model.state_dict(), os.path.join(output_dir, "final_dense_retriever.pt"))
    tokenizer.save_pretrained(output_dir)
    print("Saved final model and tokenizer")