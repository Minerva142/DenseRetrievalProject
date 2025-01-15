import torch
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader, Dataset
from torch.nn import functional as F
import faiss


class DenseDataset(Dataset):
    def __init__(self, queries, documents, labels=None):
        self.queries = queries
        self.documents = documents
        self.labels = labels

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        if self.labels:
            return self.queries[idx], self.documents[idx], self.labels[idx]
        return self.queries[idx], self.documents[idx]


class DenseEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super(DenseEncoder, self).__init__()
        self.query_encoder = AutoModel.from_pretrained(model_name)
        self.doc_encoder = AutoModel.from_pretrained(model_name)

    def forward(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        query_outputs = self.query_encoder(input_ids=query_input_ids, attention_mask=query_attention_mask)
        doc_outputs = self.doc_encoder(input_ids=doc_input_ids, attention_mask=doc_attention_mask)

        query_embeddings = query_outputs.last_hidden_state[:, 0, :]  # CLS token for queries
        doc_embeddings = doc_outputs.last_hidden_state[:, 0, :]  # CLS token for documents

        return query_embeddings, doc_embeddings


def train_dual_encoder(queries, documents, model_name, epochs=3, batch_size=16):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = DenseEncoder(model_name)

    dataset = DenseDataset(queries, documents)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            query_texts, doc_texts = batch

            query_inputs = tokenizer(query_texts, return_tensors="pt", padding=True, truncation=True)
            doc_inputs = tokenizer(doc_texts, return_tensors="pt", padding=True, truncation=True)

            query_embeds, doc_embeds = model(
                query_inputs['input_ids'], query_inputs['attention_mask'],
                doc_inputs['input_ids'], doc_inputs['attention_mask']
            )

            scores = torch.matmul(query_embeds, doc_embeds.T)
            labels = torch.arange(len(query_texts)).to(scores.device)

            loss = F.cross_entropy(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return model, tokenizer


def build_faiss_index(doc_embeddings):
    dim = doc_embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    faiss.normalize_L2(doc_embeddings)  # Normalize for cosine similarity
    index.add(doc_embeddings)
    return index


class RankingModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(RankingModel, self).__init__()
        self.fc = torch.nn.Linear(input_dim * 2, 1)  # Concatenate query and document embeddings

    def forward(self, query_embeds, doc_embeds):
        combined = torch.cat((query_embeds, doc_embeds), dim=1)
        scores = self.fc(combined)
        return scores


def train_ranking_model(queries, documents, labels, dual_encoder, tokenizer, model_name, epochs=3, batch_size=16):
    ranking_model = RankingModel(dual_encoder.query_encoder.config.hidden_size)
    optimizer = torch.optim.AdamW(ranking_model.parameters(), lr=5e-5)

    dataset = DenseDataset(queries, documents, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        ranking_model.train()
        for batch in dataloader:
            query_texts, doc_texts, labels = batch

            query_inputs = tokenizer(query_texts, return_tensors="pt", padding=True, truncation=True)
            doc_inputs = tokenizer(doc_texts, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                query_embeds, doc_embeds = dual_encoder(
                    query_inputs['input_ids'], query_inputs['attention_mask'],
                    doc_inputs['input_ids'], doc_inputs['attention_mask']
                )

            scores = ranking_model(query_embeds, doc_embeds)
            scores = scores.squeeze(-1)

            loss = F.binary_cross_entropy_with_logits(scores, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

    return ranking_model


# Example Usage
#TODO edit there
queries = ["Titles as queries"]
documents = ["Texts as documents"]

# Some of them are need to be relevant, some of them are not
# if title relevant with text : 1 else : 0
# we should mix the dataset and add some false positive parts to it
labels = torch.tensor([[1], [0]])  # Example labels for relevancels

model_name = "meta-llama/Llama-3.2-1B"
dual_encoder, tokenizer = train_dual_encoder(queries, documents, model_name)

# Generate embeddings
tokenizer = AutoTokenizer.from_pretrained(model_name)
doc_inputs = tokenizer(documents, return_tensors="pt", padding=True, truncation=True)
query_inputs = tokenizer(queries, return_tensors="pt", padding=True, truncation=True)

with torch.no_grad():
    _, doc_embeddings = dual_encoder(None, None, doc_inputs['input_ids'], doc_inputs['attention_mask'])
    query_embeddings, _ = dual_encoder(query_inputs['input_ids'], query_inputs['attention_mask'], None, None)

# Build FAISS index
faiss_index = build_faiss_index(doc_embeddings.cpu().numpy())

# Perform retrieval
distance, indices = faiss_index.search(query_embeddings.cpu().numpy(), k=5)

# Rank results
ranking_model = train_ranking_model(queries, documents, labels, dual_encoder, tokenizer, model_name)
