from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np

# Load the trained model and tokenizer
model = AutoModel.from_pretrained("dense_retriever_model")
tokenizer = AutoTokenizer.from_pretrained("dense_retriever_tokenizer")
model.eval()

def encode_texts(texts, model, tokenizer, max_length=128, device="cpu"):
    """Generate dense embeddings for a list of texts."""
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(device)
    with torch.no_grad():
        embeddings = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return embeddings.last_hidden_state[:, 0, :].cpu().numpy()

# Example documents
documents = [
    "Document 1 content goes here.",
    "Document 2 content goes here.",
    "Document 3 content goes here."
]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate embeddings for documents
doc_embeddings = encode_texts(documents, model, tokenizer, device=device)