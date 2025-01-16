from transformers import BertModel, BertTokenizer
import torch
import faiss
import numpy as np
import pytrec_eval
from sklearn.preprocessing import normalize
def normalize_vectors(vectors):
    """
    Normalize vectors to unit length
    """
    return normalize(vectors, norm='l2')
def encode_text_helper(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def search_similar_documents(query,model, index, docnos,k=10):
    # Generate embedding for the query using SentenceTransformer
    query_embeddings = model.encode([query])
    # Normalize embeddings
    normalized_embeddings = normalize_vectors(query_embeddings)
    # Perform the search on the FAISS index
    distances, indices = index.search(normalized_embeddings, k)

    # Retrieve the corresponding document numbers and distances
    similar_docs = [(docnos[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return similar_docs