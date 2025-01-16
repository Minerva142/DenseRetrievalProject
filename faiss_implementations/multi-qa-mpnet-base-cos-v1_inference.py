import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import json
import pytrec_eval
from sklearn.preprocessing import normalize


def normalize_vectors(vectors):
    """
    Normalize vectors to unit length
    """
    return normalize(vectors, norm='l2')

    # Function to perform similarity search
def search_similar_documents(query, k=5):
    # Generate embedding for the query using SentenceTransformer
    query_embeddings = model.encode([query])
    # Normalize embeddings
    normalized_embeddings = normalize_vectors(query_embeddings)
    # Perform the search on the FAISS index
    distances, indices = index.search(normalized_embeddings, k)
    
    # Retrieve the corresponding document numbers and distances
    similar_docs = [(docnos[idx], distances[0][i]) for i, idx in enumerate(indices[0])]
    return similar_docs

# Load the CSV file containing document embeddings
df = pd.read_csv('../multi-qa-mpnet-base-cos-v1_FT_embeddings.csv')

# Assuming the first column is DOCNO and the rest are embeddings
docnos = df['DOCNO'].values  # Extract document numbers
embeddings = df.drop(columns=['DOCNO']).values

# Initialize the model 
model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')


dim = embeddings.shape[1]

normalized_embeddings = normalize_vectors(embeddings)
index = faiss.IndexFlatIP(dim)
index.add(normalized_embeddings)

faiss.write_index(index, '../saved_model_multi-qa-mpnet-base-cos-v1/faiss_index.bin')



# Example usage of the similarity search function
query_text = "Apple stock rise"
similar_documents = search_similar_documents(query_text, k=5)

# Print results
for docno, distance in similar_documents:
    print(f'Document No: {docno}, Distance: {distance}')
