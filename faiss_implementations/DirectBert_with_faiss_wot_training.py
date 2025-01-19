from transformers import BertModel, BertTokenizer
import torch
import faiss
import numpy as np
import pytrec_eval
from data_prepare_files.data_preperar_for_faiss_and_validation import parse_files
from sklearn.preprocessing import normalize
import os

# Initialize BERT model and tokenizer
model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)
use_already_exists_emb = False
use_already_exists_faiss = False
model_path = '../saved_models_for_wot_train_bert'
path_of_emb = os.path.join(model_path, 'doc_embeddings_standart_tokenizer.npy')
path_of_faiss = os.path.join(model_path, 'doc_faiss_wot_train_bert.bin')

if not os.path.exists(model_path):
    os.makedirs(model_path)

# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
model = model.to(device)  # Move model to GPU

def normalize_vectors(vectors):
    """
    Normalize vectors to unit length
    """
    return normalize(vectors, norm='l2')
# Function to encode text using BERT
def encode_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    # Move input tensors to the same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()


def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]

use_desc = False
queries, documents, qrels = parse_files(use_desc)
print(len(documents.keys()))
print("data readed")

if use_already_exists_emb:
    doc_vectors = np.load(path_of_emb)
    doc_embeddings = {}
    doc_keys = list(documents.keys())  # Convert dict_keys to a list
    for i, doc_vector in enumerate(doc_vectors):
        doc_embeddings[doc_keys[i]] = doc_vector
else:
    doc_embeddings = {}
    for doc_id, text in documents.items():
        print(doc_id + " : added")
        doc_embeddings[doc_id] = encode_text(text)

# Create FAISS index
# Add document embeddings to the index
doc_ids = list(doc_embeddings.keys())

if use_already_exists_faiss:
    index = faiss.read_index(path_of_faiss)
else:
    dimension = len(next(iter(doc_embeddings.values())))
    # Use GPU for FAISS if available
    #res = faiss.StandardGpuResources() if torch.cuda.is_available() else None

    if torch.cuda.is_available():
        index = faiss.IndexFlatIP(dimension)
    #    index = faiss.index_cpu_to_gpu(res, 0, index)
    else:
        index = faiss.IndexFlatIP(dimension)

    doc_vectors = np.array(list(doc_embeddings.values()))
    doc_vectors = normalize_vectors(doc_vectors)
    index.add(doc_vectors)

    # Convert back to CPU for saving if necessary
    #if torch.cuda.is_available():
    #    index = faiss.index_gpu_to_cpu(index)
    faiss.write_index(index, path_of_faiss)

if not use_already_exists_emb:
    doc_vectors = np.array(list(doc_embeddings.values()))
    np.save(path_of_emb, doc_vectors)


query_embeddings = {query_id: encode_text(text) for query_id, text in queries.items()}

query_vectors = np.array(list(query_embeddings.values()))
query_keys = list(query_embeddings.keys())
query_vectors = normalize_vectors(query_vectors)
# If using GPU, convert index back to GPU for search
#if torch.cuda.is_available() and not use_already_exists_faiss:
    #res = faiss.StandardGpuResources()
    #index = faiss.index_cpu_to_gpu(res, 0, index)

retrieved_docs = {}
for query_id, query_vector in zip(query_keys,query_vectors):
    dists, indices = index.search(np.array([query_vector]), k=10)
    dists = normalize_scores(dists[0])
    # Create a dictionary with document IDs and a default relevance score
    dict = {}
    j = 0
    for i in indices[0]:
        dict[doc_ids[i]] = 1  # dists[j]
        j += 1
    retrieved_docs[query_id] = dict

#print(retrieved_docs)

# Evaluate using pytrec_eval
evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut.10', 'recall', 'P.10', 'P.5'})
results = evaluator.evaluate(retrieved_docs)

metric_sums = {}
for query_id, metrics in results.items():
    for metric, value in metrics.items():
        metric_sums[metric] = 0.0
    break

print(metric_sums)
query_count = len(results)

# Print evaluation results
for query_id, metrics in results.items():
    print(f"Query: {query_id}")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
        metric_sums[metric] += value

# Calculate and print means
print("\nMean Metrics:")
for metric, total in metric_sums.items():
    mean_value = total / query_count if query_count > 0 else 0
    print(f"  {metric}: {mean_value:.4f}")
