from transformers import AutoTokenizer
import torch
import faiss
import numpy as np
from model_implementations.projectBert_1 import DenseRetriever
import os
import pytrec_eval
from collections import defaultdict
from data_prepare_files.data_preperar_for_faiss_and_validation import parse_files


use_already_exist_faiss = False
use_already_exist_doc_embeddings = False
faiss_path = os.path.join("../saved_model_normal_bert", "faiss_index.bin")
doc_embeedings_path = os.path.join("../saved_model_normal_bert", "document_embeddings.npy")

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    return [(score - min_score) / (max_score - min_score) for score in scores]
def encode_texts(texts, model, tokenizer, max_length=128, device="gpu", batch_size=16):
    """Generate dense embeddings for a list of texts."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    embeddings_list = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)

        # Remove 'token_type_ids' if it exists in the inputs
        inputs.pop('token_type_ids', None)  # Skip if not needed

        with torch.no_grad():
            model.to(device)  # Move model to GPU if available
            embeddings = model(**inputs)

            # Check if embeddings is a tuple or a single tensor
            if isinstance(embeddings, tuple):
                embeddings = embeddings[0]  # Get the first element if it's a tuple

        # Ensure embeddings has the expected shape
        if embeddings.dim() == 3:  # Shape (batch_size, sequence_length, hidden_size)
            embeddings_list.append(embeddings[:, 0, :].cpu().numpy())  # Use the first token's embeddings
        else:
            embeddings_list.append(embeddings.cpu().numpy())  # Handle the case where it's a 1D tensor

    return np.concatenate(embeddings_list, axis=0)  # Combine all batches

def compute_metrics(qrels, results):
    """
    Compute retrieval metrics using pytrec_eval
    Args:
        qrels: dict of dict, ground truth relevance scores {qid: {doc_id: relevance_score}}
        results: dict of dict, retrieval results {qid: {doc_id: score}}
    """
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {'map', 'ndcg_cut.10', 'P.10', 'recall'})
    metrics = evaluator.evaluate(results)
    
    mean_metrics = defaultdict(float)
    for query_metrics in metrics.values():
        for metric, score in query_metrics.items():
            mean_metrics[metric] += score
    
    num_queries = len(metrics)
    return {metric: score/num_queries for metric, score in mean_metrics.items()}

def evaluate_retrieval(queries, documents, qrels, model, tokenizer, device="gpu"):
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if use_already_exist_faiss :
        doc_texts = list(documents.values())
        doc_ids = list(documents.keys())

        index = faiss.read_index(faiss_path)
        
    else :
        # Generate document embeddings
        doc_texts = list(documents.values())
        doc_ids = list(documents.keys())

        if use_already_exist_doc_embeddings:
            doc_embeddings = np.load(doc_embeedings_path)
        else:
            doc_embeddings = encode_texts(doc_texts, model, tokenizer, device=device)
            # Save document embeddings to doc_embeedings_path
            np.save(doc_embeedings_path, doc_embeddings)

        # Create FAISS index
        dimension = doc_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)
        index.add(doc_embeddings)
        # {{ edit_1 }}: Save the FAISS index to a file
        faiss.write_index(index, faiss_path)

    # Prepare evaluation dictionaries
    results = {}
    results_score = {}
    # Perform search for each query
    k = 10  # Number of results to retrieve
    for qid, query_text in queries.items():
        query_emb = encode_texts([query_text], model, tokenizer, device=device)
        D, I = index.search(query_emb, k)

        D[0] = normalize_scores(D[0])
        # Format results for pytrec_eval
        query_results = {}
        query_results_score = {}
        for rank, (idx, dist) in enumerate(zip(I[0], D[0])):
            doc_id = doc_ids[idx]
            query_results[doc_id] = 1  # Convert distance to similarity score
            query_results_score[doc_id] = dist
        # Add missing documents with score 0
        #for doc_id in documents.keys():
        #    if doc_id not in query_results:
        #        query_results[doc_id] = 0.0
        results_score[qid] = query_results_score
        results[qid] = query_results

    # Debug: Print the structure of results
    print("Results structure:", results)  # {{ edit_1 }}: Added debug statement

    # Compute metrics
    metrics = compute_metrics(qrels, results)
    return metrics, results_score

if __name__ == "__main__":
    # Load model and tokenizer
    model_path = "../saved_model_normal_bert"
    model = DenseRetriever("bert-base-uncased")
    model.load_state_dict(torch.load(os.path.join(model_path, "best_dense_retriever.pt")))
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.eval()
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    queries, documents, qrels = parse_files()

    # Evaluate
    metrics, results = evaluate_retrieval(queries, documents, qrels, model, tokenizer, device)

    # Print results
    print("\nSearch Results:")
    for qid, query_text in queries.items():
        print(f"\nQuery {qid}: {query_text}")
        query_results = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:5]
        for doc_id, score in query_results:
            print(f"Document: {documents[doc_id]} (score: {score:.3f})")
    

    # Print evaluation results
    print("\nRetrieval Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


