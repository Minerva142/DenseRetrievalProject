import faiss

from data_prepare_files.data_preperar_for_faiss_and_validation import parse_files
from model_implementations.projectBetMultiPosAndNeg_4 import BertDenseRetriever
import torch
from transformers import AutoTokenizer
import numpy as np
from typing import List
import os
import pytrec_eval

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

use_already_exist_faiss = False
use_already_exist_doc_embeddings = False
model_path = "../multi_pos_and_neg"
faiss_path = os.path.join(model_path, "faiss_index.bin")
doc_embeedings_path = os.path.join(model_path, "document_embeddings.npy")

def normalize_scores(scores):
    min_score = min(scores)
    max_score = max(scores)
    if min_score == max_score:
        return scores
    return [(score - min_score) / (max_score - min_score) for score in scores]
class DenseRetrieverIndex:
    def __init__(self, model_path: str = None):
        self.retriever = BertDenseRetriever()
        if model_path:
            self.retriever.load_state_dict(torch.load(model_path))
        self.retriever.to(device)  # Move model to device
        self.retriever.eval()

        self.index = None
        self.docno_to_idx = {}  # Map docno to index
        self.idx_to_docno = {}  # Map index to docno
        self.documents = {}  # Store documents as {docno: text}

    def build_index(self, documents: dict):
        self.documents = documents
        # Generate embeddings for all documents
        if use_already_exist_doc_embeddings:
            embeddings = np.load(doc_embeedings_path)
        else:
            embeddings = []

            # Create mapping between docno and index
            for idx, (docno, text) in enumerate(documents.items()):
                self.docno_to_idx[docno] = idx
                self.idx_to_docno[idx] = docno

                # Get embeddings using the existing get_embeddings method
                emb = self.retriever.get_embeddings(text).cpu().numpy()
                embeddings.append(emb)

            embeddings = np.vstack(embeddings)
            np.save(doc_embeedings_path, embeddings)

        # Build FAISS index
        dimension = embeddings.shape[1]
        if use_already_exist_faiss:
            self.index = faiss.read_index(faiss_path)
        else:
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            faiss.write_index(self.index, faiss_path)

    def search(self, query: str, k: int = 10) -> tuple[dict, dict]:
        # Generate query embedding
        query_emb = self.retriever.get_embeddings(query).cpu().numpy()

        # Search in FAISS index
        scores, indices = self.index.search(query_emb, k)
        scores[0] = normalize_scores(scores[0])
        # Return results in a format compatible with pytrec_eval
        results = {}
        results_score = {}
        for score, idx in zip(scores[0], indices[0]):
            docno = self.idx_to_docno[idx]
            results_score[docno] = float(score)
            results[docno] = 1

        return results, results_score

    def evaluate(self, queries: dict, qrels: dict, metrics: List[str] = None) -> dict:
        """
        Evaluate retrieval performance using pytrec_eval

        Args:
            queries: dict of {query_id: query_text}
            qrels: dict of {query_id: {doc_id: relevance_score}}
            metrics: list of metrics to compute

        Returns:
            dict of evaluation results
        """

        # ['ndcg', 'map', 'recall', 'P.5']
        if metrics is None:
            # Use standard trec_eval metrics
            metrics = {
                'map',                  # Mean Average Precision
                'ndcg',                 # Normalized Discounted Cumulative Gain
                'ndcg_cut_5',          # NDCG at cutoff 5
                'ndcg_cut_10',         # NDCG at cutoff 10
                'P_5',                  # Precision at 5
                'P_10',                 # Precision at 10
                'recall_5',             # Recall at 5
                'recall_10'             # Recall at 10
            }
        # Initialize evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, metrics)

        # Get search results for all queries
        run = {}
        run_score = {}
        for query_id, query_text in queries.items():
            results, results_score = self.search(query_text)
            run[query_id] = results
            run_score[query_id] = results_score

        # Compute metrics
        results = evaluator.evaluate(run)

        # Average metrics across queries
        mean_results = {}
        for metric in metrics:
            metric_values = [query_results[metric] for query_results in results.values()]
            mean_results[metric] = sum(metric_values) / len(metric_values)

        return mean_results


if __name__ == "__main__":
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # Initialize index
    retriever_index = DenseRetrieverIndex(os.path.join(model_path, "final_dense_retriever.pt"))
    retriever_index.retriever.tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load data
    queries, documents, qrels = parse_files()

    # Build index
    retriever_index.build_index(documents)

    # Search example
    query = "What is dense retrieval?"
    results, results_score = retriever_index.search(query, k=10)

    print(f"\nQuery: {query}")
    print("\nResults:")
    for docno, score in results_score.items():
        print(f"Document: {documents[docno][:100]}... (score: {score:.3f})")

    # Evaluate
    evaluation_results = retriever_index.evaluate(queries, qrels)
    print("\nEvaluation Results:")
    for metric, value in evaluation_results.items():
        print(f"{metric}: {value:.3f}")