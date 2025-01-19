from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np
import os
import pytrec_eval
from collections import defaultdict
from typing import Dict, List, Tuple
from data_prepare_files.data_preperar_for_faiss_and_validation import parse_files

class SBERTRetriever:
    def __init__(
            self,
            model_name: str = 'all-MiniLM-L6-v2',
            index_path: str = '../saved_model_SBERT/faiss_index.bin',
            embeddings_path: str = '../saved_model_SBERT/document_embeddings.npy'
    ):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.index = None

    def normalize_scores(self, scores: List[float]) -> List[float]:
        """Normalize scores to range [0,1]"""
        min_score = min(scores)
        max_score = max(scores)
        if max_score == min_score:
            return [1.0] * len(scores)
        return [(score - min_score) / (max_score - min_score) for score in scores]

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts using SBERT"""
        return self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    def build_index(self, documents: Dict[str, str], use_existing: bool = False) -> Tuple[faiss.Index, List[str]]:
        """Build or load FAISS index for documents"""
        doc_texts = list(documents.values())
        doc_ids = list(documents.keys())

        if use_existing and os.path.exists(self.index_path):
            print("Loading existing FAISS index...")
            self.index = faiss.read_index(self.index_path)
            return self.index, doc_ids

        print("Creating new document embeddings...")
        if use_existing and os.path.exists(self.embeddings_path):
            doc_embeddings = np.load(self.embeddings_path)
        else:
            doc_embeddings = self.encode_texts(doc_texts)
            os.makedirs(os.path.dirname(self.embeddings_path), exist_ok=True)
            np.save(self.embeddings_path, doc_embeddings)

        print("Building FAISS index...")
        dimension = doc_embeddings.shape[1]

        doc_embeddings = doc_embeddings / np.linalg.norm(doc_embeddings, axis=1, keepdims=True)  # {{ edit_1 }}
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(doc_embeddings)

        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        faiss.write_index(self.index, self.index_path)
        return self.index, doc_ids

    def search(self, query: str, doc_ids: List[str], k: int = 10) -> Tuple[List[str], List[float]]:
        """Search for most similar documents to query"""
        query_embedding = self.encode_texts([query])

        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        D, I = self.index.search(query_embedding, k)

        # Normalize scores
        scores = self.normalize_scores(D[0])
        retrieved_docs = [doc_ids[idx] for idx in I[0]]

        return retrieved_docs, scores

    @staticmethod
    def compute_metrics(qrels: Dict, results: Dict) -> Dict:
        """Compute retrieval metrics using pytrec_eval"""
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels,
            {'map', 'ndcg_cut.10', 'P.10', 'recall', 'P.5'}
        )
        metrics = evaluator.evaluate(results)

        # Calculate mean metrics
        mean_metrics = defaultdict(float)
        for query_metrics in metrics.values():
            for metric, score in query_metrics.items():
                mean_metrics[metric] += score

        num_queries = len(metrics)
        return {metric: score / num_queries for metric, score in mean_metrics.items()}

    def evaluate_retrieval(
            self,
            queries: Dict[str, str],
            documents: Dict[str, str],
            qrels: Dict,
            k: int = 10,
            use_existing_index: bool = False
    ) -> Tuple[Dict, Dict]:
        # Build or load index
        index, doc_ids = self.build_index(documents, use_existing_index)

        # Prepare results dictionaries
        results = {}
        results_with_scores = {}

        # Search for each query
        for qid, query_text in queries.items():
            retrieved_docs, scores = self.search(query_text, doc_ids, k)

            # Format results for evaluation
            query_results = {}
            query_results_with_scores = {}

            for doc_id, score in zip(retrieved_docs, scores):
                query_results[doc_id] = 1
                query_results_with_scores[doc_id] = float(score)

            results[qid] = query_results
            results_with_scores[qid] = query_results_with_scores

        # Compute metrics
        metrics = self.compute_metrics(qrels, results)
        return metrics, results_with_scores


def main():

    # Initialize retriever
    retriever = SBERTRetriever(
        model_name='all-MiniLM-L6-v2',
        index_path='../saved_model_SBERT/faiss_index.bin',
        embeddings_path='../saved_model_SBERT/document_embeddings.npy'
    )

    use_desc = False
    # Load your data
    queries, documents, qrels = parse_files(use_desc)

    # Evaluate
    metrics, results = retriever.evaluate_retrieval(
        queries,
        documents,
        qrels,
        k=10,
        use_existing_index=False
    )

    # Print results
    print("\nSearch Results:")
    for qid, query_text in queries.items():
        print(f"\nQuery {qid}: {query_text}")
        query_results = sorted(results[qid].items(), key=lambda x: x[1], reverse=True)[:5]
        for doc_id, score in query_results:
            print(f"Document: {documents[doc_id]} (score: {score:.3f})")

    # Print metrics
    print("\nRetrieval Metrics:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()
