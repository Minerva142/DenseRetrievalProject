import torch
from transformers import AutoTokenizer, AutoModel
import faiss
import numpy as np
import pytrec_eval
from typing import List, Dict, Tuple
from collections import defaultdict
import json
from statistics import mean, stdev
import os
from data_prepare_files.data_preperar_for_faiss_and_validation import parse_files

use_already_exist_faiss = False
use_already_exist_doc_embeddings = False
model_path = "../saved_models_just_bert_tokenizer"
faiss_path = os.path.join(model_path, "faiss_index.bin")
doc_embeedings_path = os.path.join(model_path, "document_embeddings.npy")
class SearchEvaluator:
    def __init__(self, model_name: str = 'bert-base-uncased',
                 device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the search evaluator with BERT model and FAISS index

        Args:
            model_name: Name of the BERT model to use
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()
        self.index = None
        self.doc_ids = []
        self.doc_texts = []

    def get_embedding(self, text: str) -> np.ndarray:
        """
        Get BERT embedding for a piece of text

        Args:
            text: Input text to embed

        Returns:
            numpy array of embeddings
        """
        inputs = self.tokenizer(text, return_tensors='pt',
                                padding=True, truncation=True,
                                max_length=512).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        return embedding

    def build_index(self, documents: Dict[str, str]):
        """
        Build FAISS index from a dictionary of documents

        Args:
            documents: Dictionary mapping document IDs to document texts
        """
        self.doc_ids = list(documents.keys())
        self.doc_texts = list(documents.values())

        if use_already_exist_doc_embeddings:
            embeddings = np.load(doc_embeedings_path)
        else :
            embeddings = []

            # Get embeddings for all documents
            for doc_text in self.doc_texts:
                embedding = self.get_embedding(doc_text)
                embeddings.append(embedding)

            embeddings = np.vstack(embeddings)
            np.save(doc_embeedings_path, embeddings)

        dimension = embeddings.shape[1]

        # Initialize and build FAISS index
        if use_already_exist_faiss:
            self.index = faiss.read_index(faiss_path)
        else :
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            faiss.write_index(self.index, faiss_path)

    def search(self, query: str, k: int = 10) -> List[Tuple[str, float]]:
        """
        Search the index with a query

        Args:
            query: Query text
            k: Number of results to return

        Returns:
            List of (doc_id, distance) tuples
        """
        query_embedding = self.get_embedding(query)
        distances, indices = self.index.search(query_embedding.astype('float32'), k)

        # Convert numeric indices to document IDs
        results = [(self.doc_ids[idx], dist) for idx, dist in zip(indices[0], distances[0])]
        return results

    @staticmethod
    def calculate_metric_statistics(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        """
        Calculate statistics for each metric across all queries

        Args:
            results: Dictionary of query results from pytrec_eval

        Returns:
            Dictionary containing mean, std dev, min, and max for each metric
        """
        # Organize metrics by name
        metrics_by_name = defaultdict(list)
        for query_results in results.values():
            for metric_name, value in query_results.items():
                metrics_by_name[metric_name].append(value)

        # Calculate statistics for each metric
        statistics = {}
        for metric_name, values in metrics_by_name.items():
            statistics[metric_name] = {
                'mean': mean(values),
                'std_dev': stdev(values) if len(values) > 1 else 0,
                'min': min(values),
                'max': max(values),
                'num_queries': len(values)
            }

        return statistics

    def evaluate(self, queries: Dict[str, str],
                 relevant_docs: Dict[str, Dict[str, int]],
                 k: int = 10) -> Dict[str, Dict[str, float]]:
        """
        Evaluate search results using pytrec_eval

        Args:
            queries: Dictionary of query_id to query text
            relevant_docs: Dictionary of query_id to relevant document ids and their relevance scores
            k: Number of results to retrieve per query

        Returns:
            Dictionary of evaluation metrics with statistics
        """
        # Prepare run results in TREC format
        run_results = {}

        for query_id, query_text in queries.items():
            search_results = self.search(query_text, k)
            run_results[query_id] = {
                doc_id: 1.0 / (rank + 1)  # Converting distance to score
                for rank, (doc_id, _) in enumerate(search_results)
            }

        # Define metrics to evaluate
        metrics = {
            'map',
            'ndcg_cut.10',
            'P.10',
            'recall.10',
            'bpref'  # Binary Preference
        }

        # Initialize evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(relevant_docs, metrics)

        # Calculate metrics for each query
        query_results = evaluator.evaluate(run_results)

        # Calculate statistics across queries
        metric_stats = self.calculate_metric_statistics(query_results)

        return {
            'per_query': query_results,
            'statistics': metric_stats
        }


# Example usage
if __name__ == "__main__":
    # Initialize evaluator
    evaluator = SearchEvaluator()

    queries, documents, relevant_docs= parse_files()

    evaluator.build_index(documents)
    # Evaluate and print results
    results = evaluator.evaluate(queries, relevant_docs)
    print(json.dumps(results, indent=2))