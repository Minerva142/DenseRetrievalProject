import torch
import faiss
import numpy as np
import pytrec_eval
from typing import List, Dict
from transformers import BertTokenizer
from ProjectBertDenseRetrieverMultiWotReRank_3 import BertDenseRetrieval
from tqdm import tqdm
import json
import os

from data_preperar_for_faiss_and_validation import parse_files

model_path = 'saved_models_wot_with_rerank'
path_of_emb = os.path.join(model_path, 'doc_embeddings_wot_re_ranker.npy') # we can use them from standart embeedings
path_of_faiss = os.path.join(model_path, 'doc_faiss_with_wot_reranker.bin')
use_already_exists_emb = False
use_already_exists_faiss = False

class DenseRetrievalFAISS:
    def __init__(self, model_path: str, tokenizer_path: str, device: str = 'cuda'):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_path)
        self.retriever = BertDenseRetrieval(model_name=model_path)
        self.retriever.to(device)
        self.retriever.eval()
        
        self.dimension = 768  # BERT hidden size
        self.index = None
        self.doc_mapping = {}  # Maps FAISS index to document ID
        self.documents = {}  # Store documents for retrieval

    def build_index(self, documents: Dict[str, str], batch_size: int = 32):
        """
        Build FAISS index from document dictionary
        """
        # Store documents for later retrieval
        self.documents = documents
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatIP(self.dimension)

        # Convert dict to list format for batch processing
        doc_items = list(documents.items())
        
        for i in tqdm(range(0, len(doc_items), batch_size), desc="Building index"):
            batch_docs = doc_items[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                [text for _, text in batch_docs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # Get embeddings
            with torch.no_grad():
                embeddings = self.retriever(
                    inputs['input_ids'],
                    inputs['attention_mask']
                ).cpu().numpy()

            # Add to index
            self.index.add(embeddings)

            # Update document mapping
            for idx, (doc_id, _) in enumerate(batch_docs):
                self.doc_mapping[i + idx] = doc_id

    def search(self, query: str, k: int = 5) -> Dict[str, float]:
        """
        Search for similar documents
        """
        query_inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        # Get query embedding
        with torch.no_grad():
            query_embedding = self.retriever(
                query_inputs['input_ids'],
                query_inputs['attention_mask']
            ).cpu().numpy()

        # Search in the index
        scores, indices = self.index.search(query_embedding, k)

        results = {}
        for score, idx in zip(scores[0], indices[0]):
            doc_id = self.doc_mapping[idx]
            results[doc_id] = float(score)

        return results

    def evaluate(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], k: int = 100) -> Dict[str, float]:
        """
        Evaluate retrieval performance using pytrec_eval
        """
        # Prepare run dict for pytrec_eval
        run = {}
        
        # Get retrievals for all queries
        for qid, query in tqdm(queries.items(), desc="Evaluating queries"):
            results = self.search(query, k=k)
            run[qid] = {doc_id: score for doc_id, score in results.items()}

        # Initialize evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, 
            {'map', 'ndcg_cut.10', 'P.10', 'recall.100'}
        )
        
        # Calculate metrics
        metrics = evaluator.evaluate(run)
        
        # Average metrics across queries
        mean_metrics = {
            'map': np.mean([q['map'] for q in metrics.values()]),
            'ndcg@10': np.mean([q['ndcg_cut_10'] for q in metrics.values()]),
            'P@10': np.mean([q['P_10'] for q in metrics.values()]),
            'recall@100': np.mean([q['recall_100'] for q in metrics.values()])
        }
        
        return mean_metrics

def main():

    # Initialize retriever
    retriever = DenseRetrievalFAISS(os.path.join(model_path, 'dense_retrieval_model.pt'), model_path)

    queries, documents, qrels = parse_files()

    # Build index
    retriever.build_index(documents)

    # Evaluate
    evaluation_results = retriever.evaluate(queries, qrels)
    print("\nEvaluation Results:")
    for query_id, query_metrics in evaluation_results.items():
        print(f"\nQuery {query_id}:")
        for metric, value in query_metrics.items():
            print(f"{metric}: {value:.4f}")

    # Example search
    query = "What is dense retrieval?"
    results = retriever.search(query, k=5)
    print("\nSearch Results:")
    for doc_id, score in results.items():
        print(f"Document {doc_id}: {documents[doc_id]}")
        print(f"Score: {score:.4f}\n")

if __name__ == "__main__":
    main()