import torch
import faiss
import numpy as np
from typing import List, Dict
from transformers import BertTokenizer
from model_implementations.projectBertWithReRank_2 import DenseRetrieverBERT, CrossAttentionReranker
from tqdm import tqdm
import json
import pytrec_eval
from data_prepare_files.data_preperar_for_faiss_and_validation import parse_files
import os

#TODO add load part of embeeding and faiss
model_path = '../saved_models_for_with_rerank'
path_of_emb = os.path.join(model_path, 'doc_embeddings_re_ranker.npy') # we can use them from standart embeedings
path_of_faiss = os.path.join(model_path, 'doc_faiss_with_reranker.bin')
use_already_exists_emb = False
use_already_exists_faiss = False
class DenseRetriever:
    def __init__(
            self,
            retriever_path: str,
            reranker_path: str,
            tokenizer_name: str = "bert-base-uncased",
            index_path: str = None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_name)
        self.documents = {}  # Store documents for retrieval

        # Load retriever
        self.retriever = DenseRetrieverBERT()
        self.retriever.load_state_dict(torch.load(retriever_path, map_location=device))
        self.retriever.to(device)
        self.retriever.eval()

        # Load reranker
        self.reranker = CrossAttentionReranker()
        self.reranker.load_state_dict(torch.load(reranker_path, map_location=device))
        self.reranker.to(device)
        self.reranker.eval()

        # Initialize Faiss index
        self.dimension = 768  # BERT hidden size
        self.index = None
        if index_path:
            self.load_index(index_path)

        self.doc_mapping = {}  # Maps Faiss index to document ID

    def build_index(self, documents: Dict[str, str], batch_size: int = 32):
        """
        Build Faiss index from documents.
        documents: Dict with document IDs as keys and text as values
        """
        # Store documents for later retrieval
        self.documents = documents
        
        # Initialize Faiss index
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

    def save_index(self, index_path: str, mapping_path: str = None):
        """Save Faiss index and document mapping"""
        if self.index is None:
            raise ValueError("No index to save")

        faiss.write_index(self.index, index_path)

        if mapping_path:
            with open(mapping_path, 'w') as f:
                json.dump(self.doc_mapping, f)

    def load_index(self, index_path: str, mapping_path: str = None):
        """Load Faiss index and document mapping"""
        self.index = faiss.read_index(index_path)

        if mapping_path:
            with open(mapping_path, 'r') as f:
                self.doc_mapping = json.load(f)

    @torch.no_grad()
    def retrieve(
            self,
            query: str,
            k: int = 100,
            rerank_k: int = 10,
            return_scores: bool = True
    ) -> List[Dict]:
        """
        Retrieve and rerank documents for a query.
        Returns top rerank_k documents from the top k retrieved documents.
        """
        # Encode query
        query_inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        # Get query embedding
        query_embedding = self.retriever(
            query_inputs['input_ids'],
            query_inputs['attention_mask']
        ).cpu().numpy()

        # First stage retrieval
        scores, indices = self.index.search(query_embedding, k)

        # Get document IDs
        doc_ids = [self.doc_mapping[int(idx)] for idx in indices[0]]

        if rerank_k > 0:
            # Rerank top k documents
            rerank_scores = []

            for doc_id in doc_ids[:rerank_k]:
                doc_text = self.documents[doc_id]
                doc_inputs = self.tokenizer(
                    doc_text,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors='pt'
                ).to(self.device)

                score = self.reranker(
                    query_inputs['input_ids'],
                    query_inputs['attention_mask'],
                    doc_inputs['input_ids'],
                    doc_inputs['attention_mask']
                )
                rerank_scores.append(score.item())

            # Sort by reranker scores
            reranked_indices = np.argsort(rerank_scores)[::-1]
            doc_ids = [doc_ids[i] for i in reranked_indices]
            scores = [rerank_scores[i] for i in reranked_indices]

            # Truncate to rerank_k
            doc_ids = doc_ids[:rerank_k]
            scores = scores[:rerank_k]

        # Prepare results
        results = []
        for idx, doc_id in enumerate(doc_ids):
            result = {'id': doc_id}
            if return_scores:
                result['score'] = float(scores[idx])
            results.append(result)

        return results

    def evaluate(self, queries: Dict[str, str], qrels: Dict[str, Dict[str, int]], k: int = 100, rerank_k: int = 10) -> Dict[str, float]:
        """
        Evaluate retrieval performance using pytrec_eval
        """
        # Prepare run dict for pytrec_eval
        run = {}
        
        # Get retrievals for all queries
        for qid, query in tqdm(queries.items(), desc="Evaluating queries"):
            results = self.retrieve(query, k=k, rerank_k=rerank_k)
            run[qid] = {result['id']: result['score'] for result in results}

        # Initialize evaluator
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, 
            {'map', 'ndcg_cut.10', 'P.10', 'recall.100', 'P.5'}
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

    def get_document_by_id(self, doc_id: str) -> Dict:
        """Return document by ID from the documents dictionary"""
        return {'id': doc_id, 'text': self.documents.get(doc_id, '')}


def main():
    # Initialize retriever
    retriever = DenseRetriever(
        retriever_path=os.path.join(model_path, "multi_positive_rerank_retriever.pt"),
        reranker_path=os.path.join(model_path, "multi_positive_reranker.pt")
    )

    queries, documents, qrels = parse_files()

    # Build and save index
    retriever.build_index(documents)
    retriever.save_index(
        index_path=path_of_faiss,
        mapping_path="doc_mapping.json"
    )

    # Evaluate
    metrics = retriever.evaluate(queries, qrels)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    # Example single query retrieval
    results = retriever.retrieve(
        query="What is dense retrieval?",
        k=100,
        rerank_k=10
    )

    print("\nTop results for example query:")
    for result in results:
        print(f"Document ID: {result['id']}, Score: {result['score']}")
        print(f"Text: {documents[result['id']]}\n")


if __name__ == "__main__":
    main()
