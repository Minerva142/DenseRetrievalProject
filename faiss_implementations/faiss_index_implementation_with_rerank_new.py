import torch
import faiss
import numpy as np
from typing import Dict, List
from transformers import AutoTokenizer
import pytrec_eval
from tqdm import tqdm
from sklearn.preprocessing import normalize

from data_prepare_files.data_preperar_for_faiss_and_validation import parse_files
from model_implementations.projectBertWithReRank_2 import DualEncoder, CrossEncoder
import os

model_path = '../saved_models_for_with_rerank'
path_of_emb = '../saved_models_for_with_rerank/doc_embeddings_re_ranker.npy' # we can use them from standart embeedings
path_of_faiss = '../saved_models_for_with_rerank/doc_faiss_with_reranker.bin'
use_already_exists_emb = False
use_already_exists_faiss = False

def normalize_vectors(vectors):
    """
    Normalize vectors to unit length
    """
    return normalize(vectors, norm='l2')
class DenseRetrieverReRank:
    def __init__(
            self,
            dual_encoder_path: str,
            cross_encoder_path: str,
            tokenizer_name: str = "microsoft/mpnet-base",
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        # Initialize dual encoder
        self.dual_encoder = DualEncoder()
        self.dual_encoder.load_state_dict(torch.load(dual_encoder_path, map_location=device))
        self.dual_encoder.to(device)
        self.dual_encoder.eval()

        # Initialize cross encoder
        self.cross_encoder = CrossEncoder()
        self.cross_encoder.load_state_dict(torch.load(cross_encoder_path, map_location=device))
        self.cross_encoder.to(device)
        self.cross_encoder.eval()

        # Initialize FAISS index
        self.dimension = 768  # MPNet embedding dimension
        self.index = None
        self.documents = {}
        self.doc_mapping = {}

    def build_index(self, documents: Dict[str, str], batch_size: int = 32):
        """Build FAISS index from documents"""
        self.documents = documents
        self.index = faiss.IndexFlatIP(self.dimension)

        doc_items = list(documents.items())

        for i in tqdm(range(0, len(doc_items), batch_size), desc="Building FAISS index"):
            batch_docs = doc_items[i:i + batch_size]

            # Tokenize documents
            inputs = self.tokenizer(
                [text for _, text in batch_docs],
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # Get embeddings using dual encoder
            with torch.no_grad():
                if use_already_exists_emb:
                    embeddings = np.load(path_of_emb)
                else:
                    embeddings = self.dual_encoder.encode(
                        inputs['input_ids'],
                        inputs['attention_mask']
                    ).cpu().numpy()
                    np.save(path_of_emb, embeddings)

            embeddings = normalize_vectors(embeddings)
            # Add to index
            self.index.add(embeddings)

            # Update document mapping
            for idx, (doc_id, _) in enumerate(batch_docs):
                self.doc_mapping[i + idx] = doc_id
            print(self.doc_mapping)

    @torch.no_grad()
    def retrieve(
            self,
            query: str,
            k: int = 100,
            rerank_k: int = 10,
            return_scores: bool = True
    ) -> List[Dict]:
        """Retrieve and rerank documents for query"""
        # Encode query
        query_inputs = self.tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors='pt'
        ).to(self.device)

        # Get query embedding from dual encoder
        query_embedding = self.dual_encoder.encode(
            query_inputs['input_ids'],
            query_inputs['attention_mask']
        ).cpu().numpy()

        query_embedding = normalize_vectors(query_embedding)
        # First stage retrieval with FAISS
        scores, indices = self.index.search(query_embedding, k)
        doc_ids = [self.doc_mapping[int(idx)] for idx in indices[0]]

        if rerank_k > 0:
            # Rerank top k documents using cross encoder
            rerank_scores = []

            # Prepare pairs for cross encoder
            pairs = [(query, self.documents[doc_id]) for doc_id in doc_ids[:rerank_k]]

            # Tokenize pairs
            pair_inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(self.device)

            # Get reranking scores
            scores = self.cross_encoder(
                pair_inputs['input_ids'],
                pair_inputs['attention_mask']
            )
            rerank_scores = scores.squeeze(-1).cpu().numpy()

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

    def evaluate(
            self,
            queries: Dict[str, str],
            qrels: Dict[str, Dict[str, int]],
            k: int = 100,
            rerank_k: int = 10
    ) -> Dict[str, float]:
        """Evaluate retrieval performance using pytrec_eval"""
        # Prepare run dict for pytrec_eval
        run = {}

        # Get retrievals for all queries
        for qid, query in tqdm(queries.items(), desc="Evaluating queries"):
            results = self.retrieve(query, k=k, rerank_k=rerank_k)
            run[qid] = {result['id']: result['score'] for result in results}

        # Initialize evaluator with standard metrics
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {
                'map',  # Mean Average Precision
                'ndcg_cut.10',  # NDCG@10
                'recall',  # Recall@100
                'P.10'
            }
        )

        # Calculate metrics
        metrics = evaluator.evaluate(run)

        # Average metrics across queries
        mean_metrics = {
            'map': np.mean([q['map'] for q in metrics.values()]),
            'ndcg_10': np.mean([q['ndcg_cut_10'] for q in metrics.values()]),
            'recall_10': np.mean([q['recall_10'] for q in metrics.values()]),
            'P_10': np.mean([q['P_10'] for q in metrics.values()])
        }

        return mean_metrics

    def save_index(self, index_path: str):
        """Save FAISS index"""
        if self.index is None:
            raise ValueError("No index to save")
        faiss.write_index(self.index, index_path)

    def load_index(self, index_path: str):
        """Load FAISS index"""
        self.index = faiss.read_index(index_path)


def main():
    # Initialize retriever
    retriever = DenseRetrieverReRank(
        dual_encoder_path='../saved_models_for_with_rerank/dual_encoder.pt',
        cross_encoder_path='../saved_models_for_with_rerank/cross_encoder.pt'
    )

    use_desc = True
    queries, documents, qrels = parse_files(use_desc)

    # Build and save index
    retriever.build_index(documents)
    retriever.save_index(path_of_faiss)


    metrics = retriever.evaluate(queries, qrels)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")


if __name__ == "__main__":
    main()