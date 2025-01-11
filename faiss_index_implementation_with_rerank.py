import torch
import faiss
import numpy as np
from typing import List, Dict, Tuple
from transformers import BertTokenizer
from projectBertWithReRank import DenseRetrieverBERT, CrossAttentionReranker
from tqdm import tqdm
import json


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

    def build_index(self, documents: List[Dict[str, str]], batch_size: int = 32):
        """
        Build Faiss index from documents.
        documents: List of dicts with 'id' and 'text' keys
        """
        # Initialize Faiss index
        self.index = faiss.IndexFlatIP(self.dimension)  # Inner product index (for normalized vectors)

        # Process documents in batches
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_docs = documents[i:i + batch_size]

            # Tokenize
            inputs = self.tokenizer(
                [doc['text'] for doc in batch_docs],
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
            for idx, doc in enumerate(batch_docs):
                self.doc_mapping[i + idx] = doc['id']

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
                doc_text = self.get_document_by_id(doc_id)['text']
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

    def get_document_by_id(self, doc_id: str) -> Dict:
        """
        Implement this method based on your document storage system.
        Should return a dict with at least 'id' and 'text' keys.
        """
        raise NotImplementedError(
            "Implement this method to retrieve document text by ID from your storage system"
        )


# Example usage
def main():
    # Initialize retriever
    retriever = DenseRetriever(
        retriever_path="multi_positive_retriever.pt",
        reranker_path="multi_positive_reranker.pt"
    )

    # Example documents
    documents = [
        {
            'id': '1',
            'text': 'Machine learning is a subset of artificial intelligence'
        },
        {
            'id': '2',
            'text': 'Deep learning uses neural networks with multiple layers'
        },
        # Add more documents...
    ]

    # Build and save index
    retriever.build_index(documents)
    retriever.save_index(
        index_path="retriever_index.faiss",
        mapping_path="doc_mapping.json"
    )

    # Example query
    results = retriever.retrieve(
        query="what is machine learning",
        k=100,
        rerank_k=10
    )

    print("Top results:")
    for result in results:
        print(f"Document ID: {result['id']}, Score: {result['score']}")


if __name__ == "__main__":
    main()