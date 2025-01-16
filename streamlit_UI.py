import numpy as np
from sentence_transformers import SentenceTransformer
import streamlit as st
from data_prepare_files.data_preperar_for_faiss_and_validation import parse_files_with_paths
from faiss_implementations.faiss_index_implementation_with_rerank_new import DenseRetrieverReRank
from faiss_implementations.faiss_index_normal_bert_implementation import encode_texts
from faiss_implementations.faiss_index_implementation_SBERT import SBERTRetriever
from Streamlit_helpers import encode_text_helper, search_similar_documents, normalize_vectors
from transformers import BertModel, BertTokenizer, AutoTokenizer
from model_implementations.projectBert_1 import DenseRetriever
import torch
import faiss
import os

# change path with kaggle
queries, documents, qrels = parse_files_with_paths('data_prepare_files/merged_output.json', ["data_prepare_files/q-topics-org-SET1.txt", "data_prepare_files/q-topics-org-SET2.txt", "data_prepare_files/q-topics-org-SET3.txt"], 'data_prepare_files/filtered_data.txt')
doc_ids = list(documents.keys())
# Initialize your FAISS indices
index1 = faiss.read_index("saved_models_for_wot_train_bert/doc_faiss_wot_train_bert.bin")
index2 = faiss.read_index("saved_model_normal_bert/faiss_index.bin") # change paths with kaggle
index3 = faiss.read_index("saved_model_SBERT/faiss_index.bin")
index4 = faiss.read_index("saved_models_for_with_rerank/doc_faiss_with_reranker.bin")
index5 = faiss.read_index("saved_model_multi-qa-mpnet-base-cos-v1/faiss_index.bin")

tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer2 = AutoTokenizer.from_pretrained('saved_model_normal_bert')

model1 = BertModel.from_pretrained('bert-base-uncased')
model2 = DenseRetriever("bert-base-uncased")
model2.load_state_dict(torch.load(os.path.join('saved_model_normal_bert', "best_dense_retriever.pt")))
model3 = SentenceTransformer('all-MiniLM-L6-v2')

model4 = DenseRetrieverReRank('saved_models_for_with_rerank/dual_encoder.pt', 'saved_models_for_with_rerank/cross_encoder.pt')
model4.index = index4
model4.documents = documents
doc_keys = list(documents.keys())
model4.doc_mapping = {i: value for i, value in enumerate(doc_keys)}

model5 = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')


# Streamlit UI
st.title("Document Retrieval System")

query = st.text_input("Enter your query:")

if query:
    # Block 1
    st.subheader("Results from Index 1 - BERT")
    query_emb_1 = encode_text_helper(query, tokenizer1, model1)
    dists_1, indices_1 = index1.search(np.array([query_emb_1]), k = 10)
    result1_dict = {}
    for ind in indices_1[0]:
        results1 = doc_ids[ind]
        doc_text = documents[results1]
        result1_dict[results1] = doc_text[:200] + "....."
    st.write(result1_dict)  # Display results

    # Block 2
    st.subheader("Results from Index 2 - BERT With fine-tune")
    query_emb_2 = encode_texts([query], model2, tokenizer2)
    dists_2, indices_2 = index2.search(query_emb_2, k=10)
    result2_dict = {}
    for ind in indices_2[0]:
        results2 = doc_ids[ind]
        doc_text = documents[results2]
        result2_dict[results2] = doc_text[:200]+ "....."
    st.write(result2_dict)  # Display results

    # Block 3
    st.subheader("Results from Index 3 - SBERT")
    query_emb_3 = model3.encode([query])
    query_emb_3 = normalize_vectors(query_emb_3)
    #query_emb_3 = query_emb_3 / np.linalg.norm(query_emb_3)
    dists_3, indices_3 = index3.search(query_emb_3, k=10)
    result3_dict = {}
    for ind in indices_3[0]:
        results3 = doc_ids[ind]
        doc_text = documents[results3]
        result3_dict[results3] = doc_text[:200] + "....."
    st.write(result3_dict)  # Display results

    # Block 4
    st.subheader("Results from Index 4 - Fine tune with ReRank(cross encoder and dual encoder)")
    results_4 = model4.retrieve(query)
    result4_dict = {}
    for result in results_4:
        doc_id_4 = result['id']
        doc_text = documents[doc_id_4]
        result4_dict[doc_id_4] = doc_text[:200] + "....."
    st.write(result4_dict)  # Display results

    # Block 5

    st.subheader("Results from Index 4 - Sentence transformer (multi-qa-mpnet-base-cos-v1)")
    results_5 = search_similar_documents(query, model5, index5, doc_keys)
    result5_dict = {}
    for (docn_no,_) in results_5:
        doc_text = documents[docn_no]
        result5_dict[docn_no] = doc_text[:200]
    st.write(result5_dict)  # Display results

