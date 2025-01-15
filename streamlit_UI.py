import streamlit as st

from data_prepare_files.data_preperar_for_faiss_and_validation import parse_files_with_paths
from faiss_implementations.faiss_index_normal_bert_implementation import encode_texts
from Streamlit_helpers import encode_text
from transformers import BertModel, BertTokenizer, AutoTokenizer
from model_implementations.projectBert_1 import DenseRetriever
import torch
import faiss
import os

queries, documents, qrels = parse_files_with_paths('data_prepare_files/merged_output.json', ["data_prepare_files/q-topics-org-SET1.txt", "data_prepare_files/q-topics-org-SET2.txt", "data_prepare_files/q-topics-org-SET3.txt"], 'data_prepare_files/filtered_data.txt')
doc_ids = list(documents.keys())
# Initialize your FAISS indices
index1 = faiss.read_index("saved_models_for_wot_train_bert/doc_faiss_wot_train_bert.bin")
index2 = faiss.read_index("saved_model_normal_bert/faiss_index.bin")
#index3 = faiss.read_index("path/to/faiss_index3.index")

tokenizer1 = BertTokenizer.from_pretrained('bert-base-uncased')
tokenizer2 = AutoTokenizer.from_pretrained('saved_model_normal_bert')

model1 = BertModel.from_pretrained('bert-base-uncased')
model2 = DenseRetriever("bert-base-uncased")
model2.load_state_dict(torch.load(os.path.join('saved_model_normal_bert', "best_dense_retriever.pt")))


# Streamlit UI
st.title("Document Retrieval System")

query = st.text_input("Enter your query:")

if query:
    # Block 1
    st.subheader("Results from Index 1 - BERT")
    query_emb_1 = encode_text(query, tokenizer1, model1)
    dists_1, indices_1 = index1.search([query_emb_1], k= 10)
    results1 = doc_ids[indices_1[0][0]]
    st.write(results1)  # Display results

    # Block 2
    st.subheader("Results from Index 2 - BERT With fine-tune")
    query_emb_2 = encode_texts(query, model2, tokenizer2)
    dists_2, indices_2 = index2.search([query_emb_2], k=10)
    results2 = doc_ids[indices_2[0][0]]
    st.write(results2)  # Display results

    # Block 3
    st.subheader("Results from Index 3")
    results3 = "retriever3.retrieve(query, index3, tokenizer3)"
    st.write(results3)  # Display results