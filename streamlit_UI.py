import streamlit as st
import faiss

from data_preperar_for_faiss_and_validation import parse_files
from faiss_index_normal_bert_implementation import encode_texts
from Streamlit_helpers import encode_text
from transformers import BertModel, BertTokenizer, AutoTokenizer, AutoModel
from projectBert_1 import DenseRetriever
import torch
import faiss
import os
import numpy as np
import pytrec_eval

queries, documents, qrels = parse_files()
doc_ids = list(documents.keys())
# Initialize your FAISS indices
index1 = faiss.read_index("saved_models_for_wot_train_bert\\faiss_index.bin")
index2 = faiss.read_index("saved_model_normal_bert\\faiss_index.bin")
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
    dists_1, indices_1 = index1.search(query_emb_1, k= 10)
    results1 = doc_ids[indices_1[0][0]]
    st.write(results1)  # Display results

    # Block 2
    st.subheader("Results from Index 2 - BERT With fine-tune")
    query_emb_2 = encode_texts(query, model2, tokenizer2)
    dists_2, indices_2 = index2.search(query_emb_2, k=10)
    results2 = doc_ids[indices_2[0][0]]
    st.write(results2)  # Display results

    # Block 3
    st.subheader("Results from Index 3")
    results3 = "retriever3.retrieve(query, index3, tokenizer3)"
    st.write(results3)  # Display results