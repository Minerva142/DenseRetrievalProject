# DenseRetrievalProject

Under DataOperator directory, there are some python scripts to format data, formatted data is used by the data_preparing_*.py scripts. 

## Approaches
All approaches are using BERT model and BERT encoders(for *.py files).  

# TODO add more thinks
approaches tagged with the number of approach(as suffix).
* approach_1 is training dense retriever with exact one positive and one negative document for each query. 
* approach_2 is dense retrieval with reranking model (and negative mining). training them with multiple positive and negative documents for each query.  
* aprroach_3 is training dense retriever with one positive and multiple negative documents for each query.
* approach_4 

* Furthermore there are 2 different faiss index without training. First one is BERT model and BERT tokenizer faiss, and the second one is directly using BERT tokenizer. You can see the implementation from file names. 

used faiss index are IP and L2 based, we retrieve 10 documents for each query while testing. 

# Kaggle Implementations
Other kaggle implementations can be found here:  

https://www.kaggle.com/code/eraygnlal/llama-dense-retrieval-inference/edit  
https://www.kaggle.com/code/eraygnlal/llama-dense-retrieval-inference-trec-eval-added

// TODO FILL
In there we are tried to use llama as encoder and we try to 
