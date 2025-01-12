# DenseRetrievalProject


Under DataOperator directory, there are some python scripts to format data, formatted data is used by the data_preparing_*.py scripts.

## Approaches
All approaches are using BERT model and BERT encoders.

// TODO PROVIDE MORE INFO
approach_1 is training dense retriever with exact one positive and one negative document for each query.  
approach_2 is dense retrieval with reranking model (and negative mining). training them with multiple positive and negative documents for each query.  
aprroach_3 is training dense retriever with one positive and multiple negative documents for each query. 

# Kaggle Implementations
Other kaggle implementations can be found here:  

https://www.kaggle.com/code/eraygnlal/llama-dense-retrieval-inference/edit  
https://www.kaggle.com/code/eraygnlal/llama-dense-retrieval-inference-trec-eval-added

// TODO FILL
In there we are tried to use llama as encoder and we try to 
