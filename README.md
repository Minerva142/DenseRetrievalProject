# DenseRetrievalProject

Under [format_convertor directory](https://github.com/Minerva142/DenseRetrievalProject/tree/main/format_converter) , there are some python scripts to format data, formatted data is used by the **data_preparing_X.py** scripts which are under the [data_prepare_files](https://github.com/Minerva142/DenseRetrievalProject/tree/main/data_prepare_files) directory. These files provide the training and whole data for training parts.  

## Approaches
All approaches are using BERT model and BERT encoders(for *.py files).  
approaches number tagged with the number of approach(as suffix) to implementation files.

* [aprroach_1](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/projectBert_1.py) is training dense retriever with exact one positive and one negative document for each query. 
* [aprroach_2](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/projectBertWithReRank_2.py) is dense retrieval with reranking model (and negative mining). training them with multiple positive and negative documents for each query.  
* [aprroach_3](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/ProjectBertDenseRetrieverMultiWotReRank_3.py) is training dense retriever with one positive and multiple negative documents for each query.
* [aprroach_4](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/projectBetMultiPosAndNeg_4.py)

* Furthermore there are 2 different faiss index without training. [First](https://github.com/Minerva142/DenseRetrievalProject/blob/main/faiss_implementations/DirectBert_with_faiss_wot_training.py) one is BERT model and BERT tokenizer faiss, and the [second](https://github.com/Minerva142/DenseRetrievalProject/blob/main/faiss_implementations/faiss_just_using_bert_tokenizer.py) one is directly using BERT tokenizer. You can see the implementation from file names. 

## Faiss Index Implementations
used faiss index are IP and L2 based, we retrieve 10 documents for each query while testing. 

## Kaggle Implementations
Other kaggle implementations can be found here:  

https://www.kaggle.com/code/eraygnlal/llama-dense-retrieval-inference/edit  
https://www.kaggle.com/code/eraygnlal/llama-dense-retrieval-inference-trec-eval-added

// TODO FILL
In there we are tried to use llama as encoder and we try to 

## Demo

Little [demo]() is provided.
