# DenseRetrievalProject

Under [format_convertor directory](https://github.com/Minerva142/DenseRetrievalProject/tree/main/format_converter) , there are some python scripts to format data, formatted data is used by the **data_preparing_X.py** scripts which are under the [data_prepare_files](https://github.com/Minerva142/DenseRetrievalProject/tree/main/data_prepare_files) directory. These files provide the training and whole data for training parts. Befor using, you need to create all dataset and directories which are needed. Furthermore, you need to install all needed libraries.  

## Approaches
Rather then the second approach, all approaches are using BERT model and BERT encoders(for *.py files). 

approaches number tagged with the number of approach(as suffix) to implementation files.

* [aprroach_1](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/projectBert_1.py) is training dense retriever with exact one positive and one negative document for each query. // TODO fill there
* [aprroach_2](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/projectBertWithReRank_2.py) is dense retrieval with reranking model .Training them with multiple positive and negative documents for each query. It uses already pre-trained(for starting point) cross encoder model as reranker and also use dual encoder for retrieving part. BCEWithLogitsLoss is used while training.// TODO fill there
* [aprroach_3](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/ProjectBertDenseRetrieverMultiWotReRank_3.py) is training dense retriever with one positive and multiple negative documents for each query. // TODO fill there
* [aprroach_4](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/projectBetMultiPosAndNeg_4.py) is training dense retriever with one positive and multiple negative documents for each query. // TODO fill there

* Furthermore there are 3 different faiss index without training. [First](https://github.com/Minerva142/DenseRetrievalProject/blob/main/faiss_implementations/DirectBert_with_faiss_wot_training.py) one is BERT model and BERT tokenizer faiss, and the [second](https://github.com/Minerva142/DenseRetrievalProject/blob/main/faiss_implementations/faiss_just_using_bert_tokenizer.py) one is directly using BERT tokenizer. [Third](https://github.com/Minerva142/DenseRetrievalProject/blob/main/faiss_implementations/faiss_index_implementation_SBERT.py) it directly uses the pre-trained BERT sentence transformer as encoder.You can see the implementation from file names. 

## Kaggle Implementations
Other kaggle implementations can be found here:  

https://www.kaggle.com/code/eraygnlal/llama-dense-retrieval-inference/edit  
https://www.kaggle.com/code/eraygnlal/llama-dense-retrieval-inference-trec-eval-added

In these approaches, a pre-trained llama model was used as an encoder. A faiss index was then created and metric calculations were performed with the pytrec_eval library. The data used here was created with data formatter scripts available on the repository.

## Faiss Index Implementations

Faiss implementations can be found under this [directory](https://github.com/Minerva142/DenseRetrievalProject/tree/main/faiss_implementations). Each of them feeded with dataset which are provided with this [script](https://github.com/Minerva142/DenseRetrievalProject/blob/main/data_prepare_files/data_preperar_for_faiss_and_validation.py). Document number is limited with 50000 for testing purposes because encoding tooks a considirable time for each. Furthermore, using more document, in some cases, effects the result in negative manner. 
used faiss index are IP and L2 based, we retrieve 10 documents for each query while testing. 

// TODO fill there

## Evaluation Metrics



## Demo

Little [demo]() is provided.
