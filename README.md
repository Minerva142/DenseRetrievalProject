# DenseRetrievalProject

Under [format_convertor directory](https://github.com/Minerva142/DenseRetrievalProject/tree/main/format_converter) , there are some python scripts to format data, formatted data is used by the **data_preparing_X.py** scripts which are under the [data_prepare_files](https://github.com/Minerva142/DenseRetrievalProject/tree/main/data_prepare_files) directory. These files provide the training and whole data for training parts. Befor using, you need to create all dataset and directories which are needed. Furthermore, you need to install all needed libraries.  

## Approaches
Rather then the second approach, all approaches are using BERT model and BERT encoders(for *.py files). Approximately 66 percent of the dataset was used for training, the rest was used for testing purposes to calculate metric values in faiss index implementations.

approaches number tagged with the number of approach(as suffix) to implementation files.

* [aprroach_1](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/projectBert_1.py) is training dense retriever with exact one positive and one negative document for each query. Here are the test setup: 
    | Parameter        | Value               |
    |------------------|---------------------|
    | model_name       | "bert-base-uncased" |
    | max_length       | 128                 |
    | batch_size       | 2                   |
    | learning_rate    | 2e-5                |
    | epochs           | 3                   |
    | optimizer        | AdamW               |

* [aprroach_2](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/projectBertWithReRank_2.py) is dense retrieval with reranking model .Training them with multiple positive and negative documents for each query. It uses already pre-trained(for starting point) cross encoder model as reranker and also use dual encoder for retrieving part. BCEWithLogitsLoss is used while training for cross_encoder and cross entropy loss used for dual_encoder . Here are the setup values:
  | Parameter                | Value                                      |
  |--------------------------|--------------------------------------------|
  | dual_encoder             | microsoft/mpnet-base                      |
  | cross_encoder            | cross-encoder/ms-marco-MiniLM-L-6-v2    |
  | tokenizer                | microsoft/mpnet-base                      |
  | batch_size               | 8                                          |
  | num_epochs               | 3                                         |
  | learning_rate            | 2e-5                                      |
  | dual encoder loss        | cross entropy loss                         |
  | cross encoder loss       | BCEWithLogitsLoss                         |
  | optimizer                | AdamW                                    |

* [aprroach_3](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/ProjectBertDenseRetrieverMultiWotReRank_3.py) is training dense retriever with one positive and multiple negative documents for each query.
  
* [aprroach_4](https://github.com/Minerva142/DenseRetrievalProject/blob/main/model_implementations/projectBetMultiPosAndNeg_4.py) is training dense retriever with one positive and multiple negative documents for each query. Actually, it is limited with 5 for short training. 
    | Parameter         | Value                  |
    |-------------------|------------------------|
    | epochs            | 3                      |
    | learning_rate     | 2e-5                   |
    | loss function      | TripletMarginLoss      |
    | optimizer         | AdamW                  |

* Furthermore there are 4 different faiss index without training. [First](https://github.com/Minerva142/DenseRetrievalProject/blob/main/faiss_implementations/DirectBert_with_faiss_wot_training.py) one is BERT model and BERT tokenizer faiss, and the [second](https://github.com/Minerva142/DenseRetrievalProject/blob/main/faiss_implementations/faiss_just_using_bert_tokenizer.py) one is directly using BERT tokenizer. [Third](https://github.com/Minerva142/DenseRetrievalProject/blob/main/faiss_implementations/faiss_index_implementation_SBERT.py) it directly uses the pre-trained BERT sentence transformer as encoder.[Fourth] () is using sentence transformer.You can see the implementation from file names.
  
  

## Kaggle Implementations
Other kaggle implementations can be found here:  

https://www.kaggle.com/code/eraygnlal/llama-dense-retrieval-inference/edit  
https://www.kaggle.com/code/eraygnlal/llama-dense-retrieval-inference-trec-eval-added

In these approaches, a pre-trained llama model was used as an encoder. A faiss index was then created and metric calculations were performed with the pytrec_eval library. The data used here was created with data formatter scripts available on the repository.

## Faiss Index Implementations

Faiss implementations can be found under this [directory](https://github.com/Minerva142/DenseRetrievalProject/tree/main/faiss_implementations). Each of them feeded with dataset which are provided with this [script](https://github.com/Minerva142/DenseRetrievalProject/blob/main/data_prepare_files/data_preperar_for_faiss_and_validation.py). Document number is limited with 50000 for testing purposes because encoding tooks a considirable time for each. Furthermore, using more document, in some cases, effects the result in negative manner. Used faiss index are IP and L2 based, we retrieve 10 documents for each query while testing. 100000 tests were also performed here, but not for every case due to time problems as mentioned above. 



## Evaluation Metrics

pytrec_eval library used for metric evaluations. datasets formatted like [these](https://github.com/Minerva142/DenseRetrievalProject/tree/main/data_prepare_files). Example data formats are listed in below :

qrels:

    {
    "411": {
        "FT911-1027": 0,
        "FT911-1221": 0,
        "FT911-1432": 0,
        "FT911-1499": 0,
        "FT911-1943": 0,
        "FT911-2029": 0,
        ...............

queries :

    {
    "411": "salvaging, shipwreck, treasure",
    "412": "airport security",
    "413": "steel production",
    "414": "Cuba, sugar, exports",
    "416": "Three Gorges Project",
    "417": "creativity",
    "418": "quilts, income",

Used metrics are listed in below.

| **Metrics**      | **Description**                                                                                                                            |
|------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| **map**          | Mean Average Precision (MAP) is the mean of the Average Precision scores for all queries. It measures how well documents are ranked across queries. |
| **P_10**         | Precision at 10 (P@10) measures the fraction of relevant documents in the top 10 results for a query.                                      |
| **recall_5**     | Recall at 5 (R@5) is the fraction of relevant documents retrieved among the top 5 results for a query.                                     |
| **recall_10**    | Recall at 10 (R@10) is the fraction of relevant documents retrieved among the top 10 results for a query.                                  |
| **ndcg_cut_10**  | Normalized Discounted Cumulative Gain at 10 (NDCG@10) evaluates the ranking quality of the top 10 results by considering both relevance and position. |

## Comparisons

// TODO fill there

calculated metrics, loss or other different values are listed [here](https://github.com/Minerva142/DenseRetrievalProject/blob/main/metrics_and_expreiments_result.docx) as word document.

| Model & Query Type                                  | Training Data | MAP    | P@10   | Recall@5 | Recall@10 | NDCG@10 |
|-----------------------------------------------------|---------------|--------|--------|-----------|-----------|---------|
| Normal BERT w/o training, title as query           | 50,000        | 0.0198 | 0.0613 | 0.0156    | 0.0562    | 0.0556  |
| Normal BERT w/o fine-tuning, title as query        | 100,000       | 0.0047 | 0.0452 | 0.0011    | 0.0175    | 0.0336  |
| Normal BERT trained, title as query                | 50,000        | 0.0553 | 0.0903 | 0.1017    | 0.1350    | 0.1258  |
| SBERT, title as query                              | 50,000        | 0.1211 | 0.2065 | 0.1239    | 0.2379    | 0.2447  |
| Reranker (cross-encoder & dual encoder), title     | 50,000        | 0.0023 | 0.0129 | -         | 0.0114    | 0.0094  |
| Normal BERT trained, title as query                | 100,000       | 0.0410 | 0.1419 | 0.0404    | 0.0939    | 0.1443  |
| Normal BERT trained, desc as query                 | 50,000        | 0.0796 | 0.1194 | 0.1161    | 0.1729    | 0.1556  |
| Normal BERT w/o training, desc as query            | 50,000        | 0.0079 | 0.0161 | 0.0054    | 0.0269    | 0.0165  |
| Reranker (cross-encoder & dual encoder), desc      | 50,000        | 0.0073 | 0.0161 | -         | 0.0180    | 0.0203  |


## Key Result

// TODO fill there

## Demo

Little [demo](https://github.com/Minerva142/DenseRetrievalProject/blob/main/streamlit_UI.py) is provided.

