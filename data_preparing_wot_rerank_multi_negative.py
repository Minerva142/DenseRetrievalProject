import json
import os

def parse_topics_file(file_path):
    """Parse the XML-like topics file to get query id and text"""
    queries = {}  # Changed to dict to store id-query pairs
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        topics = content.split('<top>')[1:]
        
        for topic in topics:
            # Extract number (query id)
            num_start = topic.find('<num>') + len('<num> Number: ')
            num_end = topic.find('\n', num_start)
            query_id = topic[num_start:num_end].strip()
            
            # Extract title (query)
            title_start = topic.find('<title>') + len('<title>')
            title_end = topic.find('\n', title_start)
            query = topic[title_start:title_end].strip()
            
            queries[query_id] = query
    
    return queries

def parse_documents_file(file_path):
    """Parse the JSON documents file"""
    documents = {}  # Changed to dict to store id-text pairs
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            documents[doc['DOCNO']] = doc['TEXT']
    
    return documents

def parse_qrels_file(file_path):
    """Parse the qrels file to get document relevance information"""
    qrels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            query_id, _, doc_id, relevance = line.strip().split()
            relevance = int(relevance)
            
            if query_id not in qrels:
                qrels[query_id] = {'positive': [], 'negative': []}
            
            if relevance == 1:
                qrels[query_id]['positive'].append(doc_id)
            elif relevance == 0:
                qrels[query_id]['negative'].append(doc_id)
    
    return qrels

def prepare_training_data(topics_path, docs_path, qrels_path):
    # Get queries with their IDs
    queries_dict = parse_topics_file(topics_path)
    
    # Get all documents
    documents = parse_documents_file(docs_path)
    
    # Get relevance information
    qrels = parse_qrels_file(qrels_path)
    
    # Prepare the final lists
    queries = []
    positive_docs = []
    negative_docs_list = []

    need_to_erase_list = []
    
    # For each query in the qrels
    for query_id in qrels:
        if query_id not in queries_dict:
            continue
            
        query = queries_dict[query_id]
        queries.append(query)

        if not qrels[query_id]['negative'] or qrels[query_id]['positive']:
            need_to_erase_list.append(query_id)
            continue

        # Get positive document
        pos_doc_ids = qrels[query_id]['positive']
        pos_doc = documents.get(pos_doc_ids[0], "") if pos_doc_ids else ""
        positive_docs.append(pos_doc)
        
        # Get negative documents
        neg_doc_ids = qrels[query_id]['negative'][:5]  # Get first two negative docs
        neg_docs = [documents.get(doc_id, "") for doc_id in neg_doc_ids]
        # Ensure we have exactly two negative docs per query
        while len(neg_docs) < 2:
            neg_docs.append("")
        negative_docs_list.append(neg_docs)
    
    for query in need_to_erase_list:
        queries.remove(query)
    
    # Save the prepared data
    save_prepared_data(queries, positive_docs, negative_docs_list)
    
    return queries, positive_docs, negative_docs_list

def save_prepared_data(queries, positive_docs, negative_docs_list):
    """Save the prepared data to a Python file"""
    with open('prepared_data.py', 'w', encoding='utf-8') as f:
        f.write("# Generated training data\n\n")
        
        f.write("queries = [\n")
        for query in queries:
            # Escape any quotes in the text
            query = query.replace('"', '\\"')
            f.write(f"    \"{query}\",\n")
        f.write("]\n\n")
        
        f.write("positive_docs = [\n")
        for doc in positive_docs:
            # Escape any quotes in the text
            doc = doc.replace('"', '\\"')
            f.write(f"    \"{doc}\",\n")
        f.write("]\n\n")
        
        f.write("negative_docs_list = [\n")
        for neg_docs in negative_docs_list:
            f.write("    [\n")
            for doc in neg_docs:
                # Escape any quotes in the text
                doc = doc.replace('"', '\\"')
                f.write(f"        \"{doc}\",\n")
            f.write("    ],\n")
        f.write("]\n")

def main():
    # Define your file paths
    topics_path = "path/to/topics.txt"
    docs_path = "path/to/documents.jsonl"
    qrels_path = "path/to/qrels.txt"
    
    queries, positive_docs, negative_docs_list = prepare_training_data(
        topics_path, docs_path, qrels_path
    )
    
    print(f"Processed {len(queries)} queries")
    print(f"Generated {len(positive_docs)} positive documents")
    print(f"Generated {len(negative_docs_list)} sets of negative documents")

if __name__ == "__main__":
    main()