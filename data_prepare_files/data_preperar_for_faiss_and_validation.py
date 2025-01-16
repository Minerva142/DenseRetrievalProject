import json
import re


file1_path = "../data_prepare_files/merged_output.json"
file2_paths = ["../data_prepare_files/q-topics-org-SET1.txt", "../data_prepare_files/q-topics-org-SET2.txt", "../data_prepare_files/q-topics-org-SET3.txt"]
file3_path = "../data_prepare_files/filtered_data.txt"


def save_to_json(data, filename):
    """Saves the given data to a JSON file."""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)

def parse_trec_topics(text, use_desc = False):
    # Initialize empty dictionary for queries
    queries = {}

    # Find all topic sections
    topics = re.findall(r'<top>.*?</top>', text, re.DOTALL)
    descriptions = re.findall(r"<desc> Description:\s*(.*?)\n\s*<narr>", text, re.DOTALL)

    for topic, desc in zip(topics,descriptions):
        # Extract number
        number_match = re.search(r'<num>\s*Number:\s*(\d+)', topic)
        # Extract title
        title_match = re.search(r'<title>\s*(.*?)\s*(?=<desc>|$)', topic)
        if use_desc:
            if number_match:
                number = number_match.group(1)
                desc = re.sub(r"\s+", " ", desc.strip())
                queries[number] = desc
        else:
            if number_match and title_match:
                number = number_match.group(1)
                title = title_match.group(1).strip()
                queries[number] = title

    return queries

def parse_files(use_desc = False):
    """
    Reads the files and returns the datasets in the desired structure.
    file1_path: Path to the first file containing document details.
    file2_paths: List of paths to the second files containing queries (TXT files).
    file3_path: Path to the third file containing relevance judgments.
    """
    # Step 1: Read file1 - Documents
    documents = {}
    with open(file1_path, 'r') as f1:
        file1_data = json.load(f1)
        for doc in file1_data:
            doc_id = doc['DOCNO']
            documents[doc_id] = doc['TEXT']

    # Step 2: Read file2 - Queries (TXT Files)

    all_queries = {}
    for file2_path in file2_paths:
        with open(file2_path, 'r') as f2:
            text = f2.read()
            queries = parse_trec_topics(text, use_desc)
            all_queries.update(queries)


    # Step 3: Read file3 - Qrels (relevance judgments)
    qrels = {}
    with open(file3_path, 'r') as f3:
        for line in f3:
            parts = line.strip().split()
            query_id = f"{parts[0]}"
            doc_id = parts[2]
            relevance = int(parts[3])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = relevance

    return format_datasets(all_queries, documents, qrels)

def parse_files_with_paths(file1_path, file2_paths, file3_path, use_Desc = False):
    documents = {}
    with open(file1_path, 'r') as f1:
        file1_data = json.load(f1)
        for doc in file1_data:
            doc_id = doc['DOCNO']
            documents[doc_id] = doc['TEXT']

    # Step 2: Read file2 - Queries (TXT Files)

    all_queries = {}
    for file2_path in file2_paths:
        with open(file2_path, 'r') as f2:
            text = f2.read()
            queries = parse_trec_topics(text)
            all_queries.update(queries)


    # Step 3: Read file3 - Qrels (relevance judgments)
    qrels = {}
    with open(file3_path, 'r') as f3:
        for line in f3:
            parts = line.strip().split()
            query_id = f"{parts[0]}"
            doc_id = parts[2]
            relevance = int(parts[3])
            if query_id not in qrels:
                qrels[query_id] = {}
            qrels[query_id][doc_id] = relevance

    return format_datasets(all_queries, documents, qrels)


def format_datasets(queries, documents, qrels):
    # Get the first 35000 documents
    formatted_documents = {doc_id: text for doc_id, text in list(documents.items())[:50000]}
       

    # elimate the non-relevant queries
    formatted_queries = {qid: text for qid, text in list(queries.items())}

    has_rel_qids = []
    for qid in qrels.keys():
        has_rel_qids.append(qid)

    formatted_queries = {qid: text for qid, text in formatted_queries.items() if qid in has_rel_qids}
    formatted_queries = {qid: text for qid, text in list(formatted_queries.items())[-40:]}

    # Adjust qrels based on the selected documents
    formatted_qrels = {}
    to_remove = []
    for qid, rels in qrels.items():
        # Only include relations for documents present in the selected documents
        # and only if the query is in the last 50 queries
        if qid in formatted_queries:
            rel_list = list(rels.values())
            if rel_list.count(1) < 5:
                to_remove.append(qid)  # Add the query ID to the removal list
                continue

            filtered_rels = {doc_id: rel for doc_id, rel in rels.items() if doc_id in formatted_documents}
            if filtered_rels:  # Only add if there are any valid relations
                formatted_qrels[qid] = filtered_rels

    for key in to_remove:
        formatted_queries.pop(key, None)
    
    return formatted_queries, formatted_documents, formatted_qrels

def format_datasets_all(queries, documents, qrels):

    # get them randomly, by adding 5-6
    # ıf not present in document list, dont add to qrels relations
    formatted_queries = {qid: text for qid, text in queries.items()}
    formatted_documents = {doc_id: text for doc_id, text in documents.items()}
    formatted_qrels = {qid: rels for qid, rels in qrels.items()}

    return formatted_queries, formatted_documents, formatted_qrels


def save():

    # Process the files
    queries, documents, qrels = parse_files()

    save_to_json(queries, '../formatted_queries.json')
    save_to_json(documents, '../formatted_documents.json')
    save_to_json(qrels, '../formatted_qrels.json')

# Format the datasets
#formatted_queries, formatted_documents, formatted_qrels = format_datasets(queries, documents, qrels)

# Output the formatted data
#print("queries = ")
#print(json.dumps(formatted_queries, indent=4))

#print("\nqrels = ")
#print(json.dumps(formatted_qrels, indent=4))

#print("\ndocuments = ")
#print(json.dumps(formatted_documents, indent=4))
