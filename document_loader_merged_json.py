import json


def load_text_list(json_file_path):
    documents = []

    with open(json_file_path, 'r') as file:
        data = json.load(file)
        # Extract just the TEXT values into a list
        for doc in data:
            documents.append(doc['TEXT'])

    return documents


def load_docs_with_ids(json_file_path):
    documents = []

    with open(json_file_path, 'r') as file:
        data = json.load(file)
        # Create list of dictionaries with id and text
        for doc in data:
            documents.append({
                'id': doc['DOCNO'],
                'text': doc['TEXT']
            })

    return documents

def load_docs_with_docno_as_id(json_file_path):
    documents = []

    with open(json_file_path, 'r') as file:
        data = json.load(file)
        # Create list of dictionaries with id and text
        for doc in data:
            documents.append({
                doc['DOCNO']: doc['TEXT']
            })

    return documents