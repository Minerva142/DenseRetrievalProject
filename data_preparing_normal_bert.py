import json
import re 

# File paths
file1_path = "merged_output.json"
file2_paths = ["q-topics-org-SET1.txt", "q-topics-org-SET2.txt", "q-topics-org-SET3.txt"]
file3_path = "filtered_data.txt"


# Output files
queries_file = "queries.txt"
positive_docs_file = "positive_docs.txt"
negative_docs_file = "negative_docs.txt"

# Read File 1 (JSON format)
def read_file1(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Read File 2 (TOP format)
def read_file2(file_paths):
    queries = []
    query_mapping = {}
    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            topics = re.findall(r"<top>(.*?)</top>", content, re.DOTALL)
            for topic in topics:
                num_match = re.search(r"<num> Number: (\d+)", topic)
                title_match = re.search(r"<title>\s*(.*?)\s*<desc>", topic, re.DOTALL)
                if num_match and title_match:
                    num = num_match.group(1).strip()
                    title = title_match.group(1).strip()
                    queries.append(title)
                    query_mapping[num] = title
    return queries, query_mapping

# Read File 3 (Tab-separated format)
def read_file3(file_path):
    doc_mapping = {}
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            parts = line.split()
            if len(parts) >= 4:
                query_num = parts[0]
                docno = parts[2]
                relevance = parts[3]
                if query_num not in doc_mapping:
                    doc_mapping[query_num] = {"positive": [], "negative": []}
                if relevance == "1":
                    doc_mapping[query_num]["positive"].append(docno)
                elif relevance == "0":
                    doc_mapping[query_num]["negative"].append(docno)
    return doc_mapping

# Process files
def get_elements():
    file1_data = read_file1(file1_path)
    queries, query_mapping = read_file2(file2_paths)
    doc_mapping = read_file3(file3_path)

    # Prepare positive and negative docs
    positive_docs = []
    negative_docs = []
    need_to_erase_list = []
    
    for num, query in query_mapping.items():
        # Get relevant and non-relevant docnos for the current query
        relevant_docnos = doc_mapping.get(num, {}).get("positive", [])
        non_relevant_docnos = doc_mapping.get(num, {}).get("negative", [])

        if not relevant_docnos and not non_relevant_docnos:
            need_to_erase_list.append(query)
            continue  # Skip this query if there are no positive or negative documents

        # Find one relevant document (positive)
        pos_doc = next((doc["TEXT"] for doc in file1_data if doc["DOCNO"] in relevant_docnos), "No relevant document found.")
        positive_docs.append(pos_doc)

        # Find one non-relevant document (negative)
        neg_doc = next((doc["TEXT"] for doc in file1_data if doc["DOCNO"] in non_relevant_docnos), "No unrelated document found.")
        negative_docs.append(neg_doc)
    
    for query in need_to_erase_list:
        queries.remove(query)

    return queries, positive_docs, negative_docs

def get_elements_train():
    file1_data = read_file1(file1_path)
    queries, query_mapping = read_file2(file2_paths)
    doc_mapping = read_file3(file3_path)

    # Prepare positive and negative docs
    positive_docs = []
    negative_docs = []

    need_to_erase_list = []

    for num, query in query_mapping.items():
        # Get relevant and non-relevant docnos for the current query
        relevant_docnos = doc_mapping.get(num, {}).get("positive", [])
        non_relevant_docnos = doc_mapping.get(num, {}).get("negative", [])

        # Check if there are any positive or negative documents for the query
        if not relevant_docnos or not non_relevant_docnos:
            need_to_erase_list.append(query)
            continue  # Skip this query if there are no positive or negative documents

        # Find one relevant document (positive)
        pos_doc = next((doc["TEXT"] for doc in file1_data if doc["DOCNO"] in relevant_docnos), "No relevant document found.")
        positive_docs.append(pos_doc)

        # Find one non-relevant document (negative)
        neg_doc = next((doc["TEXT"] for doc in file1_data if doc["DOCNO"] in non_relevant_docnos), "No unrelated document found.")
        negative_docs.append(neg_doc)

    for query in need_to_erase_list:
        queries.remove(query)

    return queries[:-40], positive_docs[:-40], negative_docs[:-40]

# Save data to files
def save_list_to_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(item + "\n")

def save():
    queries, positive_docs, negative_docs = get_elements()

    save_list_to_file(queries, queries_file)
    save_list_to_file(positive_docs, positive_docs_file)
    save_list_to_file(negative_docs, negative_docs_file)

    print("Data processing complete. Files saved:")
    print(f"- Queries: {queries_file}")
    print(f"- Positive Docs: {positive_docs_file}")
    print(f"- Negative Docs: {negative_docs_file}")