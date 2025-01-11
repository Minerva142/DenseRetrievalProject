import json
import re


# File paths (update with actual file paths)
file1_path = "merged_output.json"
file2_paths = ["q-topics-org-SET1.txt", "q-topics-org-SET2.txt", "q-topics-org-SET3.txt"]
file3_path = "filtered_data.txt"

out_queries_path = "queries_rerank.txt"
out_positive_docs_path = "positive_docs_rerank.json"
out_negative_docs_list_path = "negative_docs_list_rerank.json"

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

    for num, query in query_mapping.items():
        # Get relevant and non-relevant docnos for the current query
        relevant_docnos = doc_mapping.get(num, {}).get("positive", [])
        non_relevant_docnos = doc_mapping.get(num, {}).get("negative", [])

        # Find relevant documents (positive)
        pos_docs = [doc["TEXT"] for doc in file1_data if doc["DOCNO"] in relevant_docnos]
        positive_docs.append(pos_docs if pos_docs else ["No relevant document found."])

        # Find non-relevant documents (negative)
        neg_docs = [doc["TEXT"] for doc in file1_data if doc["DOCNO"] in non_relevant_docnos]
        negative_docs.append(neg_docs if neg_docs else ["No unrelated document found."])

        return queries, positive_docs, negative_docs

# Save data to files

queries, positive_docs, negative_docs = get_elements()

def save_list_to_file(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

save_list_to_file(queries, out_queries_path)
save_list_to_file(positive_docs, out_positive_docs_path)
save_list_to_file(negative_docs, out_negative_docs_list_path)

print("Data processing complete. Files saved:")
print(f"- Queries: {out_queries_path}")
print(f"- Positive Docs: {out_positive_docs_path}")
print(f"- Negative Docs: {out_negative_docs_list_path}")
