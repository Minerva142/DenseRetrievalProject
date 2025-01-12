import os
import json
import re


def parse_documents(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Split the file into individual <DOC> sections
    documents = re.split(r"<DOC>", data)

    parsed_documents = []

    for doc in documents:
        if not doc.strip():
            continue

        # Extract fields using regex
        doc_no = re.search(r"<DOCNO>(.*?)</DOCNO>", doc)
        profile = re.search(r"<PROFILE>(.*?)</PROFILE>", doc)
        date = re.search(r"<DATE>(.*?)</DATE>", doc)
        headline = re.search(r"<HEADLINE>(.*?)</HEADLINE>", doc, re.DOTALL)
        byline = re.search(r"<BYLINE>(.*?)</BYLINE>", doc, re.DOTALL)
        text = re.search(r"<TEXT>(.*?)</TEXT>", doc, re.DOTALL)
        pub = re.search(r"<PUB>(.*?)</PUB>", doc)
        page = re.search(r"<PAGE>(.*?)</PAGE>", doc)

        # Create a dictionary for each document
        parsed_document = {
            "DOCNO": doc_no.group(1).strip() if doc_no else None,
            "PROFILE": profile.group(1).strip() if profile else None,
            "DATE": date.group(1).strip() if date else None,
            "HEADLINE": headline.group(1).strip() if headline else None,
            "BYLINE": byline.group(1).strip() if byline else None,
            "TEXT": text.group(1).strip() if text else None,
            "PUB": pub.group(1).strip() if pub else None,
            "PAGE": page.group(1).strip() if page else None
        }

        parsed_documents.append(parsed_document)

    return parsed_documents


def process_directory_merge(input_dir, output_file):
    all_documents = []

    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_file_path = os.path.join(input_dir, filename)
            print(f"Processing {filename}...")
            # Parse the file and append to the list
            all_documents.extend(parse_documents(input_file_path))

    # Save all documents to a single JSON file
    with open(output_file, 'w') as file:
        json.dump(all_documents, file, indent=4)

    print(f"All files merged and saved to {output_file}")


# Directories
input_directory = "C:/Users/ERAY/OneDrive/Belgeler/ft/ft/all"  # Change to your actual input directory
output_file = "C:/Users/ERAY/OneDrive/Belgeler/ft/ft/converted/merged_output.json"  # Change to your desired output file path

# Merge all files into one JSON
process_directory_merge(input_directory, output_file)