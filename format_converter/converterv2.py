import os
import json
import re


def clean_text(text):
    """Clean the text by removing newlines and '\\a' characters."""
    if text:
        return text.replace('\n', ' ').replace('\a', ' ').strip()
    return None


def parse_documents(file_path):
    with open(file_path, 'r') as file:
        data = file.read()

    # Split the file into individual <DOC> sections
    documents = re.split(r"<DOC>", data)

    parsed_documents = []

    for doc in documents:
        if not doc.strip():
            continue

        # Extract fields using regex and clean their values
        doc_no = clean_text(re.search(r"<DOCNO>(.*?)</DOCNO>", doc).group(1)) if re.search(r"<DOCNO>(.*?)</DOCNO>",
                                                                                           doc) else None
        profile = clean_text(re.search(r"<PROFILE>(.*?)</PROFILE>", doc).group(1)) if re.search(
            r"<PROFILE>(.*?)</PROFILE>", doc) else None
        date = clean_text(re.search(r"<DATE>(.*?)</DATE>", doc).group(1)) if re.search(r"<DATE>(.*?)</DATE>",
                                                                                       doc) else None
        headline = clean_text(re.search(r"<HEADLINE>(.*?)</HEADLINE>", doc, re.DOTALL).group(1)) if re.search(
            r"<HEADLINE>(.*?)</HEADLINE>", doc, re.DOTALL) else None
        byline = clean_text(re.search(r"<BYLINE>(.*?)</BYLINE>", doc, re.DOTALL).group(1)) if re.search(
            r"<BYLINE>(.*?)</BYLINE>", doc, re.DOTALL) else None
        text = clean_text(re.search(r"<TEXT>(.*?)</TEXT>", doc, re.DOTALL).group(1)) if re.search(r"<TEXT>(.*?)</TEXT>",
                                                                                                  doc,
                                                                                                  re.DOTALL) else None
        pub = clean_text(re.search(r"<PUB>(.*?)</PUB>", doc).group(1)) if re.search(r"<PUB>(.*?)</PUB>", doc) else None
        page = clean_text(re.search(r"<PAGE>(.*?)</PAGE>", doc).group(1)) if re.search(r"<PAGE>(.*?)</PAGE>",
                                                                                       doc) else None

        # Append additional info to the TEXT field
        additional_info = {
            "Date": date,
            "Headline": headline,
            "Byline": byline,
            "Pub": pub,
            "Page": page
        }

        appended_info = " ".join(
            [f"{key}:{value}" for key, value in additional_info.items() if value]
        )
        full_text = f"{text} {appended_info}" if text else appended_info

        # Create a dictionary for each document
        parsed_document = {
            "DOCNO": doc_no,
            "PROFILE": profile,
            "TEXT": full_text
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
