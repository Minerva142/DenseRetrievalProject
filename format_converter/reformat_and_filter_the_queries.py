import json

file_paths = ['q-topics-org-SET1.txt',
             'q-topics-org-SET2.txt',
             'q-topics-org-SET3.txt']# Replace with the path to your text file

file_path_for_ids = "filtered_data.txt"

json_output_path = 'queries.json'  # Replace with the path to the JSON output file



ids = []
with open(file_path_for_ids, 'r') as file:
    data_ids = file.readlines()
    for row in data_ids:
        ids.append(row.split()[0])


# Process the structured data into JSON format
queries = []
# Read data from the file
for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = file.read()
        for block in data.split('<top>'):
            block = block.strip()
            if not block:
                continue

            # For using titles as query
            # Extract the ID and description
            num_line = next((line for line in block.splitlines() if line.startswith('<num>')), None)
            title_line = next((line for line in block.splitlines() if line.startswith('<title>')), None)
            if num_line and title_line:
                query_id = num_line.split(': ')[1].strip()
                if query_id in ids:
                    description = title_line.replace('<title>', '').strip()
                    queries.append({"id": query_id, "text": description})
        '''
            num_line = next((line for line in block.splitlines() if line.startswith('<num>')), None)
            desc_start = block.find('<desc>')
            narr_start = block.find('<narr>')

            if num_line and desc_start != -1:
                query_id = num_line.split(':')[1].strip()
                if query_id in ids:
                    description = block[desc_start + len(
                        '<desc> Description:'):narr_start].strip() if narr_start != -1 else block[desc_start + len(
                        '<desc> Description:'):].strip()
        
        
        '''


# Write the JSON data to a new file
with open(json_output_path, 'w') as json_file:
    json.dump(queries, json_file, indent=2)
