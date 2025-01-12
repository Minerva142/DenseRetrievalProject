file_paths = ['qrel_301-350_complete.txt',
              'qrels.trec7.adhoc_350-400.txt',
              'qrels.trec8.adhoc.parts1-5_400-450']  # Replace with the path to your text file

output_path = 'filtered_data.txt'  # Replace with the path to the output file

# Read data from the file
filtered_data = []
for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = file.readlines()

    # Filter the data
    data_parts = []
    for row in data:
        if row != '\n' and row.split()[2].startswith('FT'):
            filtered_data.append(row)

    # Write the filtered data to a new file
    with open(output_path, 'w') as file:
        file.writelines(filtered_data)
        #file.write('\n'.join(filtered_data))
