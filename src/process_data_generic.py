
#dowload zip files from corpora

#downloading script
import requests
from pathlib import Path
import shutil
import os
import zipfile
import json
import os
import csv
import pandas as pd
import os
import pandas as pd
from transformers import BertTokenizer

# Load BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def download_copora(corpora_names):

    #corpora_names = ['araucaria', 'US2016D1reddit', 'US2016R1reddit', 'US2016D1tv', 'US2016R1tv','US2016G1tv', 'US2016D1']

    base_url = 'https://corpora.aifdb.org/zip/'
    for corpora_name in corpora_names:
        url = base_url + corpora_name
        print(url)
        output_file_path = f'/Users/debelagemechu/Argument Mining/data/zipfiles/{corpora_name}.zip'

        try:
            # Send a GET request to the URL with stream=True
            with requests.get(url, stream=True) as response:
                response.raise_for_status()  # Raise an error for bad HTTP responses

                # Open the file using 'wb' mode and copy the response content directly
                with open(output_file_path, 'wb') as f:
                    shutil.copyfileobj(response.raw, f)

            print(f"Downloaded '{output_file_path}' successfully.")
        except requests.exceptions.RequestException as e:
            print(f"An error occurred during download: {e}")


# Extracts zip folders
def extract_zips(folder_path, extraction_destination):

    # Specify the folder path where the .zip files are located
    #folder_path = '/Users/debelagemechu/Argument Mining/data/zipfiles/'

    # Specify the destination folder for extraction
    #extraction_destination = '/Users/debelagemechu/Argument Mining/data/source/'

    # Get a list of all files in the specified folder
    files = os.listdir(folder_path)

    # Filter only the .zip files
    zip_files = [file for file in files if file.endswith('.zip')]

    def extract_zip(zip_file_path, corpora_name, extraction_destination):
        try:
            # Open the zip file
            with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                # Extract all contents preserving the inner folder structure
                zip_ref.extractall(os.path.join(extraction_destination, corpora_name))

            print(f"Extracted contents from '{zip_file_path}' to '{extraction_destination}'")
        except zipfile.BadZipFile:
            print(f"Error: '{zip_file_path}' is not a valid zip file.")
        except Exception as e:
            print(f"Error: '{zip_file_path}' could not be extracted. {e}")



    # Iterate over each zip file and extract its contents
    for zip_file in zip_files:
        zip_file_path = os.path.join(folder_path, zip_file)

        # Print for debugging
        print(f"Processing '{zip_file_path}'")

        # Extract the zip file preserving inner folder structure
        extract_zip(zip_file_path, zip_file[:-4], extraction_destination)



# Extracts relevant information from argument graphs and outpus csv files for each corpus
def get_argument_elements(source_data_folder, processed_data_folder, relation_types = ['CA', 'RA'], EDU_type = ["I"]): 


    # Specify source and processed data folders
    #source_data_folder = '/Users/debelagemechu/Argument Mining/data/source/'
    #processed_data_folder = '/Users/debelagemechu/Argument Mining/data/source/processed/'

    # Create processed data folder if not exists
    if not os.path.exists(processed_data_folder):
        os.makedirs(processed_data_folder)

    def process_json_file(json_file_path, processed_subfolder):
        # Load JSON data
        try:
            with open(json_file_path, 'r') as json_file:
                data = json.load(json_file)

            # Extract CA and RA nodes
            relation_nodes = [n for n in data['nodes'] if n.get('type') in relation_types]

            # Extract I nodes
            info_nodes = [n for n in data['nodes'] if n.get('type') in EDU_type]

            # Extract propositions
            propositions = {node['nodeID']: node['text'] for node in info_nodes}

            argument = " '[SEP]' ".join([p for i, p in propositions.items()])
            argument = "[ARGSTART] " + argument + " [ARGEND] "

            # Extract argument relations
            argument_relations = {}

            # Iterate through I nodes
            for rl_node in relation_nodes:
                rl_id = rl_node.get('nodeID')
                relation_type = rl_node.get('text')

                related_props = None
                for edge in data['edges']:
                    if rl_id == edge['fromID'] and edge['toID'] in propositions:
                        prop_id = edge['toID']
                        for edge2 in data['edges']:
                            if rl_id == edge2['toID'] and edge2['fromID'] in propositions:
                                related_props = propositions.get(edge2['fromID'], '')

                    if rl_id == edge['toID'] and edge['fromID'] in propositions:
                        prop_id = edge['fromID']
                        for edge2 in data['edges']:
                            if rl_id == edge2['fromID'] and edge2['toID'] in propositions:
                                related_props = propositions.get(edge2['toID'], '')

                if related_props:
                    if propositions.get(prop_id, '') in argument_relations:
                        argument_relations[propositions.get(prop_id, '')].append((related_props, relation_type))
                    else:
                        argument_relations[propositions.get(prop_id, '')] = [(related_props,relation_type)]

            # Save results to CSV
            df = pd.DataFrame(data={"Argument": [argument] * len(argument_relations),
                                    "Proposition": [prop for prop, related_props in argument_relations.items()],
                                    "related_ps": [related_props for prop, related_props in argument_relations.items()],
                                    "source": [str(processed_subfolder)] * len(argument_relations)
                                    } 
                            )

            return df
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in {json_file_path}: {e}")
            return None
        except FileNotFoundError:
            print(f"File not found: {json_file_path}")
            return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None


    # Process every JSON file in source data folder
 

    for root, dirs, files in os.walk(source_data_folder):
        for filename in files:
            if filename.endswith('.json'):
                json_file_path = os.path.join(root, filename)
                
                # Create corresponding subfolder in processed_data_folder
                relative_folder_path = os.path.relpath(root, source_data_folder)
                processed_subfolder = os.path.join(processed_data_folder, relative_folder_path)
                os.makedirs(processed_subfolder, exist_ok=True)

                result_df = process_json_file(json_file_path,processed_subfolder)

                if result_df is not None:

                    # Save processed data to CSV in the processed subfolder
                    result_df.to_csv(os.path.join(processed_subfolder, f"{os.path.splitext(filename)[0]}_processed.csv"),index=False)


# tag each argument based on BIO scheme
                    




def bio_scheme_tagging(parent_folder, output_folder):

    # Example usage:
    #parent_folder = '../processed/'  # This folder contains subfolders, each with processed CSV files
    #output_folder = '../tagged_data_folder'

    # Function to extract start and end indices of each sequence in the subset
    def get_start_end_indices(full_sequence, related_props):
        subsets = [text for text, relation in related_props]

        indices = []
        full_sequence_tokens = tokenizer.tokenize(full_sequence)

        for subset in subsets:
            subset_tokens = tokenizer.tokenize(subset)

            start_index = None
            for i in range(len(full_sequence_tokens) - len(subset_tokens) + 1):
                if full_sequence_tokens[i:i+len(subset_tokens)] == subset_tokens:
                    start_index = i
                    end_index = i + len(subset_tokens) - 1
                    indices.append((start_index, end_index))
                    break

        return indices
    
    # Create tagged data folder if not exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Process every subfolder in the parent folder
    for subfolder_name in os.listdir(parent_folder):
        subfolder_path = os.path.join(parent_folder, subfolder_name)
        
        try:

            # Create a list to store DataFrames from each file
            dfs = []

            # Process every CSV file in the subfolder
            for file_no, filename in enumerate(os.listdir(subfolder_path), start=1):
                if filename.endswith('.csv'):
                    csv_file_path = os.path.join(subfolder_path, filename)

                    # Read the CSV file into a DataFrame
                    df = pd.read_csv(csv_file_path)


                    if not df.empty:


                        # Apply the function to each row
                        df['Start_End_Indices'] = df.apply(lambda row: get_start_end_indices(row['Argument'], eval(row['related_ps'])), axis=1)

                        # Add 'arg_no' column to indicate the source file
                        df['arg_no'] = file_no

                        # Append the DataFrame to the list
                        dfs.append(df)

            if len(dfs) > 0:
                # Concatenate all DataFrames into one
                result_df = pd.concat(dfs, ignore_index=True)

                # Save the result DataFrame to a new CSV file in the output folder
                result_df.to_csv(os.path.join(output_folder, f"tagged_propsoitions_and_relations_combined_{subfolder_name}.csv"), index=False)
        except NotADirectoryError as e:
            print(f"Error processing subfolder '{subfolder_name}': {e}")





def generate_relation_classifcation_data(input_folder, output_folder):

    #input_folder = '../data/tagged_data_folder/'
    #output_folder = '../data/final/'

    def get_rel_prop_and_type(prop, related_props):
        #print("related_props", related_props)
        
        for text,rl in related_props:
            if text == prop:
                return (text, rl)
        return (None,None)

    # Specify the output file path for the resulting CSV
    output_file_path = os.path.join(output_folder, 'resulting_file.csv')

    # Initialize an empty list to store DataFrames from each file
    dfs = []

    # Process every CSV file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.csv'):
            csv_file_path = os.path.join(input_folder, filename)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path)

            # Your existing code for processing each row
            p1 = []
            p2 = []
            arg = []
            start_end_indexes = []
            relations = []
            all_relations = []
            sources = []
            original_relation_types = []

            for index, row in df.iterrows():
                proposition = row['Proposition']
                related_propositions = eval(row['related_ps'])
                contatenated_arguments = row['Argument']
                argument = row['Argument']
                start_end_indices = row['Start_End_Indices']
                source = row['source']

                replacements = {"[ARGSTART]": "", "[ARGEND]": ""}

                for old, new in replacements.items():
                    contatenated_arguments = contatenated_arguments.replace(old, new)

                argument_propositions = contatenated_arguments.split("'[SEP]'")

                for argument_proposition in argument_propositions:
                    arg.append(contatenated_arguments)
                    start_end_indexes.append(start_end_indices)
                    p1.append(proposition)
                    p2.append(argument_proposition)
                    all_relations.append(row['related_ps'])
                    sources.append(source)
                    relation = 1
                    prop, rl = get_rel_prop_and_type(argument_proposition.strip(), related_propositions)
                    original_relation_types.append(rl)
                    if prop is None:
                        relation = 0
                    else:
                        if rl == "Default Inference":
                            relation = 1
                        elif rl == "Default Conflict":
                            relation = 2

                    relations.append(relation)

            # Create a DataFrame for the current CSV file
            result_df = pd.DataFrame({
                'argument': arg,
                'proposition_1': p1,
                'proposition_2': p2,
                'original_relation_types':original_relation_types,
                'source': sources,
                'relations': relations,
                'start_end_indexes': start_end_indexes,
                'all_relations': all_relations
            })

            # Append the DataFrame to the list
            dfs.append(result_df)

    # Concatenate all DataFrames into one
    final_result_df = pd.concat(dfs, ignore_index=True)

    # Save the resulting DataFrame to a new CSV file
    final_result_df.to_csv(output_file_path, index=False)

    print(f"Resulting CSV file saved to: {output_file_path}")



