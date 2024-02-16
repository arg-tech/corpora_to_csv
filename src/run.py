import argparse
from process_data_generic import (get_argument_elements, 
                          extract_zips,
                          bio_scheme_tagging, 
                          generate_relation_classifcation_data)


#'/Users/debelagemechu/Argument Mining/data/source/'

def run_pipeline(zip_folder_path, 
                 extraction_destination, 
                 source_data_folder, 
                 processed_data_folder, 
                 tagged_data_folder,
                 final_output_folder,
                 relation_types,
                 EDU_type):

    # Step 1: Download zip files
    print("Step 1: Downloading zip files")
    #download_zip_files(zip_folder_path, extraction_destination)

    # Step 1: Extract zip files
    print("\nExtract zip files")
    extract_zips(zip_folder_path, extraction_destination)

    # Step 2: Process JSON files
    print("\nStep 2: Processing JSON files")
    get_argument_elements(source_data_folder, processed_data_folder,relation_types,EDU_type)

    # Step 3: Process parent folder
    print("\nStep 3: Tagging argument maps using BIO schemes")
    bio_scheme_tagging(processed_data_folder, tagged_data_folder)

    # Step 4: Process input folder
    print("\nStep 4: Generating csv file for training")
    generate_relation_classifcation_data(tagged_data_folder, final_output_folder)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the argument mining pipeline.")
    
    parser.add_argument("--zip_folder_path", default="zipfiles/", help="Path to the folder containing zip files.")
    parser.add_argument("--extraction_destination", default="source/", help="Destination folder for extracting zip files.")
    parser.add_argument("--source_data_folder", default="source/", help="Folder containing source data in JSON format.")
    parser.add_argument("--processed_data_folder", default="processed/", help="Folder to store processed data in CSV format.")
    parser.add_argument("--tagged_data_folder", default="tagged/", help="Folder to store tagged propositions and relations.")
    parser.add_argument("--final_output_folder", default="final/", help="Final output folder for the resulting CSV file.")
    parser.add_argument("--relation_types", default=['CA', 'RA'], help="The type of relations: argument relations(RA, CA), Illocutions (YA).")
    parser.add_argument("--EDU_type", default=["I"], help="The type of argument unit required (I, L).")

    
    args = parser.parse_args()

    # Run the pipeline with provided or default folder paths
    run_pipeline(zip_folder_path=args.zip_folder_path,
                 extraction_destination=args.extraction_destination,
                 source_data_folder=args.source_data_folder,
                 processed_data_folder=args.processed_data_folder,
                 tagged_data_folder=args.tagged_data_folder,
                 final_output_folder=args.final_output_folder,
                 relation_types=args.relation_types,
                 EDU_type=args.EDU_type
                 )
    

    
#python your_script_name.py --zip_folder_path /custom/path/to/zipfiles --extraction_destination /custom/path/to/source --source_data_folder /custom/path/to/source_data --processed_data_folder /custom/path/to/processed_data --parent_folder /custom/path/to/parent_folder --tagged_data_folder /custom/path/to/tagged_data --final_output_folder /custom/path/to/final_output

#python your_script_name.py
