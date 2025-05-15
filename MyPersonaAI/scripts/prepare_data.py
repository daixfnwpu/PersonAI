# This script prepares the data for LoRA training and Elasticsearch indexing
import json

def prepare_data(input_path, output_path):
    with open(input_path, 'r') as infile:
        data = json.load(infile)
    
    # Further processing...
    with open(output_path, 'w') as outfile:
        json.dump(data, outfile)

prepare_data('data/user_corpus.jsonl', 'data/processed_user_corpus.jsonl')