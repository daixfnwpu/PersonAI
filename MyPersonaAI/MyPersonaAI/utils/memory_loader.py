# This script loads and formats the memory data
def load_memory(memory_path):
    with open(memory_path, 'r') as file:
        memory_data = file.read()
    return memory_data

if __name__ == "__main__":
    memory_path = "data/user_corpus.jsonl"
    print(load_memory(memory_path))