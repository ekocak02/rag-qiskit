from src.database.storage_manager import QiskitVectorStore

# Constants
INPUT_FILE = "data/merged/unified.jsonl"
BATCH_SIZE = 12

def main():
    try:
        print("--- Qiskit RAG Indexing Started ---")
        
        store = QiskitVectorStore()
        store.process_and_index(jsonl_path=INPUT_FILE, batch_size=BATCH_SIZE)
        
        print("--- Process Finished Successfully ---")
        
    except Exception as e:
        print(f"Critical Error: {e}")

if __name__ == "__main__":
    main()