# File: src/5_main_app.py
from src.retrieval import load_retriever_from_disk, retrieve_docs
from src.augmentation import create_augmented_prompt
from src.generation import generate_answer

def main():
    retriever = load_retriever_from_disk()
    print("\n--- Aloo Sahayak is ready! Ask your questions. Type 'exit' to quit. ---")

    while True:
        query = input("\nYour Question: ")
        if query.lower() == 'exit':
            break

        retrieved_docs = retrieve_docs(retriever, query)
        augmented_prompt = create_augmented_prompt(retrieved_docs, query)
        answer = generate_answer(augmented_prompt)
        
        print("\nAnswer:")
        print(answer)

        print("\nSources Used:")
        for doc in retrieved_docs:
            print(f"- {doc.metadata.get('source', 'Unknown')}")

if __name__ == "__main__":
    main()