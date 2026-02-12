#!/usr/bin/env python3
"""
Example script demonstrating how to query the LiverTox RAG-LLM system.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from livertoxrag.vector_store import load_index
from livertoxrag.rag_llm import answer_question

# Example clinical questions
SAMPLE_QUESTIONS = [
    "What is the hepatotoxicity pattern of acetaminophen?",
    "What is the likelihood score for valproate causing liver injury?",
    "What are the risk factors for isoniazid hepatotoxicity?",
    "How long after starting amoxicillin-clavulanate does liver injury typically occur?",
    "What is the mechanism of liver injury from statins?",
]

def main():
    """
    Run example queries against the LiverTox RAG-LLM system.
    """
    print("LiverTox RAG-LLM Query Examples")
    print("================================\n")

    # Check if embeddings exist
    if not os.path.exists('embeddings/faiss_index.index'):
        print("ERROR: FAISS index not found!")
        print("Please run: python main.py --train")
        return

    # Load index and metadata
    print("Loading FAISS index and metadata...")
    index = load_index("embeddings/faiss_index.index")
    with open("embeddings/chunk_metadata.json") as f:
        metadata = json.load(f)

    print(f"Loaded index with {index.ntotal} vectors")
    print(f"Loaded metadata for {len(metadata)} chunks\n")

    # Run example queries
    for i, question in enumerate(SAMPLE_QUESTIONS[:3], 1):
        print(f"\n--- Example {i} ---")
        print(f"Question: {question}")
        print("-" * 40)

        try:
            answer = answer_question(question, index, metadata, top_k=5)
            print(f"Answer:\n{answer}")
        except Exception as e:
            print(f"Error: {e}")
            continue

if __name__ == "__main__":
    main()
