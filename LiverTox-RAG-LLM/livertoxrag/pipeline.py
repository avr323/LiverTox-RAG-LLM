# livertoxrag/pipeline.py
"""
Pipeline module for the LiverTox RAG-LLM system.

This module provides high-level functions that orchestrate the entire
RAG-LLM pipeline, from data loading and processing to model training
and evaluation.
"""

from livertoxrag.data_loader import load_nxml_files
from livertoxrag.chunker import chunk_text
from livertoxrag.embedder import embed_chunks
from livertoxrag.vector_store import build_faiss_index, save_index, load_index
from livertoxrag.rag_llm import answer_question
from livertoxrag.evaluator import run_evaluation

from pathlib import Path
import json

def run_train():
    """
    Run the training pipeline to build the vector store from raw NXML files.
    
    This function orchestrates the following steps:
    1. Loading and parsing NXML files with automatic section filtering
    2. Chunking sections into manageable text segments
    3. Creating embeddings for the chunks
    4. Building and saving a FAISS vector index
    
    The pipeline creates the following artifacts:
    - data/processed_chunks.jsonl: JSON lines file containing the processed chunks
    - embeddings/faiss_index.index: FAISS index for similarity search
    - embeddings/chunk_metadata.json: Metadata for the indexed chunks
    
    Args:
        None
        
    Returns:
        None. Progress is printed to standard output, and files are created as side effects.
    
    Note:
        This function expects NXML files to be present in the data/raw_nxml directory.
        It will create the necessary output directories if they don't exist.
    """
    print("[1] Loading and parsing .nxml files with section filtering...")
    # The filtering of sections is handled within load_nxml_files
    docs = load_nxml_files(Path("data/raw_nxml"))

    print("[2] Chunking included sections...")
    chunks = []
    section_counts = {}
    
    for doc_id, section_dict in docs.items():
        for section_title, section_text in section_dict.items():
            # Keep track of section counts
            section_counts[section_title] = section_counts.get(section_title, 0) + 1
            
            # Create chunks
            section_chunks = chunk_text(doc_id, section_text, section_title=section_title)
            chunks.extend(section_chunks)

    print(f"Created {len(chunks)} chunks from included sections")
    print("Section distribution in chunks:")
    for section, count in sorted(section_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  - {section}: {count} sections")

    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/processed_chunks.jsonl", "w") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")

    print("[3] Embedding...")
    embeddings, metadata = embed_chunks(chunks)

    print("[4] Building vector index...")
    Path("embeddings").mkdir(parents=True, exist_ok=True)
    index = build_faiss_index(embeddings)
    save_index(index, "embeddings/faiss_index.index")
    with open("embeddings/chunk_metadata.json", "w") as f:
        json.dump(metadata, f)

    print("Training complete with section filtering applied.")

def run_test():
    """
    Run the evaluation pipeline to test the RAG-LLM system.
    
    This function loads the previously trained FAISS index and metadata,
    then evaluates the system using the questions in the default evaluation file.
    The evaluation results are saved to a CSV file for analysis.
    
    Args:
        None
        
    Returns:
        None. Evaluation metrics are printed to standard output, and 
        results are saved to a CSV file.
    
    Note:
        This function expects the following files to exist:
        - embeddings/faiss_index.index: FAISS index created by run_train()
        - embeddings/chunk_metadata.json: Metadata created by run_train()
        - evaluation/questions.csv: Questions file used for evaluation
        
        The results will be saved to evaluation/model_outputs.csv by default.
    
    Raises:
        FileNotFoundError: If any of the required files are missing
        json.JSONDecodeError: If the metadata file cannot be parsed
    """
    index = load_index("embeddings/faiss_index.index")
    with open("embeddings/chunk_metadata.json") as f:
        metadata = json.load(f)
    run_evaluation(index, metadata)