# livertoxrag/evaluator.py
"""
Evaluation module for the LiverTox RAG-LLM system.

This module provides functionality to evaluate the performance of the
RAG-LLM system by running batch evaluations on a set of clinical questions,
tracking metrics such as drug matching success rate and answer quality.
"""

import csv
import json
import numpy as np
from pathlib import Path
from livertoxrag.rag_llm import answer_question
from sentence_transformers import SentenceTransformer

def run_evaluation(index, metadata, top_k=10, questions_path="evaluation/questions.csv", output_path="evaluation/model_outputs.csv"):
    """
    Run a batch evaluation of clinical questions using the LiverTox RAG-LLM system.
    
    This function processes a CSV file containing evaluation questions, runs each
    question through the RAG-LLM pipeline, and records the results including the
    answer and retrieval metrics. It tracks drug matching success, answer length,
    and stores the retrieved document sections for each question.
    
    The evaluation can handle the CSV format with separate Drug and question columns.
    It reports the actual chunks used by answer_question and provides a detailed summary
    of the evaluation results.

    Args:
        index (faiss.Index): FAISS vector index used for similarity search
        metadata (dict): Dictionary of chunk metadata with string keys mapping to 
                      dictionaries containing at minimum "doc_id", "section", and "text" fields
        top_k (int, optional): Number of chunks to retrieve per question. Defaults to 5.
        questions_path (str, optional): Path to CSV file with Drug, question, and expected columns.
                                      Defaults to "evaluation/questions.csv".
        output_path (str, optional): Path to save evaluation results CSV.
                                   Defaults to "evaluation/model_outputs.csv".
                                   
    Returns:
        None. Results are written to the specified output_path CSV file and a summary
        is printed to standard output.
        
    Raises:
        FileNotFoundError: If the questions file cannot be found
        Exception: For other errors during file loading or processing
        
    CSV Format:
        Input CSV should have columns:
        - Drug: String with drug name (case-insensitive)
        - question: String with the question text
        - expected: String with expected answer (optional)
        
        Output CSV will have columns:
        - drug: Drug name from input
        - question: Question from input
        - expected: Expected answer from input
        - answer: Generated answer from the RAG-LLM system
        - retrieved_docs: Pipe-delimited list of document_id:section pairs used in retrieval
        - drug_match: "Yes" or "No" indicating if a document matching the drug was retrieved
    """
    # Ensure evaluation directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Load questions and expected answers
    try:
        with open(questions_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except FileNotFoundError:
        print(f"Error: Questions file not found at {questions_path}")
        return
    except Exception as e:
        print(f"Error loading questions: {e}")
        return
    
    # Statistics tracking
    stats = {
        "total_questions": len(rows),
        "drug_match_success": 0,
        "drug_match_failure": 0,
        "avg_answer_length": 0,
    }

    # Output CSV
    with open(output_path, "w", newline='', encoding='utf-8') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=["drug", "question", "expected", "answer", "retrieved_docs", "drug_match"])
        writer.writeheader()

        for row in rows:
            # Get drug and question from CSV
            drug = row.get('Drug', '').lower().strip()
            question = row.get('question', '')
            expected = row.get('expected', '')

            print(f"\nDrug: {drug}, Question: {question}")
            
            # Get answer and the chunks used to generate it
            answer, retrieved_chunks = answer_question(question, index, metadata, top_k=top_k, 
                                                      target_drug=drug, return_chunks=True)
            print(f"LLM Answer:\n{answer}")
            
            # Track retrieved documents and check for drug match
            retrieved_docs = []
            drug_match_found = False
            
            print("Sections retrieved:")
            for chunk in retrieved_chunks:
                doc_id = chunk["doc_id"]
                section = chunk["section"]
                
                # Check if document matches the drug
                is_drug_match = drug and drug in doc_id.lower()
                if is_drug_match:
                    drug_match_found = True
                    print(f"  - {section} - {doc_id} [MATCH]")
                else:
                    print(f"  - {section} - {doc_id}")
                    
                retrieved_docs.append(f"{doc_id}:{section}")
            
            # Update statistics
            if drug_match_found:
                stats["drug_match_success"] += 1
            else:
                stats["drug_match_failure"] += 1
                
            stats["avg_answer_length"] += len(answer.split())

            # Write to CSV
            writer.writerow({
                "drug": drug,
                "question": question,
                "expected": expected,
                "answer": answer,
                "retrieved_docs": "|".join(retrieved_docs),
                "drug_match": "Yes" if drug_match_found else "No"
            })
    
    # Calculate final stats
    if stats["total_questions"] > 0:
        stats["avg_answer_length"] /= stats["total_questions"]
        stats["drug_match_rate"] = stats["drug_match_success"] / stats["total_questions"] * 100
    
    # Print summary statistics
    print("\n===== EVALUATION SUMMARY =====")
    print(f"Total questions: {stats['total_questions']}")
    print(f"Drug match success: {stats['drug_match_success']} ({stats.get('drug_match_rate', 0):.1f}%)")
    print(f"Drug match failure: {stats['drug_match_failure']}")
    print(f"Average answer length: {stats['avg_answer_length']:.1f} words")
    
    print(f"\nEvaluation complete. Results saved to {output_path}")