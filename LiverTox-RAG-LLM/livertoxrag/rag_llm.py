# livertoxrag/rag_llm.py

from llama_cpp import Llama
from sentence_transformers import SentenceTransformer
import numpy as np
import re
from pathlib import Path
import faiss
import os

# Add the API client imports
from anthropic import Anthropic
from openai import OpenAI


# Load LLM once
model_path = os.environ.get("LIVERTOX_MODEL_PATH", "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf")
print(f"Loading LLM from: {model_path}")

# Add API clients initialization
claude_client = None
openai_client = None

# Initialize Anthropic API client
api_key = os.environ.get("ANTHROPIC_API_KEY")
if api_key:
    try:
        claude_client = Anthropic(api_key=api_key)
        print("Claude API client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize Claude API client: {e}")
        print("Will use local model instead")
else:
    print("No ANTHROPIC_API_KEY found in environment variables")

# Initialize OpenAI API client
openai_key = os.environ.get("OPENAI_API_KEY")
if openai_key:
    try:
        openai_client = OpenAI(api_key=openai_key)
        print("OpenAI API client initialized successfully")
    except Exception as e:
        print(f"Failed to initialize OpenAI API client: {e}")
        print("Will use local model instead")
else:
    print("No OPENAI_API_KEY found in environment variables")

# Keep the original local LLM code
try:
    llm = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_gpu_layers=-1  # Use GPU acceleration if available
    )
except Exception as e:
    print(f"Error loading model: {e}")
    print("Falling back to phi-2 model if available")
    try:
        llm = Llama(
            model_path="models/phi-2.Q4_K_M.gguf",
            n_ctx=4096
        )
    except Exception as e2:
        print(f"Error loading fallback model: {e2}")
        print("Please make sure you have downloaded one of the models.")
        print("Run: python -m scripts.download_model --model mistral-7b-instruct-v0.1")
        raise e

# Load embedding model once
embedder = SentenceTransformer('pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb')

# Clinical priority order for section titles
SECTION_PRIORITY = {
    "hepatotoxicity": 0,      # Highest priority
    "outcome_and_management": 0,
    "mechanism_of_injury": 0,          
    "introduction": 1,        
    "background": 1,          
    "case_report": 2,
    "comment": 10,
    "key_points": 2,
    "chemical_formula_and_structure": 99,
    "product_information": 3,
    "annotated_bibliography": 99,  # Essentially exclude from results
    "other_reference_links": 99,
    "untitled": 99,
}

# Build drug name list dynamically from document IDs
def build_drug_list(metadata):
    """
    Build a list of drug names from document IDs in metadata.
    
    Args:
        metadata (dict): Chunk metadata dictionary
        
    Returns:
        set: Set of unique drug names
    """
    drug_names = set()
    
    # Extract unique document IDs
    for idx, info in metadata.items():
        doc_id = info.get("doc_id", "")
        if doc_id:
            # Extract drug name from document ID
            drug_name = extract_drug_name_from_doc_id(doc_id)
            if drug_name:
                drug_names.add(drug_name)
    
    return drug_names

def extract_drug_name_from_doc_id(doc_id):
    """
    More robust drug name extraction.
    
    Args:
        doc_id (str): Document identifier
        
    Returns:
        str: Lowercase drug name
    """
    # Remove file extension if present
    doc_id = doc_id.split('.')[0]
    
    # Convert to lowercase
    drug_name = doc_id.lower()
    
    # Strip any numeric suffixes or prefixes
    drug_name = re.sub(r'(^[0-9]+_|_[0-9]+$)', '', drug_name)
    
    return drug_name

def extract_potential_drugs_from_query(query, metadata):
    """
    Enhanced drug name extraction from query with improved matching.
    
    Args:
        query (str): The user query
        metadata (dict): The chunk metadata containing document IDs
        
    Returns:
        list: List of potential drug names found in the query
    """
    # Build drug list from corpus
    drug_names = build_drug_list(metadata)
    
    # Normalize query text
    query_lower = query.lower()
    
    # Extract words for potential partial matching
    query_words = re.findall(r'\b\w+\b', query_lower)
    
    # Simple word stemming/normalization to improve matching
    normalized_tokens = []
    for token in query_words:
        # Simple stemming rules for common drug name endings
        for suffix in ['in', 'ine', 'ide', 'one', 'ol', 'il', 'ate']:
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                normalized_tokens.append(token[:-len(suffix)])
        normalized_tokens.append(token)
    
    # Check for exact and stemmed drug name matches
    found_drugs = []
    for drug in drug_names:
        # Check for exact match
        if drug in query_lower:
            found_drugs.append(drug)
            continue
            
        # Check for stemmed/partial matches
        drug_stem = drug
        for suffix in ['in', 'ine', 'ide', 'one', 'ol', 'il', 'ate']:
            if drug.endswith(suffix) and len(drug) > len(suffix) + 2:
                drug_stem = drug[:-len(suffix)]
                
        if drug_stem in normalized_tokens and drug not in found_drugs:
            found_drugs.append(drug)
    
    return found_drugs

def answer_question(query, index, metadata, top_k=10, target_drug=None, return_chunks=False, 
                   llm_type='local', openai_model="gpt-4o", claude_model="claude-3-opus-20240229"):
    """
    Retrieves relevant sections that directly answer the question about drug-induced liver injury.
    Uses the specified model to identify and extract the most relevant content.

    Args:
        query (str): Clinical question
        index (FAISS): FAISS index
        metadata (dict): Chunk metadata with string keys
        top_k (int): How many chunks to retrieve
        target_drug (str, optional): Explicitly specified drug to target
        return_chunks (bool, optional): Whether to return retrieved chunks for analysis
        llm_type (str): Which model to use ('local', 'claude', or 'openai')
        openai_model (str): OpenAI model to use if llm_type is 'openai'
        claude_model (str): Claude model to use if llm_type is 'claude'

    Returns:
        str or tuple: If return_chunks is False, returns just the answer string.
                     If return_chunks is True, returns (answer, retrieved_chunks) tuple.
    """
    # Check if LLM type is set in environment variables
    env_llm_type = os.environ.get("LLM_TYPE", "").lower()
    if env_llm_type in ['local', 'claude', 'openai']:
        llm_type = env_llm_type
    
    # Check if model is specified in environment variables
    if llm_type == 'claude':
        env_model = os.environ.get("CLAUDE_MODEL")
        if env_model:
            claude_model = env_model
        # Check if Anthropic is available
        if claude_client is None:
            print("WARNING: Claude API was requested but client initialization failed.")
            print("Falling back to local model.")
            llm_type = 'local'
    elif llm_type == 'openai':
        env_model = os.environ.get("OPENAI_MODEL")
        if env_model:
            openai_model = env_model
        # Check OpenAI availability
        if openai_client is None:
            print("WARNING: OpenAI API was requested but client initialization failed.")
            print("Falling back to local model.")
            llm_type = 'local'
            
    # Use target drug if provided, otherwise try to extract from query
    if target_drug:
        query_drugs = [target_drug.lower()]
        print(f"Using specified drug: {target_drug}")
    else:
        query_drugs = extract_potential_drugs_from_query(query, metadata)
        
        if query_drugs:
            print(f"Extracted drugs from query: {', '.join(query_drugs)}")
        else:
            print("No drugs identified in query")
    
    # FILTER APPROACH: First, collect all chunks for the target drug(s)
    retrieved_chunks = []
    
    if query_drugs:
        drug_specific_chunks = []
        drug_chunk_indices = {}  # Map to store original metadata indices for drug chunks
        
        # Find all chunks for the target drug(s)
        for idx_str, info in metadata.items():
            doc_id = info.get("doc_id", "")
            chunk_drug = extract_drug_name_from_doc_id(doc_id)
            
            # Check if this document is about any of our target drugs
            if any(drug == chunk_drug for drug in query_drugs):
                chunk = info.copy()
                idx = len(drug_specific_chunks)
                drug_specific_chunks.append(chunk)
                drug_chunk_indices[idx] = int(idx_str)  # Store mapping
        
        print(f"Found {len(drug_specific_chunks)} chunks for drugs: {', '.join(query_drugs)}")
        
        # If we found drug-specific chunks, use vector search within those chunks
        if drug_specific_chunks:
            # Create embeddings for drug-specific chunks
            query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
            
            # Get texts for encoding
            texts = [chunk["text"] for chunk in drug_specific_chunks]
            
            # Encode all chunks at once (more efficient than one-by-one)
            chunk_embeddings = embedder.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                normalize_embeddings=True,
                convert_to_numpy=True
            )
            
            # Create search index for drug-specific chunks
            dim = chunk_embeddings.shape[1]
            temp_index = faiss.IndexFlatIP(dim)
            temp_index.add(np.array(chunk_embeddings).astype("float32"))
            
            # Perform search within drug-specific chunks
            temp_k = min(top_k, len(drug_specific_chunks))
            D, I = temp_index.search(query_emb.astype("float32"), temp_k)
            
            # Map results back to original chunks and add to retrieved chunks
            for i, idx in enumerate(I[0]):
                c = drug_specific_chunks[idx].copy()
                
                # Add score from vector similarity
                c["score"] = float(D[0][i])
                
                # Apply section boosting
                section_lower = c["section"].lower() if "section" in c else "unknown"
                section_priority = SECTION_PRIORITY.get(section_lower, 99)
                
                # Extra boost for matching keywords in section titles
                if "mechanism" in query.lower() and "mechanism" in section_lower:
                    c["score"] += 0.5  # Extra boost for mechanism questions
                elif "outcome" in query.lower() and "outcome" in section_lower:
                    c["score"] += 0.5  # Extra boost for outcome questions
                elif "hepatotoxicity" in query.lower() and "hepatotoxicity" in section_lower:
                    c["score"] += 0.5  # Extra boost for hepatotoxicity questions
                elif section_priority <= 1:  # Top priority sections
                    c["score"] += 0.3
                elif section_priority <= 2:  # Medium priority
                    c["score"] += 0.1
                
                retrieved_chunks.append(c)
            
            print("Retrieved chunks about target drug(s):")
            for i, c in enumerate(retrieved_chunks[:min(5, len(retrieved_chunks))]):
                print(f"  {i+1}. {c['doc_id']} - {c['section']} (Score: {c['score']:.2f})")
        else:
            # No chunks found for the specified drug(s) - return "not found" message
            drug_name = ', '.join(query_drugs)
            answer = f"No hepatotoxicity information available for {drug_name} in the LiverTox database. Please verify the drug name is correct."
            
            if return_chunks:
                return answer, []
            return answer
    
    # If no drug-specific chunks were found or no drugs were specified, fall back to regular search
    if not retrieved_chunks:
        print("Using general vector search (no drug filtering)...")
        query_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        
        # Retrieve more chunks than needed (we'll filter and re-rank)
        expanded_k = min(top_k * 3, len(metadata))
        D, I = index.search(query_emb.astype("float32"), expanded_k)
        
        # Score and rank chunks with stronger drug-based boosting
        for i, idx in enumerate(I[0]):
            idx_str = str(idx)
            if idx_str in metadata:
                c = metadata[idx_str].copy()
                
                # Base score from vector similarity
                base_score = float(D[0][i])
                c["score"] = base_score
                
                # Extract drug from document ID for matching
                doc_id = c["doc_id"]
                chunk_drug = extract_drug_name_from_doc_id(doc_id)
                
                # Apply much stronger drug-specific boost
                if query_drugs:
                    for drug in query_drugs:
                        if drug == chunk_drug:
                            # Very strong boost for exact match to target drug
                            c["score"] += 0.8  # Increased from 0.3 for stronger drug prioritization
                        elif drug in c["text"].lower():
                            # Minor boost if drug is mentioned in text
                            c["score"] += 0.2  # Increased from 0.1
                
                # Add section priority boost based on query keywords
                section_lower = c["section"].lower() if "section" in c else "unknown"
                section_priority = SECTION_PRIORITY.get(section_lower, 99)
                
                # Extra boost for matching keywords in section titles
                if "mechanism" in query.lower() and "mechanism" in section_lower:
                    c["score"] += 0.5  # Extra boost for mechanism questions
                elif "outcome" in query.lower() and "outcome" in section_lower:
                    c["score"] += 0.5  # Extra boost for outcome questions
                elif "hepatotoxicity" in query.lower() and "hepatotoxicity" in section_lower:
                    c["score"] += 0.5  # Extra boost for hepatotoxicity questions
                elif section_priority <= 1:  # Top priority sections
                    c["score"] += 0.3
                elif section_priority <= 2:  # Medium priority
                    c["score"] += 0.1
                
                retrieved_chunks.append(c)

        # Sort by score (descending)
        retrieved_chunks.sort(key=lambda c: -c["score"])
        
        # Take top chunks after re-ranking
        retrieved_chunks = retrieved_chunks[:top_k]

    # Debug: Print the first few characters of each retrieved chunk
    for i, c in enumerate(retrieved_chunks):
        text_len = len(c.get("text", ""))
        preview = c.get("text", "")[:200] + "..." if len(c.get("text", "")) > 200 else c.get("text", "")
        print(f"  Chunk {i+1}: {c['doc_id']} - {c['section']} - {text_len} chars")
        print(f"  Preview: {preview}")

    # Format the retrieved sections
    formatted_chunks = []
    for i, c in enumerate(retrieved_chunks):
        section_title = c["section"].replace("_", " ").title() if "section" in c else "Unknown"
        doc_id = c["doc_id"]
        formatted_chunks.append(f"[Source {i+1}: {doc_id}]\n{c['text']}")
    
    context = "\n\n".join(formatted_chunks)
    
    # Debug: Print context size
    
    # Check if context is empty
    if not context.strip():
        print("WARNING: Empty context - no chunks provided meaningful content")
        drug_name = target_drug if target_drug else query_drugs[0] if query_drugs else "this drug"
        answer = f"No relevant information found for {drug_name} regarding this question."
        
        if return_chunks:
            return answer, retrieved_chunks
        return answer

    # Create explicit combined question with drug name
    if target_drug:
        # Replace "this drug" with actual drug name
        combined_question = query.replace("this drug", target_drug)
        # If drug name still not in question, prepend it
        if target_drug.lower() not in combined_question.lower():
            combined_question = f"Regarding {target_drug}: {combined_question}"
    else:
        combined_question = query

    # Debug: Print question information
    
    # Process based on selected LLM type
    if llm_type == 'openai' and openai_client:
        # Format prompt for API call
        openai_prompt = f"""You are a clinical pharmacology expert answering questions about drug-induced liver injury. 
Answer this question using ONLY information from the provided context: "{combined_question}"

CONTEXT:
{context}

Your answer should be concise, factual, and based solely on the information in the context. If the answer isn't clearly stated in the context, indicate this."""


        try:
            # Get API response
            response = openai_client.chat.completions.create(
                model=openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant specializing in clinical pharmacology, particularly drug-induced liver injury. You'll answer questions based solely on the context provided."},
                    {"role": "user", "content": openai_prompt}
                ],
                max_tokens=500,
                temperature=0.1
            )

            # Extract response content
            answer = response.choices[0].message.content
            
            # Fall back if no relevant content found
            if not answer or len(answer) < 20:
                drug_name = target_drug if target_drug else query_drugs[0] if query_drugs else "this drug"
                answer = f"No relevant information found for {drug_name} regarding this question."
        except Exception as e:
            print(f"ERROR during OpenAI API call: {str(e)}")
            drug_name = target_drug if target_drug else query_drugs[0] if query_drugs else "this drug"
            answer = f"Error generating response for {drug_name}. Technical issue: {str(e)}"
    
    elif llm_type == 'claude' and claude_client:
        # Format prompt for API call
        claude_prompt = f"""You are a clinical pharmacology expert answering questions about drug-induced liver injury. 
Answer this question using ONLY information from the provided context: "{combined_question}"

CONTEXT:
{context}

Your answer should be concise, factual, and based solely on the information in the context. If the answer isn't clearly stated in the context, indicate this."""


        try:
            # Get API response
            response = claude_client.messages.create(
                model=claude_model,
                max_tokens=500,
                temperature=0.1,
                system="You are Claude, a helpful AI assistant that specializes in clinical pharmacology, particularly drug-induced liver injury. You'll answer questions based solely on the context provided.",
                messages=[
                    {"role": "user", "content": claude_prompt}
                ]
            )

            # Extract response content
            answer = response.content[0].text
            
            # Fall back if no relevant content found
            if not answer or len(answer) < 20:
                drug_name = target_drug if target_drug else query_drugs[0] if query_drugs else "this drug"
                answer = f"No relevant information found for {drug_name} regarding this question."
        except Exception as e:
            print(f"ERROR during Claude API call: {str(e)}")
            drug_name = target_drug if target_drug else query_drugs[0] if query_drugs else "this drug"
            answer = f"Error generating response for {drug_name}. Technical issue: {str(e)}"
    else:
        # Use local LLM (original code)
        # Mistral 7B Instruct specific prompt format
        prompt = f"[INST] Answer this question using ONLY information from the provided context: \"{combined_question}\"\n\nCONTEXT:\n{context}\n\nYour answer should be concise, factual, and based solely on the information in the context. If the answer isn't clearly stated in the context, indicate this. [/INST]"
        
        
        try:
            # Get the retrieval result
            response = llm(
                prompt, 
                max_tokens=600,
                temperature=0.1,
                top_p=0.95,
                top_k=40,
                repeat_penalty=1.1,
                stop=["</s>", "[INST]"]  # Mistral specific stop tokens
            )

            answer = response["choices"][0]["text"].strip()
            
            # Fall back if no relevant content found
            if not answer or len(answer) < 20:
                drug_name = target_drug if target_drug else query_drugs[0] if query_drugs else "this drug"
                answer = f"No relevant information found for {drug_name} regarding this question."
        except Exception as e:
            print(f"ERROR during LLM generation: {str(e)}")
            drug_name = target_drug if target_drug else query_drugs[0] if query_drugs else "this drug"
            answer = f"Error generating response for {drug_name}. Technical issue: {str(e)}"
    
    if return_chunks:
        return answer, retrieved_chunks
    return answer