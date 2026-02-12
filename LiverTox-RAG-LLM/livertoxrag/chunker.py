# livertoxrag/chunker.py
# Splits parsed text into manageable chunks

import textwrap

def chunk_text(doc_id: str, text: str, section_title="Untitled", max_chars=1500):
    """
    Chunking strategy that preserves paragraph structure and avoids tiny chunks.
    
    Args:
        doc_id (str): Document identifier
        text (str): Text content to be chunked
        section_title (str): Title of the section this text belongs to
        max_chars (int): Approximate maximum characters per chunk
        
    Returns:
        list: List of chunk dictionaries with doc_id, chunk_id, section, and text fields
    """
    paragraphs = text.split("\n")
    chunks = []
    chunk_id = 0
    current_chunk = ""
    
    for p in paragraphs:
        p = p.strip()
        if not p:  # Skip empty paragraphs
            continue
            
        # If this paragraph would make the chunk too large, save current chunk and start new
        if len(current_chunk) + len(p) > max_chars and current_chunk:
            chunks.append({
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}_{chunk_id}",
                "section": section_title,
                "text": current_chunk.strip()
            })
            chunk_id += 1
            current_chunk = p
        else:
            # Add paragraph to current chunk
            if current_chunk:
                current_chunk += "\n\n" + p  # Double newline to preserve paragraph structure
            else:
                current_chunk = p
    
    # Add the last chunk if it contains anything
    if current_chunk:
        chunks.append({
            "doc_id": doc_id,
            "chunk_id": f"{doc_id}_{chunk_id}",
            "section": section_title,
            "text": current_chunk.strip()
        })
    
    return chunks