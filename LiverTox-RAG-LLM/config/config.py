
import os
from typing import Dict, Any

# Model configurations
MODEL_CONFIG = {
    'embedding_model': 'dmis-lab/biobert-base-cased-v1.1',
    'chunk_size': 512,
    'chunk_overlap': 50,
    'max_context_length': 4000,
    'top_k_retrieval': 3,
}

# Section priority for retrieval
SECTION_PRIORITY = {
    "hepatotoxicity": 0,
    "outcome_and_management": 0,
    "mechanism_of_injury": 0,
    "introduction": 1,
    "background": 1,
    "case_report": 2,
    "drug_interactions": 2,
    "nonclinical_toxicity": 3,
    "references": 4,
    "chemical_structure": 5,
}

# File paths
DATA_PATHS = {
    'livertox_base': 'data/',
    'embeddings_dir': 'embeddings/',
    'models_dir': 'models/',
    'output_dir': 'output/',
}

# LLM specific configurations
LLM_CONFIGS = {
    'openai': {
        'model': os.environ.get('OPENAI_MODEL', 'gpt-4o'),
        'temperature': 0.0,
        'max_tokens': 1000,
    },
    'claude': {
        'model': os.environ.get('CLAUDE_MODEL', 'claude-3-opus-20240229'),
        'temperature': 0.0,
        'max_tokens': 1000,
    },
    'gemini': {
        'model': os.environ.get('GEMINI_MODEL', 'gemini-pro'),
        'temperature': 0.0,
        'max_tokens': 1000,
    },
    'local': {
        'model_path': 'models/llama-2-7b-chat.gguf',
        'temperature': 0.0,
        'max_tokens': 1000,
        'context_length': 4096,
    }
}

def get_config() -> Dict[str, Any]:
    """
    Get complete configuration dictionary
    
    Returns:
        Dictionary with all configuration settings
    """
    return {
        'model': MODEL_CONFIG,
        'sections': SECTION_PRIORITY,
        'paths': DATA_PATHS,
        'llm': LLM_CONFIGS,
        'environment': {
            'llm_type': os.environ.get('LLM_TYPE', 'local'),
            'debug': os.environ.get('DEBUG', 'False').lower() == 'true',
            'verbose': os.environ.get('VERBOSE', 'False').lower() == 'true',
        }
    }

def validate_api_keys():
    """
    Check if required API keys are available for selected LLM type
    """
    llm_type = os.environ.get('LLM_TYPE', 'local')
    
    if llm_type == 'openai':
        if not os.environ.get('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY not found in environment variables")
    
    elif llm_type == 'claude':
        if not os.environ.get('ANTHROPIC_API_KEY'):
            raise ValueError("ANTHROPIC_API_KEY not found in environment variables")
    
    elif llm_type == 'gemini':
        if not os.environ.get('GOOGLE_API_KEY'):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    return True

def setup_directories():
    """
    Create necessary directories if they don't exist
    """
    for path in DATA_PATHS.values():
        os.makedirs(path, exist_ok=True)