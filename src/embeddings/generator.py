import os
from openai import OpenAI
from typing import List, Union, Dict, Any
import numpy as np
from dotenv import load_dotenv
import streamlit as st
import tiktoken
from src.utils.cache import get_cache, set_cache
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get API key from environment or Streamlit secrets
api_key = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')
if not api_key:
    st.error(
        "OpenAI API key not found. Please set it in one of these ways:\n"
        "1. Create a .env file with: OPENAI_API_KEY=your_api_key_here\n"
        "2. Set it in your terminal: export OPENAI_API_KEY=your_api_key_here\n"
        "3. Enter it below:"
    )
    api_key = st.text_input("OpenAI API Key", type="password")
    if not api_key:
        st.stop()

client = OpenAI(api_key=api_key)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list:
    """
    Split text into overlapping chunks of specified size.
    
    Args:
        text: Text to split
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        # Initialize tokenizer
        tokenizer = tiktoken.get_encoding("cl100k_base")
        
        # Split text into sentences (rough approximation)
        sentences = text.replace('\n', ' ').split('. ')
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sentence in sentences:
            # Add period back to sentence
            sentence = sentence.strip() + '. '
            
            # Get token count for sentence
            sentence_tokens = len(tokenizer.encode(sentence))
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_size + sentence_tokens > chunk_size and current_chunk:
                chunk_text = ' '.join(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with overlap
                overlap_text = ' '.join(current_chunk[-3:])  # Use last 3 sentences for overlap
                current_chunk = [overlap_text]
                current_size = len(tokenizer.encode(overlap_text))
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_size += sentence_tokens
        
        # Add the last chunk if it exists
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        st.write(f"Created {len(chunks)} chunks from text")
        return chunks
    
    except Exception as e:
        st.error(f"Error in chunk_text: {str(e)}")
        raise

def generate_embeddings(texts: List[str], batch_size: int = 20) -> List[List[float]]:
    """
    Generate embeddings for a list of texts using OpenAI's API.
    Implements batch processing and caching for efficiency.
    
    Args:
        texts: List of text chunks to generate embeddings for
        batch_size: Number of texts to process in each batch
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        logger.warning("No texts provided for embedding generation")
        return []
    
    embeddings = []
    total_texts = len(texts)
    
    try:
        # Process texts in batches
        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}/{(total_texts + batch_size - 1)//batch_size}")
            
            # Check cache for each text in batch
            batch_embeddings = []
            texts_to_process = []
            text_indices = []
            
            for j, text in enumerate(batch):
                cache_key = f"embedding:{text[:100]}"  # Use first 100 chars as key
                cached_embedding = get_cache(cache_key)
                
                if cached_embedding is not None:
                    batch_embeddings.append(cached_embedding)
                else:
                    texts_to_process.append(text)
                    text_indices.append(j)
            
            # Generate embeddings for uncached texts
            if texts_to_process:
                try:
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=texts_to_process
                    )
                    
                    # Process and cache new embeddings
                    for idx, embedding in zip(text_indices, response.data):
                        vector = embedding.embedding
                        batch_embeddings.insert(idx, vector)
                        
                        # Cache the embedding
                        cache_key = f"embedding:{batch[idx][:100]}"
                        set_cache(cache_key, vector)
                except Exception as e:
                    logger.error(f"Error generating embeddings: {str(e)}")
                    # Use zero vector as fallback
                    for idx in text_indices:
                        batch_embeddings.insert(idx, [0.0] * 1536)
            
            embeddings.extend(batch_embeddings)
        
        logger.info(f"Generated embeddings for {len(embeddings)} texts")
        return embeddings
    
    except Exception as e:
        logger.error(f"Error in generate_embeddings: {str(e)}")
        # Return zero vectors as fallback
        return [[0.0] * 1536 for _ in texts]

def get_embedding_dimension() -> int:
    """Get the dimension of the embedding vectors."""
    return 1536  # OpenAI's text-embedding-ada-002 dimension

def normalize_embedding(embedding: List[float]) -> List[float]:
    """
    Normalize an embedding vector to unit length.
    
    Args:
        embedding: Input embedding vector
        
    Returns:
        Normalized embedding vector
    """
    try:
        vector = np.array(embedding)
        norm = np.linalg.norm(vector)
        if norm == 0:
            return embedding
        return (vector / norm).tolist()
    except Exception as e:
        logger.error(f"Error normalizing embedding: {str(e)}")
        return embedding

def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score between 0 and 1
    """
    try:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)
    except Exception as e:
        logger.error(f"Error computing similarity: {str(e)}")
        return 0.0

def process_document_for_embeddings(text: str) -> List[List[float]]:
    """
    Process a document by chunking it and generating embeddings for each chunk.
    
    Args:
        text: Document text
        
    Returns:
        List of embedding vectors for each chunk
    """
    if not text.strip():
        raise ValueError("Empty document text provided")
        
    chunks = chunk_text(text)
    if not chunks:
        raise ValueError("No valid text chunks generated from document")
        
    return generate_embeddings(chunks) 