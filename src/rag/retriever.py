import os
from typing import List, Dict, Any
import numpy as np
from src.database.connection import search_similar_chunks
from src.embeddings.generator import generate_embeddings, normalize_embedding
import logging
import traceback

logger = logging.getLogger(__name__)

def retrieve_relevant_docs(query: str, current_doc_title: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve relevant document chunks for a given query using hybrid search.
    
    Args:
        query: The user's question
        current_doc_title: Title of the currently loaded document
        top_k: Number of chunks to retrieve
        
    Returns:
        List of relevant document chunks with their metadata
    """
    try:
        # Generate query embedding
        query_embedding = generate_embeddings([query])[0]
        
        # Normalize the embedding
        query_embedding = normalize_embedding(query_embedding)
        
        # Get similar chunks from database, filtered by current document
        similar_chunks = search_similar_chunks(query_embedding, current_doc_title, top_k=top_k)
        
        if not similar_chunks:
            logger.warning(f"No similar chunks found for query in document: {current_doc_title}")
            return []
            
        # Sort chunks by similarity score
        similar_chunks.sort(key=lambda x: x['similarity'], reverse=True)
        
        # For general questions like "what is this document", be more lenient
        is_general_question = any(word in query.lower() for word in ['what is', 'about', 'summarize', 'overview'])
        
        if is_general_question:
            # For general questions, return all chunks from the current document
            relevant_chunks = similar_chunks
            logger.info(f"General question detected, returning all {len(relevant_chunks)} chunks from current document")
        else:
            # For specific questions, use a moderate threshold
            threshold = 0.3
            relevant_chunks = [chunk for chunk in similar_chunks if chunk['similarity'] > threshold]
            
            if not relevant_chunks:
                # If no chunks meet threshold, return top 2
                relevant_chunks = similar_chunks[:2]
                logger.info("No chunks met threshold, returning top 2 chunks")
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks with similarity scores: {[chunk['similarity'] for chunk in relevant_chunks]}")
        return relevant_chunks
        
    except Exception as e:
        logger.error(f"Error retrieving relevant docs: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def format_context(documents: List[Dict]) -> str:
    """
    Format retrieved documents into a context string for the LLM.
    
    Args:
        documents: List of retrieved documents
        
    Returns:
        Formatted context string
    """
    context = []
    for doc in documents:
        context.append(f"Document: {doc['title']}\n")
        context.append(f"Content: {doc['content']}\n")
        if doc['metadata']:
            context.append(f"Metadata: {doc['metadata']}\n")
        context.append("---\n")
    
    return "\n".join(context) 