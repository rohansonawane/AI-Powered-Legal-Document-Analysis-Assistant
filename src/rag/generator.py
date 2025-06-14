import os
from typing import List, Dict, Any
import openai
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def generate_response(query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
    """
    Generate a response based on the query and relevant document chunks.
    
    Args:
        query: The user's question
        relevant_chunks: List of relevant document chunks with metadata
        
    Returns:
        Generated response
    """
    try:
        if not relevant_chunks:
            return "I couldn't find any relevant information in the document to answer your question."
            
        # Prepare context from chunks
        context = "\n\n".join([
            f"Document: {chunk['title']}\nContent: {chunk['content']}"
            for chunk in relevant_chunks
        ])
        
        # Create system message based on query type
        if any(word in query.lower() for word in ['what is', 'about', 'summarize', 'overview']):
            system_message = """You are a helpful legal document assistant. Your task is to provide a comprehensive overview 
            of the document's content. Focus on:
            1. The main purpose and type of document
            2. Key parties involved
            3. Major terms and conditions
            4. Important dates and deadlines
            5. Critical obligations and responsibilities
            Be thorough but concise, and cite specific sections when relevant."""
        else:
            system_message = """You are a helpful legal document assistant. Your task is to provide accurate and detailed 
            answers based on the document content. Always:
            1. Cite specific sections or clauses when possible
            2. Explain legal terms in simple language
            3. Provide context for your answers
            4. Be precise and professional
            5. Include relevant details from the document"""
        
        # Generate response using OpenAI
        response = client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": f"Question: {query}\n\nContext:\n{context}"}
            ],
            temperature=0.3,  # Lower temperature for more focused responses
            max_tokens=1000
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I encountered an error while generating the response. Please try again."

def format_context(context: List[Dict[str, Any]]) -> str:
    """
    Format context chunks for the prompt.
    
    Args:
        context: List of context chunks with metadata
        
    Returns:
        Formatted context string
    """
    formatted_context = []
    for i, chunk in enumerate(context, 1):
        formatted_context.append(
            f"[Chunk {i} - Relevance: {chunk['relevance']:.2f}]\n"
            f"Source: {chunk['title']}\n"
            f"Content: {chunk['content']}\n"
        )
    return "\n".join(formatted_context)

def generate_feedback_prompt(query: str, response: str, correction: str) -> str:
    """
    Generate a prompt for fine-tuning based on user feedback.
    
    Args:
        query: Original question
        response: Generated response
        correction: User's correction
        
    Returns:
        Fine-tuning prompt
    """
    return f"""Original Question: {query}
    Generated Response: {response}
    User Correction: {correction}
    
    Please provide an improved response that incorporates the user's correction while maintaining accuracy and context-awareness.""" 