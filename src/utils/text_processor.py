from typing import List, Dict, Any
import re
from nltk.tokenize import sent_tokenize
import nltk

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks with improved handling of document structure.
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum size of each chunk
        overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        # First, try to split by sections if they exist
        sections = text.split('\n\n')
        
        # If we have clear sections, process them
        if len(sections) > 1:
            chunks = []
            current_chunk = []
            current_size = 0
            
            for section in sections:
                section = section.strip()
                if not section:
                    continue
                    
                # If section is a header or title, add it to all following chunks
                if len(section.split()) <= 10 and section.isupper():
                    header = section
                    continue
                
                # Add header to section if we have one
                if 'header' in locals():
                    section = f"{header}\n{section}"
                
                # If section is too long, split it
                if len(section) > chunk_size:
                    words = section.split()
                    temp_chunk = []
                    temp_size = 0
                    
                    for word in words:
                        if temp_size + len(word) + 1 > chunk_size:
                            if temp_chunk:
                                chunks.append(' '.join(temp_chunk))
                            temp_chunk = [word]
                            temp_size = len(word)
                        else:
                            temp_chunk.append(word)
                            temp_size += len(word) + 1
                    
                    if temp_chunk:
                        chunks.append(' '.join(temp_chunk))
                else:
                    # If current chunk + section would exceed size, save current and start new
                    if current_size + len(section) > chunk_size:
                        if current_chunk:
                            chunks.append('\n'.join(current_chunk))
                        current_chunk = [section]
                        current_size = len(section)
                    else:
                        current_chunk.append(section)
                        current_size += len(section)
            
            # Add any remaining chunk
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
        else:
            # If no clear sections, use sliding window approach
            words = text.split()
            chunks = []
            current_chunk = []
            current_size = 0
            
            for word in words:
                if current_size + len(word) + 1 > chunk_size:
                    if current_chunk:
                        chunks.append(' '.join(current_chunk))
                    # Start new chunk with overlap
                    overlap_words = current_chunk[-overlap:] if overlap > 0 else []
                    current_chunk = overlap_words + [word]
                    current_size = sum(len(w) + 1 for w in current_chunk)
                else:
                    current_chunk.append(word)
                    current_size += len(word) + 1
            
            if current_chunk:
                chunks.append(' '.join(current_chunk))
        
        # Ensure we have at least one chunk
        if not chunks:
            chunks = [text]
        
        # Clean up chunks
        chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
        
        # Log chunk information
        logger.info(f"Created {len(chunks)} chunks from text")
        for i, chunk in enumerate(chunks):
            logger.info(f"Chunk {i+1} length: {len(chunk)} characters")
        
        return chunks
        
    except Exception as e:
        logger.error(f"Error chunking text: {str(e)}")
        # Fallback to simple chunking if there's an error
        return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-overlap)]

def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    Enhanced for legal documents.
    
    Args:
        text: Raw text content
        
    Returns:
        Cleaned text
    """
    # Replace multiple newlines with single newline
    text = '\n'.join(line.strip() for line in text.split('\n') if line.strip())
    
    # Replace multiple spaces with single space
    text = ' '.join(text.split())
    
    # Remove special characters but keep legal document markers
    legal_chars = '.,;:!?()[]{}§¶©®™'
    text = ''.join(char for char in text if char.isprintable() or char in legal_chars)
    
    # Normalize whitespace around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    
    # Normalize section numbers
    text = re.sub(r'(\d+)\s*\.\s*(\d+)', r'\1.\2', text)
    
    return text

def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extract metadata from legal document text.
    
    Args:
        text: Document text
        
    Returns:
        Dictionary of metadata
    """
    metadata = {
        'section_count': len(re.findall(r'Section \d+\.', text)),
        'article_count': len(re.findall(r'Article \d+\.', text)),
        'clause_count': len(re.findall(r'Clause \d+\.', text)),
        'sentence_count': len(sent_tokenize(text)),
        'word_count': len(text.split()),
        'char_count': len(text)
    }
    
    # Extract document type if present
    doc_types = {
        'contract': r'contract|agreement',
        'legislation': r'act|statute|regulation',
        'case_law': r'case|judgment|opinion',
        'legal_memo': r'memorandum|memo'
    }
    
    for doc_type, pattern in doc_types.items():
        if re.search(pattern, text.lower()):
            metadata['document_type'] = doc_type
            break
    
    return metadata 