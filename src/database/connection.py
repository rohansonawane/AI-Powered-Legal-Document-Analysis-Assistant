import os
from dotenv import load_dotenv
import psycopg2
from psycopg2.extras import RealDictCursor, Json
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import json
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def get_db_config() -> Dict[str, str]:
    """
    Get database configuration from environment variables.
    Returns default values if not set.
    """
    return {
        'dbname': os.getenv('DB_NAME', 'legal_docs'),
        'user': os.getenv('DB_USER', 'postgres'),
        'password': os.getenv('DB_PASSWORD', 'postgres'),
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': os.getenv('DB_PORT', '5432')
    }

def get_db_connection():
    """Get database connection with proper configuration and error handling."""
    try:
        config = get_db_config()
        logger.info(f"Connecting to database: {config['dbname']} on {config['host']}:{config['port']}")
        
        conn = psycopg2.connect(
            **config,
            cursor_factory=RealDictCursor
        )
        
        # Test connection
        with conn.cursor() as cur:
            cur.execute('SELECT version();')
            version = cur.fetchone()
            logger.info(f"Connected to PostgreSQL: {version['version']}")
        
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

def init_db():
    """Initialize the database with required tables and extensions."""
    conn = None
    try:
        conn = get_db_connection()
        with conn.cursor() as cur:
            # Enable pgvector extension
            cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
            logger.info("Vector extension enabled")
            
            # Create documents table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            
            # Create document_chunks table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id SERIAL PRIMARY KEY,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector(1536),
                    metadata JSONB,
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
                );
            ''')
            
            # Create feedback table
            cur.execute('''
                CREATE TABLE IF NOT EXISTS feedback (
                    id SERIAL PRIMARY KEY,
                    feedback_type TEXT NOT NULL,
                    content TEXT NOT NULL,
                    user_email TEXT,
                    document_name TEXT,
                    rating INTEGER CHECK (rating >= 1 AND rating <= 5),
                    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                );
            ''')
            
            # Create indexes for better performance
            cur.execute('CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding ON document_chunks USING ivfflat (embedding vector_cosine_ops)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_document_chunks_title ON document_chunks (title)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_document_chunks_metadata ON document_chunks USING GIN (metadata)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_feedback_type ON feedback (feedback_type)')
            cur.execute('CREATE INDEX IF NOT EXISTS idx_feedback_rating ON feedback (rating)')
            
            conn.commit()
            logger.info("Database initialized successfully")
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error initializing database: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def store_document(title: str, content: str, metadata: dict) -> int:
    """
    Store a document in the database.
    
    Args:
        title: Document title
        content: Document content
        metadata: Document metadata
        
    Returns:
        Document ID
    """
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Convert metadata to JSON
        metadata_json = Json(metadata)
        
        # Insert document
        cursor.execute("""
            INSERT INTO documents (title, content, metadata)
            VALUES (%s, %s, %s)
            RETURNING id;
        """, (title, content, metadata_json))
        
        doc_id = cursor.fetchone()['id']
        conn.commit()
        
        return doc_id
    except Exception as e:
        if conn:
            conn.rollback()
        raise Exception(f"Error storing document: {str(e)}")
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()

def store_document_chunk(title: str, content: str, embedding: List[float], metadata: Optional[Dict] = None) -> int:
    """
    Store a document chunk in the database.
    
    Args:
        title: Title of the document
        content: Content of the chunk
        embedding: Vector embedding of the chunk
        metadata: Optional metadata
        
    Returns:
        ID of the stored chunk
    """
    conn = None
    cur = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Convert embedding to proper format
        embedding_str = str(embedding).replace(" ", "")  # Remove spaces for proper vector format
        
        # Insert chunk and get ID
        cur.execute("""
            INSERT INTO document_chunks (title, content, embedding, metadata)
            VALUES (%s, %s, %s::vector, %s)
            RETURNING id;
        """, (title, content, embedding_str, Json(metadata) if metadata else None))
        
        # Get the returned ID
        result = cur.fetchone()
        if not result:
            raise Exception("Failed to get chunk ID after insertion")
            
        chunk_id = result['id']  # Access using column name instead of index
        conn.commit()
        return chunk_id
        
    except Exception as e:
        logger.error(f"Error storing document chunk: {str(e)}")
        if conn:
            conn.rollback()
        raise
    finally:
        if cur:
            cur.close()
        if conn:
            conn.close()

def search_similar_chunks(query_embedding: List[float], current_doc_title: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar document chunks using vector similarity.
    Implements efficient vector search with proper indexing.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Validate input
        if not query_embedding:
            raise ValueError("Query embedding is required")
        
        # Convert query embedding to proper vector format
        query_array = np.array(query_embedding)
        if query_array.shape[0] != 1536:
            raise ValueError(f"Invalid query embedding dimension: {query_array.shape[0]}, expected 1536")
        
        # Format the vector properly for PostgreSQL
        query_vector = f"[{','.join(map(str, query_array))}]"
        
        # Log the search parameters
        logger.info(f"Searching for chunks in document: {current_doc_title}")
        
        # Search using cosine similarity, strictly filtered by current document title
        cur.execute("""
            SELECT 
                title,
                content,
                metadata,
                1 - (embedding <=> %s::vector) as similarity
            FROM document_chunks
            WHERE embedding IS NOT NULL
            AND title = %s
            ORDER BY similarity DESC
            LIMIT %s
        """, (query_vector, current_doc_title, top_k))
        
        results = []
        for row in cur.fetchall():
            results.append({
                'title': row['title'],
                'content': row['content'],
                'metadata': row['metadata'],
                'similarity': float(row['similarity'])
            })
        
        logger.info(f"Found {len(results)} similar chunks for document: {current_doc_title}")
        return results
    except Exception as e:
        logger.error(f"Error searching similar chunks: {str(e)}")
        logger.error(traceback.format_exc())
        return []
    finally:
        if conn:
            conn.close()

def store_feedback(feedback_type: str, content: str, user_email: str = None, 
                  document_name: str = None, rating: int = None, metadata: dict = None) -> int:
    """
    Store user feedback in the database.
    
    Args:
        feedback_type: Type of feedback (e.g., 'bug', 'suggestion', 'question')
        content: Feedback content
        user_email: Optional user email
        document_name: Optional document name
        rating: Optional rating (1-5)
        metadata: Optional additional metadata
        
    Returns:
        Feedback ID
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Validate input
        if not feedback_type or not content:
            raise ValueError("Feedback type and content are required")
        
        if rating is not None and (rating < 1 or rating > 5):
            raise ValueError("Rating must be between 1 and 5")
        
        # Insert feedback
        cur.execute("""
            INSERT INTO feedback (feedback_type, content, user_email, document_name, rating, metadata)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        """, (feedback_type, content, user_email, document_name, rating, Json(metadata)))
        
        feedback_id = cur.fetchone()['id']
        conn.commit()
        logger.info(f"Stored feedback: {feedback_type}")
        return feedback_id
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error storing feedback: {str(e)}")
        raise
    finally:
        if conn:
            conn.close()

def get_feedback_stats() -> Dict[str, Any]:
    """
    Get feedback statistics.
    
    Returns:
        Dictionary containing feedback statistics
    """
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Get feedback counts by type
        cur.execute("""
            SELECT feedback_type, COUNT(*) as count
            FROM feedback
            GROUP BY feedback_type
        """)
        
        type_counts = {row[0]: row[1] for row in cur.fetchall()}
        
        # Get average rating
        cur.execute("""
            SELECT AVG(rating) as avg_rating
            FROM feedback
            WHERE rating IS NOT NULL
        """)
        
        avg_rating = cur.fetchone()[0] or 0
        
        return {
            'type_counts': type_counts,
            'avg_rating': round(avg_rating, 2) if avg_rating else 0,
            'total_feedback': sum(type_counts.values())
        }
    except Exception as e:
        print(f"Error getting feedback stats: {str(e)}")
        return {'type_counts': {}, 'avg_rating': 0, 'total_feedback': 0}
    finally:
        if conn:
            conn.close()

def clear_database():
    """
    Clear all data from the database tables.
    This will remove all documents, chunks, and feedback.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        
        # Disable foreign key checks temporarily
        cur.execute("SET session_replication_role = 'replica';")
        
        # Clear tables if they exist
        tables = ['document_chunks', 'documents', 'feedback']
        for table in tables:
            try:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE;")
                logger.info(f"Cleared table: {table}")
            except Exception as e:
                logger.warning(f"Table {table} does not exist or could not be cleared: {str(e)}")
                conn.rollback()  # Rollback on error
                continue
        
        # Re-enable foreign key checks
        cur.execute("SET session_replication_role = 'origin';")
        
        conn.commit()
        logger.info("Database cleared successfully")
        
    except Exception as e:
        if conn:
            conn.rollback()
        logger.error(f"Error clearing database: {str(e)}")
        raise
    finally:
        if conn:
            conn.close() 