import os
import json
import redis
from typing import Any, Optional, List, Dict
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Redis client
try:
    redis_client = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        port=int(os.getenv('REDIS_PORT', 6379)),
        db=int(os.getenv('REDIS_DB', 0)),
        decode_responses=True
    )
    # Test connection
    redis_client.ping()
    logger.info("Successfully connected to Redis")
except Exception as e:
    logger.error(f"Error connecting to Redis: {str(e)}")
    redis_client = None

def get_cache(key: str) -> Optional[Any]:
    """
    Get value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        Cached value or None if not found
    """
    if not redis_client:
        return None
    
    try:
        value = redis_client.get(key)
        if value:
            return json.loads(value)
        return None
    except Exception as e:
        logger.error(f"Error getting from cache: {str(e)}")
        return None

def set_cache(key: str, value: Any, expire: int = 3600) -> bool:
    """
    Set value in cache with expiration.
    
    Args:
        key: Cache key
        value: Value to cache
        expire: Expiration time in seconds (default: 1 hour)
        
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        redis_client.setex(
            key,
            expire,
            json.dumps(value)
        )
        return True
    except Exception as e:
        logger.error(f"Error setting cache: {str(e)}")
        return False

def delete_cache(key: str) -> bool:
    """
    Delete value from cache.
    
    Args:
        key: Cache key
        
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        redis_client.delete(key)
        return True
    except Exception as e:
        logger.error(f"Error deleting from cache: {str(e)}")
        return False

def clear_cache() -> bool:
    """
    Clear all cached values.
    
    Returns:
        True if successful, False otherwise
    """
    if not redis_client:
        return False
    
    try:
        redis_client.flushdb()
        return True
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        return False

def get_cache_stats() -> dict:
    """
    Get cache statistics.
    
    Returns:
        Dictionary containing cache statistics
    """
    if not redis_client:
        return {
            'status': 'disconnected',
            'keys': 0,
            'memory_used': 0
        }
    
    try:
        info = redis_client.info()
        return {
            'status': 'connected',
            'keys': info.get('db0', {}).get('keys', 0),
            'memory_used': info.get('used_memory_human', '0B')
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {str(e)}")
        return {
            'status': 'error',
            'error': str(e)
        }

class Cache:
    def __init__(self):
        """Initialize Redis cache connection."""
        self.redis_client = redis_client
        self.embedding_cache = {}
        self.chunk_cache = {}
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding from cache.
        
        Args:
            text: Text to get embedding for
            
        Returns:
            Cached embedding or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            cache_key = f"embedding:{text[:100]}"  # Use first 100 chars as key
            value = self.redis_client.get(cache_key)
            if value:
                return json.loads(value)
            return None
        except Exception as e:
            logger.error(f"Error getting embedding from cache: {str(e)}")
            return None
    
    def set_embedding(self, text: str, embedding: List[float], expire: int = 3600) -> bool:
        """
        Set embedding in cache.
        
        Args:
            text: Text to cache embedding for
            embedding: Embedding vector to cache
            expire: Expiration time in seconds (default: 1 hour)
            
        Returns:
            True if successful, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            cache_key = f"embedding:{text[:100]}"  # Use first 100 chars as key
            self.redis_client.setex(
                cache_key,
                expire,
                json.dumps(embedding)
            )
            return True
        except Exception as e:
            logger.error(f"Error setting embedding in cache: {str(e)}")
            return False
    
    def get_chunk(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get document chunk from cache."""
        cache_key = f"chunk:{chunk_id}"
        
        # Check memory cache first
        if chunk_id in self.chunk_cache:
            return self.chunk_cache[chunk_id]
        
        # Check Redis cache
        cached = self.redis_client.get(cache_key)
        if cached:
            chunk = json.loads(cached)
            self.chunk_cache[chunk_id] = chunk
            return chunk
        
        return None
    
    def set_chunk(self, chunk_id: str, chunk_data: Dict[str, Any], expire: int = 3600):
        """Cache document chunk with expiration."""
        cache_key = f"chunk:{chunk_id}"
        
        # Update memory cache
        self.chunk_cache[chunk_id] = chunk_data
        
        # Update Redis cache
        self.redis_client.setex(
            cache_key,
            expire,
            json.dumps(chunk_data)
        )
    
    def clear_cache(self):
        """Clear both memory and Redis caches."""
        self.embedding_cache.clear()
        self.chunk_cache.clear()
        self.redis_client.flushdb()

# Initialize global cache instance
cache = Cache() 