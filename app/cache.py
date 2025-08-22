import asyncio
from typing import Any, Optional, Union
from aiocache import caches, Cache
from aiocache.serializers import JsonSerializer, PickleSerializer
import logging
import os

from .config import CACHE_TTL_SECONDS

logger = logging.getLogger(__name__)

# Cache configuration based on environment
def get_cache_config():
    """Get cache configuration based on environment variables"""
    
    # Check if Redis is available
    redis_url = os.getenv("REDIS_URL")
    redis_host = os.getenv("REDIS_HOST", "localhost")
    redis_port = int(os.getenv("REDIS_PORT", 6379))
    redis_db = int(os.getenv("REDIS_DB", 0))
    redis_password = os.getenv("REDIS_PASSWORD")
    
    if redis_url or (redis_host and redis_port):
        logger.info("Configuring Redis cache")
        
        # Redis configuration
        redis_config = {
            "cache": "aiocache.RedisCache",
            "endpoint": redis_host,
            "port": redis_port,
            "db": redis_db,
            "serializer": {
                "class": "aiocache.serializers.JsonSerializer"
            },
            "ttl": CACHE_TTL_SECONDS
        }
        
        if redis_url:
            redis_config["endpoint"] = redis_url
        if redis_password:
            redis_config["password"] = redis_password
            
        return {
            "default": redis_config,
            "articles": redis_config,
            "sentiment": redis_config,
            "clusters": redis_config
        }
    else:
        logger.info("Configuring in-memory cache (Redis not available)")
        
        # In-memory cache configuration
        memory_config = {
            "cache": "aiocache.SimpleMemoryCache",
            "serializer": {
                "class": "aiocache.serializers.JsonSerializer"
            },
            "ttl": CACHE_TTL_SECONDS
        }
        
        return {
            "default": memory_config,
            "articles": memory_config,
            "sentiment": memory_config,
            "clusters": memory_config
        }

# Configure caches
try:
    cache_config = get_cache_config()
    caches.set_config(cache_config)
    logger.info("Cache configuration set successfully")
except Exception as e:
    logger.error(f"Failed to configure cache: {e}")
    # Fallback to simple memory cache
    caches.set_config({
        "default": {
            "cache": "aiocache.SimpleMemoryCache",
            "ttl": CACHE_TTL_SECONDS,
        }
    })

# Get cache instances
default_cache: Cache = caches.get("default")
articles_cache: Cache = caches.get("articles") if "articles" in caches._caches else default_cache
sentiment_cache: Cache = caches.get("sentiment") if "sentiment" in caches._caches else default_cache
clusters_cache: Cache = caches.get("clusters") if "clusters" in caches._caches else default_cache

class CacheManager:
    """Enhanced cache manager with error handling and monitoring"""
    
    def __init__(self):
        self.stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "sets": 0
        }
    
    async def get(self, key: str, cache: Cache = None) -> Optional[Any]:
        """Get value from cache with error handling"""
        if cache is None:
            cache = default_cache
            
        try:
            value = await cache.get(key)
            if value is not None:
                self.stats["hits"] += 1
                logger.debug(f"Cache hit for key: {key}")
                return value
            else:
                self.stats["misses"] += 1
                logger.debug(f"Cache miss for key: {key}")
                return None
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, cache: Cache = None) -> bool:
        """Set value in cache with error handling"""
        if cache is None:
            cache = default_cache
            
        try:
            await cache.set(key, value, ttl=ttl or CACHE_TTL_SECONDS)
            self.stats["sets"] += 1
            logger.debug(f"Cache set for key: {key}")
            return True
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str, cache: Cache = None) -> bool:
        """Delete key from cache"""
        if cache is None:
            cache = default_cache
            
        try:
            await cache.delete(key)
            logger.debug(f"Cache delete for key: {key}")
            return True
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def clear(self, cache: Cache = None) -> bool:
        """Clear entire cache"""
        if cache is None:
            cache = default_cache
            
        try:
            await cache.clear()
            logger.info("Cache cleared")
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    async def exists(self, key: str, cache: Cache = None) -> bool:
        """Check if key exists in cache"""
        if cache is None:
            cache = default_cache
            
        try:
            return await cache.exists(key)
        except Exception as e:
            logger.error(f"Cache exists check error for key {key}: {e}")
            return False
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            "hits": self.stats["hits"],
            "misses": self.stats["misses"],
            "errors": self.stats["errors"],
            "sets": self.stats["sets"],
            "hit_rate_percentage": round(hit_rate, 2),
            "total_requests": total_requests
        }
    
    def reset_stats(self):
        """Reset cache statistics"""
        self.stats = {
            "hits": 0,
            "misses": 0,
            "errors": 0,
            "sets": 0
        }

# Global cache manager instance
cache_manager = CacheManager()

# Utility functions
async def get_cached_articles() -> Optional[list]:
    """Get cached articles with fallback"""
    return await cache_manager.get("articles", articles_cache)

async def cache_articles(articles: list, ttl: Optional[int] = None) -> bool:
    """Cache articles list"""
    return await cache_manager.set("articles", articles, ttl, articles_cache)

async def get_cached_sentiment(key: str) -> Optional[dict]:
    """Get cached sentiment analysis"""
    return await cache_manager.get(f"sentiment:{key}", sentiment_cache)

async def cache_sentiment(key: str, data: dict, ttl: Optional[int] = None) -> bool:
    """Cache sentiment analysis results"""
    return await cache_manager.set(f"sentiment:{key}", data, ttl, sentiment_cache)

async def get_cached_clusters(key: str) -> Optional[list]:
    """Get cached clustering results"""
    return await cache_manager.get(f"clusters:{key}", clusters_cache)

async def cache_clusters(key: str, clusters: list, ttl: Optional[int] = None) -> bool:
    """Cache clustering results"""
    return await cache_manager.set(f"clusters:{key}", clusters, ttl, clusters_cache)


async def get_all_cached_data() -> dict:
    """Retrieve all key-value pairs from every configured cache.

    Uses batch lookups where possible to minimise round trips and falls back to
    concurrent per-key retrieval when ``multi_get`` isn't supported. Each cache
    is also processed concurrently for better overall throughput.
    """

    async def fetch_cache(name: str) -> tuple[str, dict]:
        """Fetch all key/value pairs for a single cache."""
        try:
            cache = caches.get(name)
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to get cache {name}: {e}")
            return name, {}

        try:
            try:
                keys = await cache.raw("keys", "*")
            except TypeError:  # SimpleMemoryCache doesn't accept pattern
                keys = await cache.raw("keys")
        except Exception as e:  # pragma: no cover - defensive
            logger.error(f"Failed to list keys for cache {name}: {e}")
            return name, {}

        if not keys:
            return name, {}

        decoded_keys = [k.decode() if isinstance(k, bytes) else k for k in keys]

        try:
            # Preferred: single round-trip to fetch all values
            values = await cache.multi_get(decoded_keys)
        except Exception:
            # Fallback: parallel per-key fetches
            values = await asyncio.gather(
                *(cache.get(k) for k in decoded_keys),
                return_exceptions=True,
            )

        cache_content: dict[str, Any] = {}
        for key, value in zip(decoded_keys, values):
            if isinstance(value, Exception):  # pragma: no cover - defensive
                logger.error(f"Failed to retrieve {key} from {name}: {value}")
            else:
                cache_content[key] = value

        return name, cache_content

    results = await asyncio.gather(
        *(fetch_cache(n) for n in ["default", "articles", "sentiment", "clusters"])
    )
    return {name: content for name, content in results}

async def cache_health_check() -> dict:
    """Perform cache health check"""
    try:
        # Test basic operations
        test_key = "health_check_test"
        test_value = {"test": True, "timestamp": str(asyncio.get_event_loop().time())}
        
        # Test set
        set_success = await cache_manager.set(test_key, test_value, ttl=60)
        if not set_success:
            return {"status": "unhealthy", "error": "Failed to set test value"}
        
        # Test get
        retrieved_value = await cache_manager.get(test_key)
        if retrieved_value != test_value:
            return {"status": "unhealthy", "error": "Retrieved value doesn't match"}
        
        # Test delete
        delete_success = await cache_manager.delete(test_key)
        if not delete_success:
            return {"status": "unhealthy", "error": "Failed to delete test value"}
        
        # Get stats
        stats = cache_manager.get_stats()
        
        return {
            "status": "healthy",
            "cache_type": cache_config["default"]["cache"],
            "stats": stats
        }
        
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {"status": "unhealthy", "error": str(e)}

# Context manager for cache operations
class CacheContext:
    """Context manager for cache operations with automatic cleanup"""
    
    def __init__(self, key_prefix: str = ""):
        self.key_prefix = key_prefix
        self.keys_used = set()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Cleanup on error
        if exc_type is not None:
            logger.warning(f"Cache context exiting with error: {exc_type}")
            # Optionally clean up keys that were set during this context
            for key in self.keys_used:
                try:
                    await cache_manager.delete(f"{self.key_prefix}:{key}")
                except Exception as e:
                    logger.error(f"Failed to cleanup cache key {key}: {e}")
    
    async def get(self, key: str, cache: Cache = None) -> Optional[Any]:
        """Get with context tracking"""
        full_key = f"{self.key_prefix}:{key}" if self.key_prefix else key
        return await cache_manager.get(full_key, cache)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None, cache: Cache = None) -> bool:
        """Set with context tracking"""
        full_key = f"{self.key_prefix}:{key}" if self.key_prefix else key
        self.keys_used.add(key)
        return await cache_manager.set(full_key, value, ttl, cache)

# Decorator for caching function results
def cached_result(key_func=None, ttl=None, cache=None):
    """Decorator for caching function results"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_parts = [func.__name__]
                key_parts.extend(str(arg) for arg in args)
                key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
                cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = await cache_manager.get(cache_key, cache)
            if result is not None:
                return result
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache result
            if result is not None:
                await cache_manager.set(cache_key, result, ttl, cache)
            
            return result
        return wrapper
    return decorator
