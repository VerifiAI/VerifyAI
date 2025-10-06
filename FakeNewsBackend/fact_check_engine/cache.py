import asyncio
import logging
import json
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import time

try:
    import redis.asyncio as redis
    from redis.asyncio import Redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. Caching will be disabled.")

from .config import config
from .errors import SystemError, ErrorCode

logger = logging.getLogger(__name__)


@dataclass
class CacheStats:
    """Cache statistics and metrics."""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    errors: int = 0
    total_size: int = 0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "hits": self.hits,
            "misses": self.misses,
            "sets": self.sets,
            "deletes": self.deletes,
            "errors": self.errors,
            "total_size": self.total_size,
            "hit_rate": self.hit_rate
        }


class CacheKey:
    """Cache key management utilities."""
    
    @staticmethod
    def fact_check_result(claim: str, config_hash: str = "") -> str:
        """Generate cache key for fact-check results."""
        claim_hash = hashlib.md5(claim.encode()).hexdigest()
        return f"fact_check:{claim_hash}:{config_hash}"
    
    @staticmethod
    def search_result(query: str, params_hash: str = "") -> str:
        """Generate cache key for search results."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"search:{query_hash}:{params_hash}"
    
    @staticmethod
    def evidence_collection(claim: str, sources_hash: str = "") -> str:
        """Generate cache key for evidence collection."""
        claim_hash = hashlib.md5(claim.encode()).hexdigest()
        return f"evidence:{claim_hash}:{sources_hash}"
    
    @staticmethod
    def proof_validation(evidence_hash: str) -> str:
        """Generate cache key for proof validation."""
        return f"validation:{evidence_hash}"
    
    @staticmethod
    def consensus_result(validation_hash: str, scoring_hash: str = "") -> str:
        """Generate cache key for consensus results."""
        return f"consensus:{validation_hash}:{scoring_hash}"
    
    @staticmethod
    def api_response(api_name: str, endpoint: str, params_hash: str) -> str:
        """Generate cache key for API responses."""
        return f"api:{api_name}:{endpoint}:{params_hash}"
    
    @staticmethod
    def user_session(user_id: str, session_id: str) -> str:
        """Generate cache key for user sessions."""
        return f"session:{user_id}:{session_id}"
    
    @staticmethod
    def rate_limit(identifier: str, window: str) -> str:
        """Generate cache key for rate limiting."""
        return f"rate_limit:{identifier}:{window}"
    
    @staticmethod
    def hash_params(params: Dict[str, Any]) -> str:
        """Generate hash for parameters."""
        params_str = json.dumps(params, sort_keys=True, default=str)
        return hashlib.md5(params_str.encode()).hexdigest()


class RedisCache:
    """Redis-based caching system for fact-checking operations."""
    
    def __init__(self):
        """Initialize Redis cache."""
        self.redis_client: Optional[Redis] = None
        self.stats = CacheStats()
        self.enabled = config.cache.enable_cache and REDIS_AVAILABLE
        
        # Cache configuration
        self.default_ttl = config.cache.cache_ttl
        self.max_cache_size = config.cache.max_cache_size
        
        # Key prefixes for different data types
        self.key_prefixes = {
            "fact_check": "fc:",
            "search": "sr:",
            "evidence": "ev:",
            "validation": "vl:",
            "consensus": "cs:",
            "api": "api:",
            "session": "sess:",
            "rate_limit": "rl:"
        }
        
        logger.info(f"RedisCache initialized (enabled: {self.enabled})")
    
    async def connect(self) -> bool:
        """Connect to Redis server."""
        if not self.enabled:
            return False
        
        try:
            self.redis_client = redis.from_url(
                config.cache.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_timeout=True,
                health_check_interval=30
            )
            
            # Test connection
            await self.redis_client.ping()
            logger.info("Successfully connected to Redis")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.enabled = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis server."""
        if self.redis_client:
            await self.redis_client.close()
            logger.info("Disconnected from Redis")
    
    async def get(
        self,
        key: str,
        default: Any = None,
        deserialize: bool = True
    ) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            deserialize: Whether to deserialize JSON data
            
        Returns:
            Cached value or default
        """
        if not self.enabled or not self.redis_client:
            return default
        
        try:
            value = await self.redis_client.get(key)
            
            if value is None:
                self.stats.misses += 1
                return default
            
            self.stats.hits += 1
            
            if deserialize:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    # Try pickle for complex objects
                    try:
                        return pickle.loads(value.encode('latin1'))
                    except:
                        return value
            else:
                return value
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            self.stats.errors += 1
            return default
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[int] = None,
        serialize: bool = True
    ) -> bool:
        """
        Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
            serialize: Whether to serialize data as JSON
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            if serialize:
                try:
                    serialized_value = json.dumps(value, default=str)
                except (TypeError, ValueError):
                    # Fallback to pickle for complex objects
                    serialized_value = pickle.dumps(value).decode('latin1')
            else:
                serialized_value = value
            
            ttl = ttl or self.default_ttl
            
            result = await self.redis_client.setex(key, ttl, serialized_value)
            
            if result:
                self.stats.sets += 1
                return True
            else:
                self.stats.errors += 1
                return False
                
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            self.stats.errors += 1
            return False
    
    async def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.delete(key)
            
            if result:
                self.stats.deletes += 1
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            self.stats.errors += 1
            return False
    
    async def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.
        
        Args:
            key: Cache key to check
            
        Returns:
            True if key exists, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.exists(key)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            self.stats.errors += 1
            return False
    
    async def expire(self, key: str, ttl: int) -> bool:
        """
        Set expiration time for key.
        
        Args:
            key: Cache key
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            result = await self.redis_client.expire(key, ttl)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            self.stats.errors += 1
            return False
    
    async def get_ttl(self, key: str) -> int:
        """
        Get time to live for key.
        
        Args:
            key: Cache key
            
        Returns:
            TTL in seconds, -1 if no expiration, -2 if key doesn't exist
        """
        if not self.enabled or not self.redis_client:
            return -2
        
        try:
            return await self.redis_client.ttl(key)
            
        except Exception as e:
            logger.error(f"Cache TTL error for key {key}: {e}")
            self.stats.errors += 1
            return -2
    
    async def increment(self, key: str, amount: int = 1, ttl: Optional[int] = None) -> int:
        """
        Increment counter in cache.
        
        Args:
            key: Cache key
            amount: Amount to increment
            ttl: Time to live for new keys
            
        Returns:
            New value after increment
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            # Use pipeline for atomic operation
            async with self.redis_client.pipeline() as pipe:
                await pipe.incr(key, amount)
                if ttl and not await self.exists(key):
                    await pipe.expire(key, ttl)
                results = await pipe.execute()
                return results[0]
                
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            self.stats.errors += 1
            return 0
    
    async def get_multiple(self, keys: List[str]) -> Dict[str, Any]:
        """
        Get multiple values from cache.
        
        Args:
            keys: List of cache keys
            
        Returns:
            Dictionary of key-value pairs
        """
        if not self.enabled or not self.redis_client or not keys:
            return {}
        
        try:
            values = await self.redis_client.mget(keys)
            result = {}
            
            for key, value in zip(keys, values):
                if value is not None:
                    try:
                        result[key] = json.loads(value)
                        self.stats.hits += 1
                    except json.JSONDecodeError:
                        result[key] = value
                        self.stats.hits += 1
                else:
                    self.stats.misses += 1
            
            return result
            
        except Exception as e:
            logger.error(f"Cache mget error: {e}")
            self.stats.errors += 1
            return {}
    
    async def set_multiple(
        self,
        data: Dict[str, Any],
        ttl: Optional[int] = None
    ) -> bool:
        """
        Set multiple values in cache.
        
        Args:
            data: Dictionary of key-value pairs
            ttl: Time to live in seconds
            
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client or not data:
            return False
        
        try:
            # Serialize all values
            serialized_data = {}
            for key, value in data.items():
                try:
                    serialized_data[key] = json.dumps(value, default=str)
                except (TypeError, ValueError):
                    serialized_data[key] = pickle.dumps(value).decode('latin1')
            
            # Use pipeline for atomic operation
            async with self.redis_client.pipeline() as pipe:
                await pipe.mset(serialized_data)
                
                if ttl:
                    for key in data.keys():
                        await pipe.expire(key, ttl)
                
                await pipe.execute()
                
            self.stats.sets += len(data)
            return True
            
        except Exception as e:
            logger.error(f"Cache mset error: {e}")
            self.stats.errors += 1
            return False
    
    async def delete_pattern(self, pattern: str) -> int:
        """
        Delete keys matching pattern.
        
        Args:
            pattern: Key pattern (supports wildcards)
            
        Returns:
            Number of keys deleted
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            keys = []
            async for key in self.redis_client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                deleted = await self.redis_client.delete(*keys)
                self.stats.deletes += deleted
                return deleted
            
            return 0
            
        except Exception as e:
            logger.error(f"Cache delete pattern error for {pattern}: {e}")
            self.stats.errors += 1
            return 0
    
    async def clear_all(self) -> bool:
        """
        Clear all cache data.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.enabled or not self.redis_client:
            return False
        
        try:
            await self.redis_client.flushdb()
            logger.info("Cache cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            self.stats.errors += 1
            return False
    
    async def get_info(self) -> Dict[str, Any]:
        """
        Get Redis server information.
        
        Returns:
            Redis server info dictionary
        """
        if not self.enabled or not self.redis_client:
            return {}
        
        try:
            info = await self.redis_client.info()
            return info
            
        except Exception as e:
            logger.error(f"Cache info error: {e}")
            return {}
    
    async def get_size(self) -> int:
        """
        Get total number of keys in cache.
        
        Returns:
            Number of keys
        """
        if not self.enabled or not self.redis_client:
            return 0
        
        try:
            return await self.redis_client.dbsize()
            
        except Exception as e:
            logger.error(f"Cache size error: {e}")
            return 0
    
    async def get_memory_usage(self) -> Dict[str, Any]:
        """
        Get memory usage statistics.
        
        Returns:
            Memory usage information
        """
        if not self.enabled or not self.redis_client:
            return {}
        
        try:
            info = await self.redis_client.info("memory")
            return {
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "used_memory_peak": info.get("used_memory_peak", 0),
                "used_memory_peak_human": info.get("used_memory_peak_human", "0B"),
                "memory_fragmentation_ratio": info.get("mem_fragmentation_ratio", 0)
            }
            
        except Exception as e:
            logger.error(f"Cache memory usage error: {e}")
            return {}
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        return self.stats
    
    def reset_stats(self):
        """Reset cache statistics."""
        self.stats = CacheStats()


# Cache decorators
def cached(
    ttl: int = None,
    key_prefix: str = "",
    serialize_args: bool = True
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
        serialize_args: Whether to serialize function arguments for key
    """
    def decorator(func):
        async def wrapper(*args, **kwargs):
            if not cache.enabled:
                return await func(*args, **kwargs)
            
            # Generate cache key
            if serialize_args:
                args_str = json.dumps([str(arg) for arg in args], default=str)
                kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
                key_data = f"{args_str}:{kwargs_str}"
            else:
                key_data = f"{len(args)}:{len(kwargs)}"
            
            key_hash = hashlib.md5(key_data.encode()).hexdigest()
            cache_key = f"{key_prefix}:{func.__name__}:{key_hash}"
            
            # Try to get from cache
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Global cache instance
cache = RedisCache()


# Cache management functions
async def initialize_cache() -> bool:
    """Initialize cache connection."""
    return await cache.connect()


async def cleanup_cache():
    """Cleanup cache connection."""
    await cache.disconnect()


async def warm_cache(data: Dict[str, Any]) -> bool:
    """
    Warm cache with initial data.
    
    Args:
        data: Dictionary of key-value pairs to cache
        
    Returns:
        True if successful, False otherwise
    """
    return await cache.set_multiple(data)


async def get_cache_health() -> Dict[str, Any]:
    """
    Get cache health status.
    
    Returns:
        Health status information
    """
    if not cache.enabled:
        return {
            "status": "disabled",
            "enabled": False,
            "redis_available": REDIS_AVAILABLE
        }
    
    try:
        # Test basic operations
        test_key = "health_check"
        test_value = {"timestamp": datetime.now().isoformat()}
        
        # Test set
        set_success = await cache.set(test_key, test_value, ttl=60)
        
        # Test get
        get_result = await cache.get(test_key)
        get_success = get_result is not None
        
        # Test delete
        delete_success = await cache.delete(test_key)
        
        # Get server info
        server_info = await cache.get_info()
        memory_info = await cache.get_memory_usage()
        
        return {
            "status": "healthy" if all([set_success, get_success, delete_success]) else "unhealthy",
            "enabled": True,
            "operations": {
                "set": set_success,
                "get": get_success,
                "delete": delete_success
            },
            "stats": cache.get_stats().to_dict(),
            "server_info": {
                "version": server_info.get("redis_version", "unknown"),
                "uptime": server_info.get("uptime_in_seconds", 0),
                "connected_clients": server_info.get("connected_clients", 0)
            },
            "memory": memory_info
        }
        
    except Exception as e:
        logger.error(f"Cache health check failed: {e}")
        return {
            "status": "unhealthy",
            "enabled": True,
            "error": str(e)
        }


## Suggestions for Upgrade:
# 1. Implement cache partitioning and sharding for horizontal scaling
# 2. Add cache warming strategies based on usage patterns and machine learning
# 3. Integrate with Redis Cluster for high availability and automatic failover
# 4. Add cache compression and optimization for large objects to reduce memory usage