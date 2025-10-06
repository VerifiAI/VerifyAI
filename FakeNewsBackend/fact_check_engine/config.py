import os
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class APIConfig:
    """API configuration settings."""
    news_api_key: str
    serper_api_key: str
    newsdata_api_key: str
    
    # Rate limiting settings
    news_api_rate_limit: int = 1000  # requests per day
    serper_api_rate_limit: int = 2500  # requests per month
    newsdata_api_rate_limit: int = 200  # requests per day
    
    # Timeout settings
    request_timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class CacheConfig:
    """Cache configuration settings."""
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600  # 1 hour default
    max_cache_size: int = 10000
    enable_cache: bool = True


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    database_url: str = "sqlite:///fact_check.db"
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5


class Config:
    """Main configuration class that loads and validates all settings."""
    
    def __init__(self, env_file: Optional[str] = None):
        """
        Initialize configuration from environment variables.
        
        Args:
            env_file: Optional path to .env file to load
        """
        self._load_env_file(env_file)
        self._validate_required_keys()
        
        self.api = self._load_api_config()
        self.cache = self._load_cache_config()
        self.database = self._load_database_config()
        self.logging = self._load_logging_config()
        
        # Application settings
        self.debug = self._get_bool("DEBUG", False)
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "5001"))
        self.workers = int(os.getenv("WORKERS", "4"))
        
        logger.info("Configuration loaded successfully")
    
    def _load_env_file(self, env_file: Optional[str]) -> None:
        """Load environment variables from .env file if it exists."""
        if env_file:
            env_path = Path(env_file)
        else:
            # Look for .env file in current directory and parent directories
            current_dir = Path.cwd()
            for parent in [current_dir] + list(current_dir.parents):
                env_path = parent / ".env"
                if env_path.exists():
                    break
            else:
                logger.warning("No .env file found")
                return
        
        if env_path.exists():
            try:
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            os.environ.setdefault(key.strip(), value.strip())
                logger.info(f"Loaded environment variables from {env_path}")
            except Exception as e:
                logger.error(f"Error loading .env file: {e}")
    
    def _validate_required_keys(self) -> None:
        """Validate that all required environment variables are present."""
        required_keys = [
            "NEWS_API_KEY",
            "SERPER_API_KEY", 
            "NEWSDATA_API_KEY"
        ]
        
        missing_keys = [key for key in required_keys if not os.getenv(key)]
        if missing_keys:
            raise ValueError(f"Missing required environment variables: {missing_keys}")
    
    def _load_api_config(self) -> APIConfig:
        """Load API configuration from environment variables."""
        return APIConfig(
            news_api_key=os.getenv("NEWS_API_KEY", ""),
            serper_api_key=os.getenv("SERPER_API_KEY", ""),
            newsdata_api_key=os.getenv("NEWSDATA_API_KEY", ""),
            news_api_rate_limit=int(os.getenv("NEWS_API_RATE_LIMIT", "1000")),
            serper_api_rate_limit=int(os.getenv("SERPER_API_RATE_LIMIT", "2500")),
            newsdata_api_rate_limit=int(os.getenv("NEWSDATA_API_RATE_LIMIT", "200")),
            request_timeout=int(os.getenv("REQUEST_TIMEOUT", "30")),
            max_retries=int(os.getenv("MAX_RETRIES", "3")),
            retry_delay=float(os.getenv("RETRY_DELAY", "1.0"))
        )
    
    def _load_cache_config(self) -> CacheConfig:
        """Load cache configuration from environment variables."""
        return CacheConfig(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            cache_ttl=int(os.getenv("CACHE_TTL", "3600")),
            max_cache_size=int(os.getenv("MAX_CACHE_SIZE", "10000")),
            enable_cache=self._get_bool("ENABLE_CACHE", True)
        )
    
    def _load_database_config(self) -> DatabaseConfig:
        """Load database configuration from environment variables."""
        return DatabaseConfig(
            database_url=os.getenv("DATABASE_URL", "sqlite:///fact_check.db"),
            pool_size=int(os.getenv("DB_POOL_SIZE", "10")),
            max_overflow=int(os.getenv("DB_MAX_OVERFLOW", "20")),
            pool_timeout=int(os.getenv("DB_POOL_TIMEOUT", "30"))
        )
    
    def _load_logging_config(self) -> LoggingConfig:
        """Load logging configuration from environment variables."""
        return LoggingConfig(
            level=os.getenv("LOG_LEVEL", "INFO"),
            format=os.getenv("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            file_path=os.getenv("LOG_FILE_PATH"),
            max_file_size=int(os.getenv("LOG_MAX_FILE_SIZE", str(10 * 1024 * 1024))),
            backup_count=int(os.getenv("LOG_BACKUP_COUNT", "5"))
        )
    
    def _get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean value from environment variable."""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)."""
        return {
            "debug": self.debug,
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "cache": {
                "redis_url": self.cache.redis_url,
                "cache_ttl": self.cache.cache_ttl,
                "enable_cache": self.cache.enable_cache
            },
            "logging": {
                "level": self.logging.level,
                "file_path": self.logging.file_path
            }
        }


# Global configuration instance
config = Config()

## Suggestions for Upgrade:
# 1. Add configuration validation using Pydantic models for better type safety and validation
# 2. Implement configuration hot-reloading capability for dynamic updates without restart
# 3. Add support for multiple environment profiles (dev, staging, prod) with inheritance
# 4. Integrate with external configuration management systems like Consul or etcd