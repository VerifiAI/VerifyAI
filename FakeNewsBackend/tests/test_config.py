import os
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch, mock_open

from fact_check_engine.config import Config, APIConfig, CacheConfig, DatabaseConfig, LoggingConfig


class TestConfig:
    """Test cases for the Config class."""
    
    def setup_method(self):
        """Setup test environment before each test."""
        # Store original environment
        self.original_env = os.environ.copy()
        
        # Set required environment variables
        os.environ.update({
            "NEWS_API_KEY": "test_news_key",
            "SERPER_API_KEY": "test_serper_key",
            "NEWSDATA_API_KEY": "test_newsdata_key"
        })
    
    def teardown_method(self):
        """Cleanup after each test."""
        # Restore original environment
        os.environ.clear()
        os.environ.update(self.original_env)
    
    def test_config_initialization_with_required_keys(self):
        """Test successful config initialization with required keys."""
        config = Config()
        
        assert config.api.news_api_key == "test_news_key"
        assert config.api.serper_api_key == "test_serper_key"
        assert config.api.newsdata_api_key == "test_newsdata_key"
        assert config.debug is False
        assert config.host == "0.0.0.0"
        assert config.port == 5001
    
    def test_config_missing_required_keys(self):
        """Test config initialization fails with missing required keys."""
        del os.environ["NEWS_API_KEY"]
        
        with pytest.raises(ValueError, match="Missing required environment variables"):
            Config()
    
    def test_config_with_custom_values(self):
        """Test config initialization with custom environment values."""
        os.environ.update({
            "DEBUG": "true",
            "HOST": "127.0.0.1",
            "PORT": "8080",
            "WORKERS": "8",
            "REQUEST_TIMEOUT": "60",
            "MAX_RETRIES": "5"
        })
        
        config = Config()
        
        assert config.debug is True
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.workers == 8
        assert config.api.request_timeout == 60
        assert config.api.max_retries == 5
    
    def test_config_boolean_parsing(self):
        """Test boolean environment variable parsing."""
        test_cases = [
            ("true", True),
            ("True", True),
            ("1", True),
            ("yes", True),
            ("on", True),
            ("false", False),
            ("False", False),
            ("0", False),
            ("no", False),
            ("off", False),
            ("", False),
            ("invalid", False)
        ]
        
        for env_value, expected in test_cases:
            os.environ["DEBUG"] = env_value
            config = Config()
            assert config.debug == expected, f"Failed for value: {env_value}"
    
    def test_config_load_env_file(self):
        """Test loading configuration from .env file."""
        env_content = """
NEWS_API_KEY=file_news_key
SERPER_API_KEY=file_serper_key
NEWSDATA_API_KEY=file_newsdata_key
DEBUG=true
PORT=9000
# This is a comment
INVALID_LINE_WITHOUT_EQUALS
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
            f.write(env_content)
            env_file_path = f.name
        
        try:
            # Clear environment variables
            for key in ["NEWS_API_KEY", "SERPER_API_KEY", "NEWSDATA_API_KEY"]:
                if key in os.environ:
                    del os.environ[key]
            
            config = Config(env_file=env_file_path)
            
            assert config.api.news_api_key == "file_news_key"
            assert config.api.serper_api_key == "file_serper_key"
            assert config.api.newsdata_api_key == "file_newsdata_key"
            assert config.debug is True
            assert config.port == 9000
        
        finally:
            os.unlink(env_file_path)
    
    def test_config_env_file_not_found(self):
        """Test config behavior when .env file doesn't exist."""
        # Should not raise an error, just log a warning
        config = Config(env_file="/nonexistent/path/.env")
        
        assert config.api.news_api_key == "test_news_key"
    
    def test_api_config_defaults(self):
        """Test APIConfig default values."""
        config = Config()
        api_config = config.api
        
        assert api_config.news_api_rate_limit == 1000
        assert api_config.serper_api_rate_limit == 2500
        assert api_config.newsdata_api_rate_limit == 200
        assert api_config.request_timeout == 30
        assert api_config.max_retries == 3
        assert api_config.retry_delay == 1.0
    
    def test_cache_config_defaults(self):
        """Test CacheConfig default values."""
        config = Config()
        cache_config = config.cache
        
        assert cache_config.redis_url == "redis://localhost:6379"
        assert cache_config.cache_ttl == 3600
        assert cache_config.max_cache_size == 10000
        assert cache_config.enable_cache is True
    
    def test_database_config_defaults(self):
        """Test DatabaseConfig default values."""
        config = Config()
        db_config = config.database
        
        assert db_config.database_url == "sqlite:///fact_check.db"
        assert db_config.pool_size == 10
        assert db_config.max_overflow == 20
        assert db_config.pool_timeout == 30
    
    def test_logging_config_defaults(self):
        """Test LoggingConfig default values."""
        config = Config()
        log_config = config.logging
        
        assert log_config.level == "INFO"
        assert "%(asctime)s" in log_config.format
        assert log_config.file_path is None
        assert log_config.max_file_size == 10 * 1024 * 1024
        assert log_config.backup_count == 5
    
    def test_config_to_dict(self):
        """Test config serialization to dictionary."""
        config = Config()
        config_dict = config.to_dict()
        
        assert "debug" in config_dict
        assert "host" in config_dict
        assert "port" in config_dict
        assert "cache" in config_dict
        assert "logging" in config_dict
        
        # Ensure sensitive data is not included
        assert "api" not in config_dict or "news_api_key" not in str(config_dict)
    
    @pytest.mark.parametrize("env_var,config_attr,expected", [
        ("REDIS_URL", "cache.redis_url", "redis://test:6379"),
        ("CACHE_TTL", "cache.cache_ttl", 7200),
        ("DATABASE_URL", "database.database_url", "postgresql://test"),
        ("LOG_LEVEL", "logging.level", "DEBUG"),
        ("LOG_FILE_PATH", "logging.file_path", "/tmp/test.log")
    ])
    def test_config_custom_settings(self, env_var, config_attr, expected):
        """Test various configuration settings with custom values."""
        os.environ[env_var] = str(expected)
        config = Config()
        
        # Navigate nested attributes
        value = config
        for attr in config_attr.split('.'):
            value = getattr(value, attr)
        
        assert value == expected
    
    def test_config_integer_parsing(self):
        """Test integer environment variable parsing."""
        os.environ.update({
            "PORT": "8080",
            "WORKERS": "16",
            "CACHE_TTL": "7200",
            "MAX_RETRIES": "10"
        })
        
        config = Config()
        
        assert config.port == 8080
        assert config.workers == 16
        assert config.cache.cache_ttl == 7200
        assert config.api.max_retries == 10
    
    def test_config_float_parsing(self):
        """Test float environment variable parsing."""
        os.environ["RETRY_DELAY"] = "2.5"
        
        config = Config()
        
        assert config.api.retry_delay == 2.5
    
    def test_config_list_parsing(self):
        """Test that configuration handles list-like values properly."""
        # This test ensures the config doesn't break with complex values
        os.environ["COMPLEX_VALUE"] = "value1,value2,value3"
        
        config = Config()
        
        # Should not raise an error
        assert config.api.news_api_key == "test_news_key"


class TestConfigDataClasses:
    """Test cases for configuration data classes."""
    
    def test_api_config_creation(self):
        """Test APIConfig creation and attributes."""
        api_config = APIConfig(
            news_api_key="test_key",
            serper_api_key="test_serper",
            newsdata_api_key="test_newsdata"
        )
        
        assert api_config.news_api_key == "test_key"
        assert api_config.serper_api_key == "test_serper"
        assert api_config.newsdata_api_key == "test_newsdata"
        assert api_config.request_timeout == 30  # Default value
    
    def test_cache_config_creation(self):
        """Test CacheConfig creation and attributes."""
        cache_config = CacheConfig(
            redis_url="redis://custom:6379",
            cache_ttl=1800,
            enable_cache=False
        )
        
        assert cache_config.redis_url == "redis://custom:6379"
        assert cache_config.cache_ttl == 1800
        assert cache_config.enable_cache is False
        assert cache_config.max_cache_size == 10000  # Default value
    
    def test_database_config_creation(self):
        """Test DatabaseConfig creation and attributes."""
        db_config = DatabaseConfig(
            database_url="postgresql://test",
            pool_size=20
        )
        
        assert db_config.database_url == "postgresql://test"
        assert db_config.pool_size == 20
        assert db_config.max_overflow == 20  # Default value
    
    def test_logging_config_creation(self):
        """Test LoggingConfig creation and attributes."""
        log_config = LoggingConfig(
            level="DEBUG",
            file_path="/tmp/test.log"
        )
        
        assert log_config.level == "DEBUG"
        assert log_config.file_path == "/tmp/test.log"
        assert log_config.backup_count == 5  # Default value


@pytest.fixture
def mock_env_file():
    """Fixture to create a mock .env file."""
    content = """
NEWS_API_KEY=mock_news_key
SERPER_API_KEY=mock_serper_key
NEWSDATA_API_KEY=mock_newsdata_key
DEBUG=true
"""
    with patch("builtins.open", mock_open(read_data=content)):
        with patch("pathlib.Path.exists", return_value=True):
            yield


def test_config_with_mock_env_file(mock_env_file):
    """Test config loading with mocked .env file."""
    # Clear environment
    for key in ["NEWS_API_KEY", "SERPER_API_KEY", "NEWSDATA_API_KEY"]:
        if key in os.environ:
            del os.environ[key]
    
    with patch.dict(os.environ, {}, clear=True):
        config = Config()
        
        # Should load from mocked file
        assert config.api.news_api_key == "mock_news_key"
        assert config.api.serper_api_key == "mock_serper_key"
        assert config.api.newsdata_api_key == "mock_newsdata_key"
        assert config.debug is True