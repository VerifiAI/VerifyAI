__version__ = "1.0.0"

# Module exports
from .config import Config
from .newsapi_client import NewsAPIClient
from .serperapi_client import SerperAPIClient
from .newsdata_client import NewsDataClient

__all__ = [
    "Config",
    "NewsAPIClient", 
    "SerperAPIClient",
    "NewsDataClient"
]