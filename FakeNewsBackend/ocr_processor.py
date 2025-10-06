import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OCRProcessor:
    """Simple OCR processor placeholder to satisfy import requirements"""
    
    def __init__(self):
        logger.info("OCRProcessor initialized (placeholder implementation)")
    
    def extract_text_from_image(self, image_path: str) -> Tuple[str, Optional[str]]:
        """Extract text from image - placeholder implementation"""
        logger.warning("OCR functionality not available - returning empty text")
        return "", "OCR functionality not implemented"
    
    def process_image_url(self, image_url: str) -> Tuple[str, Optional[str]]:
        """Process image from URL - placeholder implementation"""
        logger.warning("OCR functionality not available - returning empty text")
        return "", "OCR functionality not implemented"