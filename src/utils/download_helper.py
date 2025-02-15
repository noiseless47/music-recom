import gdown
import os
from .logger import Logger

logger = Logger()

def safe_download(url: str, output: str) -> bool:
    """Safely download file with retries and error handling"""
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output), exist_ok=True)
        
        # Try downloading with gdown
        success = gdown.download(url, output, quiet=False)
        
        if not success:
            logger.error(f"Failed to download from {url}")
            return False
            
        if not os.path.exists(output):
            logger.error(f"Download completed but file not found: {output}")
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error downloading file: {str(e)}")
        return False 

def try_multiple_sources(urls: list, output: str) -> bool:
    """Try downloading from multiple sources"""
    for url in urls:
        if safe_download(url, output):
            return True
    return False 