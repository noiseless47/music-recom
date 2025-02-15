import hashlib
from .logger import Logger

logger = Logger()

def verify_model_file(file_path: str, expected_hash: str) -> bool:
    """Verify downloaded model file integrity"""
    try:
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
        return file_hash == expected_hash
    except Exception as e:
        logger.error(f"Error verifying model file: {str(e)}")
        return False 