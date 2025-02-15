import os
import hashlib

def verify_downloads():
    # Expected MD5 hashes
    expected_hashes = {
        'models/emotion_model.h5': 'a7b9d8c3e4f5g6h7i8j9k0l1m2n3o4p',
        'data/music/sample_music.zip': 'p4o3n2m1l0k9j8i7h6g5f4e3d2c1b8a'
    }
    
    for file_path, expected_hash in expected_hashes.items():
        if not os.path.exists(file_path):
            print(f"Error: {file_path} not found!")
            return False
            
        with open(file_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()
            
        if file_hash != expected_hash:
            print(f"Warning: {file_path} may be corrupted!")
            return False
    
    return True 