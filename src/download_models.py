import requests
import os
from tqdm import tqdm

def download_file(url, filename):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for data in response.iter_content(chunk_size=1024):
            size = f.write(data)
            pbar.update(size)

def download_models():
    """Download pre-trained emotion detection model"""
    # Using a different, more reliable source
    model_url = "https://huggingface.co/spaces/asishkumar/emotion-detection/resolve/main/model.h5"
    backup_url = "https://www.dropbox.com/s/1wqmzro7984hxok/emotion_model.h5?dl=1"
    
    model_path = "models/emotion_model.h5"
    os.makedirs("models", exist_ok=True)
    
    print("Downloading emotion detection model...")
    try:
        response = requests.get(model_url, stream=True)
        if response.status_code == 200:
            total_size = int(response.headers.get('content-length', 0))
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
            print("Download complete!")
        else:
            print("Trying backup URL...")
            response = requests.get(backup_url, stream=True)
            if response.status_code == 200:
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                print("Download complete!")
            else:
                print("Failed to download model. Please check your internet connection.")
    except Exception as e:
        print(f"Error downloading model: {str(e)}")

if __name__ == "__main__":
    download_models() 