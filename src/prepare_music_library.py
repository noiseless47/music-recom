import os
import requests
import zipfile
from tqdm import tqdm
from audio_processor import AudioProcessor
import json
import gdown  # Add this import at the top

class MusicLibraryPreparer:
    def __init__(self):
        self.audio_processor = AudioProcessor()
        
    def download_sample_music(self):
        """Download sample music dataset"""
        if not os.path.exists('data/music'):
            os.makedirs('data/music')
            
        # Download GTZAN dataset sample (or any other music dataset)
        url = "https://drive.google.com/uc?id=1jR3qKh6nFfkQUVA7UyT6OB_qbKGw0YeL"
        output = "data/music/sample_music.zip"
        
        if not os.path.exists(output):
            print("Downloading sample music...")
            gdown.download(url, output, quiet=False)
            
            # Extract the zip file
            with zipfile.ZipFile(output, 'r') as zip_ref:
                zip_ref.extractall('data/music')
                
    def process_music_library(self):
        """Process all music files and extract features"""
        music_dir = 'data/music'
        features_file = 'data/music_features.json'
        
        if not os.path.exists(features_file):
            features = {}
            
            for filename in tqdm(os.listdir(music_dir)):
                if filename.endswith(('.mp3', '.wav')):
                    filepath = os.path.join(music_dir, filename)
                    audio_features = self.audio_processor.extract_features(filepath)
                    
                    if audio_features:
                        emotion_features = self.audio_processor.get_emotion_features(audio_features)
                        features[filename] = emotion_features
            
            # Save features to file
            with open(features_file, 'w') as f:
                json.dump(features, f)
            
            print(f"Processed {len(features)} music files!")
        
if __name__ == "__main__":
    preparer = MusicLibraryPreparer()
    preparer.download_sample_music()
    preparer.process_music_library() 