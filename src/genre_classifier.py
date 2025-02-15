import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import gdown
import os
import requests

class GenreClassifier:
    def __init__(self):
        self.model = None
        self.genres = ['classical', 'jazz', 'country', 'pop', 'rock', 'metal', 'hip-hop']
        
    def build_model(self):
        self.model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            Flatten(),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(len(self.genres), activation='softmax')
        ])
        
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def download_pretrained_model(self):
        model_path = 'models/genre_model.h5'
        
        if not os.path.exists(model_path):
            urls = [
                "https://huggingface.co/spaces/asishkumar/music-genre/resolve/main/genre_model.h5",
                "https://github.com/asishkumar/music-models/releases/download/v1.0/genre_model.h5"
            ]
            
            success = False
            for url in urls:
                try:
                    print(f"Trying to download from {url}")
                    response = requests.get(url)
                    if response.status_code == 200:
                        os.makedirs('models', exist_ok=True)
                        with open(model_path, 'wb') as f:
                            f.write(response.content)
                        success = True
                        break
                except Exception as e:
                    print(f"Failed to download from {url}: {str(e)}")
                    continue
            
            if not success:
                print("Could not download the model. Please train a new model or download manually.")
                return
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Genre model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")
    
    def predict_genre(self, audio_path):
        # Load and preprocess audio
        y, sr = librosa.load(audio_path, duration=30)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        mel_spec_db = librosa.util.fix_length(mel_spec_db, size=128, axis=1)
        mel_spec_db = np.expand_dims(mel_spec_db, axis=[0, -1])
        
        # Predict genre
        predictions = self.model.predict(mel_spec_db)[0]
        genre_idx = np.argmax(predictions)
        
        return {
            'genre': self.genres[genre_idx],
            'confidence': float(predictions[genre_idx])
        } 