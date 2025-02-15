import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
import numpy as np
import requests
import os
import gdown
from utils.download_helper import safe_download

class EmotionModelTrainer:
    def __init__(self):
        self.model = None
        self.emotions = ['angry', 'happy', 'neutral', 'sad', 'surprised']
    
    def build_model(self):
        """Build the CNN architecture"""
        self.model = Sequential([
            Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(64, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, kernel_size=(3, 3), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.25),
            Dense(len(self.emotions), activation='softmax')
        ])
        
        self.model.compile(
            optimizer=Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
    def download_pretrained_model(self):
        """Download pre-trained emotion detection model"""
        model_path = 'models/emotion_model.h5'
        
        if not os.path.exists(model_path):
            # Try multiple download sources
            urls = [
                "https://huggingface.co/spaces/asishkumar/emotion-detection/resolve/main/emotion_model.h5",
                "https://github.com/asishkumar/emotion-models/releases/download/v1.0/emotion_model.h5"
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
                print("Could not download the model. Please download manually from:")
                print("https://drive.google.com/file/d/1jQwHYHxc0PKyXkI3Jf9XeNWxgHqbUuJv/view?usp=sharing")
                print(f"And place it in: {model_path}")
                return
        
        try:
            self.model = tf.keras.models.load_model(model_path)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {str(e)}")

    def train_simple_model(self):
        """Train a simple emotion detection model for testing"""
        self.build_model()
        
        # Generate dummy data for testing
        X = np.random.randn(100, 48, 48, 1)
        y = np.random.randint(0, len(self.emotions), 100)
        y = tf.keras.utils.to_categorical(y, len(self.emotions))
        
        # Train for a few epochs
        self.model.fit(X, y, epochs=2, batch_size=32)
        
        # Save the model
        os.makedirs('models', exist_ok=True)
        self.model.save('models/emotion_model.h5')
        print("Simple test model trained and saved!")

if __name__ == "__main__":
    trainer = EmotionModelTrainer()
    try:
        trainer.download_pretrained_model()
    except:
        print("Download failed, training simple test model...")
        trainer.train_simple_model() 