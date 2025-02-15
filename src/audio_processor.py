import librosa
import numpy as np
import soundfile as sf
from typing import Dict, Any

class AudioProcessor:
    def __init__(self, sr=22050, duration=30):
        self.sr = sr
        self.duration = duration
        
    def extract_features(self, audio_path: str) -> Dict[str, Any]:
        """Extract relevant audio features from a music file."""
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            
            # Extract features
            # Mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Chromagram
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            
            # Tempo and beat features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            
            # MFCC
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            
            # Spectral features
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
            
            # Energy
            energy = np.sum(mel_spec, axis=0)
            
            return {
                'mel_spectrogram': mel_spec_db,
                'chroma': np.mean(chroma, axis=1),
                'tempo': tempo,
                'mfcc': np.mean(mfcc, axis=1),
                'spectral_centroids': np.mean(spectral_centroids),
                'spectral_rolloff': np.mean(spectral_rolloff),
                'energy': np.mean(energy)
            }
            
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
            
    def get_emotion_features(self, features: Dict[str, Any]) -> Dict[str, float]:
        """Convert audio features to emotion-relevant characteristics."""
        
        # These mappings should be refined based on music psychology research
        valence = np.mean(features['chroma'])  # Key characteristics
        arousal = np.mean(features['energy'])  # Energy level
        
        return {
            'valence': float(valence),
            'arousal': float(arousal),
            'tempo': float(features['tempo']),
            'energy': float(np.mean(features['energy']))
        }

    def get_genre_features(self, genre_classifier, audio_path):
        """Get genre prediction for the audio file"""
        return genre_classifier.predict_genre(audio_path) 