import numpy as np
from typing import List, Dict
from sklearn.metrics.pairwise import cosine_similarity

class MusicRecommender:
    def __init__(self):
        self.song_features = {}  # Will store processed song features
        self.emotion_mapping = {
            'happy': {'valence': 0.8, 'arousal': 0.7, 'tempo': 0.7},
            'sad': {'valence': 0.2, 'arousal': 0.3, 'tempo': 0.3},
            'neutral': {'valence': 0.5, 'arousal': 0.5, 'tempo': 0.5},
            'angry': {'valence': 0.3, 'arousal': 0.8, 'tempo': 0.8},
            'surprised': {'valence': 0.6, 'arousal': 0.7, 'tempo': 0.6}
        }
    
    def add_song(self, song_id: str, features: Dict):
        """Add a song and its features to the recommendation system."""
        self.song_features[song_id] = features
    
    def get_recommendations(self, 
                          current_emotion: str, 
                          n_recommendations: int = 5) -> List[str]:
        """Get song recommendations based on current emotion."""
        
        if not self.song_features:
            return []
            
        # Get target features based on emotion
        target = self.emotion_mapping.get(current_emotion, self.emotion_mapping['neutral'])
        
        # Calculate similarity scores
        similarities = {}
        for song_id, features in self.song_features.items():
            score = self._calculate_similarity(target, features)
            similarities[song_id] = score
        
        # Sort by similarity and return top N
        recommended_songs = sorted(similarities.items(), 
                                 key=lambda x: x[1], 
                                 reverse=True)[:n_recommendations]
        
        return [song[0] for song in recommended_songs]
    
    def _calculate_similarity(self, target: Dict, features: Dict) -> float:
        """Calculate similarity between target emotion and song features."""
        target_vector = np.array([target['valence'], target['arousal'], target['tempo']])
        feature_vector = np.array([
            features['valence'],
            features['arousal'],
            features['tempo']
        ])
        
        return float(cosine_similarity(
            target_vector.reshape(1, -1), 
            feature_vector.reshape(1, -1)
        )[0][0]) 