import numpy as np
from typing import List, Dict
from utils.logger import Logger
from utils.decorators import handle_errors

class PlaylistGenerator:
    def __init__(self):
        self.logger = Logger()
    
    @handle_errors
    def generate_playlist(self, 
                         songs: List[Dict], 
                         duration_mins: int = 30,
                         smooth_transitions: bool = True) -> List[Dict]:
        """Generate a playlist with smooth emotion transitions"""
        
        total_duration = 0
        playlist = []
        current_emotion = None
        
        # Sort songs by emotion similarity
        for song in songs:
            if total_duration >= duration_mins * 60:
                break
                
            if not current_emotion:
                current_emotion = song['emotion']
                playlist.append(song)
                total_duration += song['duration']
                continue
            
            # Calculate emotion transition smoothness
            emotion_diff = self._calculate_emotion_difference(
                current_emotion,
                song['emotion']
            )
            
            if not smooth_transitions or emotion_diff < 0.5:
                playlist.append(song)
                current_emotion = song['emotion']
                total_duration += song['duration']
        
        self.logger.info(f"Generated playlist with {len(playlist)} songs")
        return playlist
    
    def _calculate_emotion_difference(self, emotion1: Dict, emotion2: Dict) -> float:
        """Calculate difference between two emotion states"""
        e1 = np.array([
            emotion1['valence'],
            emotion1['arousal'],
            emotion1['energy']
        ])
        
        e2 = np.array([
            emotion2['valence'],
            emotion2['arousal'],
            emotion2['energy']
        ])
        
        return np.linalg.norm(e1 - e2) 