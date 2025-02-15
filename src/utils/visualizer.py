import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

class EmotionVisualizer:
    def __init__(self):
        self.emotions = ['angry', 'happy', 'neutral', 'sad', 'surprised']
        self.colors = ['red', 'green', 'gray', 'blue', 'purple']
    
    def plot_emotion_history(self, emotion_history):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, emotion in enumerate(self.emotions):
            counts = [h.count(emotion) for h in emotion_history]
            ax.plot(counts, label=emotion, color=self.colors[i])
        
        ax.set_xlabel('Time')
        ax.set_ylabel('Emotion Intensity')
        ax.legend()
        
        return fig
    
    def plot_music_emotion_match(self, current_emotion, recommended_songs):
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot as radar chart
        angles = np.linspace(0, 2*np.pi, len(self.emotions), endpoint=False)
        
        # Add current emotion
        values = [1 if e == current_emotion else 0 for e in self.emotions]
        ax.plot(angles, values, 'o-', label='Current Emotion')
        
        # Add recommended songs
        for song in recommended_songs[:3]:  # Top 3 recommendations
            values = [song['match_scores'][e] for e in self.emotions]
            ax.plot(angles, values, 'o-', label=song['name'])
        
        ax.set_xticks(angles)
        ax.set_xticklabels(self.emotions)
        ax.legend()
        
        return fig 