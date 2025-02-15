import streamlit as st
import cv2
import time
from emotion_detector import EmotionDetector
from audio_processor import AudioProcessor
from recommender import MusicRecommender
from genre_classifier import GenreClassifier
from utils.database import DatabaseHandler
from utils.visualizer import EmotionVisualizer
import numpy as np
from playlist_generator import PlaylistGenerator
from music_player import MusicPlayer
from utils.logger import Logger
from utils.decorators import handle_errors

def main():
    st.title("Emotion-Based Music Recommender")
    
    # Initialize components
    emotion_detector = EmotionDetector()
    audio_processor = AudioProcessor()
    recommender = MusicRecommender()
    genre_classifier = GenreClassifier()
    genre_classifier.download_pretrained_model()
    db = DatabaseHandler()
    visualizer = EmotionVisualizer()
    playlist_generator = PlaylistGenerator()
    music_player = MusicPlayer()
    logger = Logger()
    
    # Add emotion history tracking
    emotion_history = []
    
    # Sidebar for uploading music
    with st.sidebar:
        st.header("Upload Music")
        uploaded_file = st.file_uploader("Choose a music file", type=['mp3', 'wav'])
        if uploaded_file:
            # Process and add to recommender
            features = audio_processor.extract_features(uploaded_file)
            if features:
                recommender.add_song(uploaded_file.name, features)
                st.success(f"Added {uploaded_file.name} to library")
        
        st.header("Playlist Settings")
        duration = st.slider("Playlist Duration (minutes)", 15, 120, 30)
        smooth_transitions = st.checkbox("Smooth Transitions", value=True)
        
        if st.button("Generate Playlist"):
            playlist = playlist_generator.generate_playlist(
                recommendations,
                duration_mins=duration,
                smooth_transitions=smooth_transitions
            )
            if playlist:
                st.success(f"Generated playlist with {len(playlist)} songs")
                
                # Add music controls
                st.header("Music Controls")
                if st.button("Play"):
                    music_player.play(f"data/music/{playlist[0]['name']}")
                if st.button("Pause"):
                    music_player.pause()
                if st.button("Resume"):
                    music_player.resume()
                if st.button("Stop"):
                    music_player.stop()
                    
                volume = st.slider("Volume", 0.0, 1.0, 0.5)
                music_player.set_volume(volume)
    
    # Main content
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("Emotion Detection")
        run = st.checkbox('Start Emotion Detection')
        FRAME_WINDOW = st.image([])
        
        if run:
            camera = cv2.VideoCapture(0)
            
            while run:
                _, frame = camera.read()
                emotion_result = emotion_detector.detect_emotion(frame)
                
                if emotion_result:
                    # Draw bounding box and emotion
                    x, y, w, h = emotion_result['bbox']
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, 
                              f"{emotion_result['emotion']} ({emotion_result['confidence']:.2f})",
                              (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 
                              0.9, 
                              (0, 255, 0), 
                              2)
                    
                    # Add emotion history
                    emotion_history.append(emotion_result['emotion'])
                    if len(emotion_history) > 50:  # Keep last 50 emotions
                        emotion_history.pop(0)
                    
                    # Plot emotion history
                    with col2:
                        st.subheader("Emotion History")
                        fig = visualizer.plot_emotion_history(emotion_history)
                        st.pyplot(fig)
                    
                    # Get and display recommendations with genre info
                    recommendations = recommender.get_recommendations(
                        emotion_result['emotion']
                    )
                    
                    with col2:
                        st.subheader("Recommended Songs")
                        for song in recommendations:
                            genre_info = genre_classifier.predict_genre(f"data/music/{song}")
                            st.write(f"🎵 {song} ({genre_info['genre']})")
                            
                            # Add to database
                            features = audio_processor.extract_features(f"data/music/{song}")
                            db.add_song(
                                song_id=song,
                                name=song,
                                features=features,
                                genre=genre_info['genre']
                            )
                
                FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                time.sleep(0.1)
            
            camera.release()

if __name__ == "__main__":
    main() 