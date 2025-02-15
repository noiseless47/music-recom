import pygame
import threading
from utils.logger import Logger
from utils.decorators import handle_errors

class MusicPlayer:
    def __init__(self):
        self.logger = Logger()
        pygame.mixer.init()
        self.current_song = None
        self.is_playing = False
        self.volume = 0.5
        pygame.mixer.music.set_volume(self.volume)
    
    @handle_errors
    def play(self, song_path: str):
        """Play a song"""
        if self.is_playing:
            self.stop()
            
        try:
            pygame.mixer.music.load(song_path)
            pygame.mixer.music.play()
            self.current_song = song_path
            self.is_playing = True
            self.logger.info(f"Playing: {song_path}")
        except Exception as e:
            self.logger.error(f"Error playing {song_path}: {str(e)}")
    
    def stop(self):
        """Stop playing"""
        pygame.mixer.music.stop()
        self.is_playing = False
        self.current_song = None
    
    def pause(self):
        """Pause playback"""
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False
    
    def resume(self):
        """Resume playback"""
        if not self.is_playing and self.current_song:
            pygame.mixer.music.unpause()
            self.is_playing = True
    
    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        pygame.mixer.music.set_volume(self.volume) 