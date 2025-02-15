import sqlite3
import json
import os

class DatabaseHandler:
    def __init__(self):
        self.db_path = 'data/music.db'
        self.setup_database()
    
    def setup_database(self):
        if not os.path.exists('data'):
            os.makedirs('data')
            
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Create tables
        c.execute('''
            CREATE TABLE IF NOT EXISTS songs
            (id TEXT PRIMARY KEY,
             name TEXT,
             features TEXT,
             genre TEXT,
             emotion_ratings TEXT)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_song(self, song_id, name, features, genre, emotion_ratings=None):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT OR REPLACE INTO songs
            (id, name, features, genre, emotion_ratings)
            VALUES (?, ?, ?, ?, ?)
        ''', (song_id, name, json.dumps(features), genre,
              json.dumps(emotion_ratings or {})))
        
        conn.commit()
        conn.close()
    
    def get_song_features(self, song_id):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT features FROM songs WHERE id = ?', (song_id,))
        result = c.fetchone()
        
        conn.close()
        return json.loads(result[0]) if result else None 