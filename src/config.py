# Audio processing settings
SAMPLE_RATE = 22050
DURATION = 30
N_MELS = 128
HOP_LENGTH = 512

# Emotion detection settings
FACE_DETECTION_CONFIDENCE = 0.5
EMOTION_MODEL_INPUT_SIZE = (48, 48)

# Genre classification settings
GENRE_MODEL_INPUT_SIZE = (128, 128)

# Recommendation settings
N_RECOMMENDATIONS = 5
SIMILARITY_THRESHOLD = 0.5

# Database settings
DATABASE_PATH = 'data/music.db'

# Visualization settings
EMOTION_COLORS = {
    'angry': 'red',
    'happy': 'green',
    'neutral': 'gray',
    'sad': 'blue',
    'surprised': 'purple'
}

# Add these to config.py
MODEL_URLS = {
    'emotion': {
        'primary': "https://drive.google.com/uc?id=1-LWwpXrGXeq5u1WGxkn4KeBr0kXQyqxS",
        'backup': "https://huggingface.co/datasets/asishkumar/emotion-models/resolve/main/emotion_model.h5"
    },
    'genre': {
        'primary': "https://drive.google.com/uc?id=1-NKQpYYaC7RU2hWv6R-m0KEy6VhGNq2M",
        'backup': "https://huggingface.co/datasets/asishkumar/music-models/resolve/main/genre_model.h5"
    }
} 