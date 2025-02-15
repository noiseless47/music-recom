import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

class EmotionDetector:
    def __init__(self):
        self.face_detection = mp.solutions.face_detection.FaceDetection(
            min_detection_confidence=0.5
        )
        self.emotions = ['angry', 'happy', 'neutral', 'sad', 'surprised']
        self.model = None
        self.load_model()
        
        # Emotion smoothing
        self.emotion_history = []
        self.history_length = 10
    
    def load_model(self):
        model_path = 'models/emotion_model.h5'
        if not os.path.exists(model_path):
            raise Exception("Model not found! Please run download_models.py first")
        
        self.model = tf.keras.models.load_model(model_path)
        print("Loaded pre-trained emotion model")
    
    def _smooth_predictions(self, current_emotion):
        """Smooth predictions using a rolling window"""
        self.emotion_history.append(current_emotion)
        if len(self.emotion_history) > self.history_length:
            self.emotion_history.pop(0)
        
        # Return most common emotion in history
        from collections import Counter
        return Counter(self.emotion_history).most_common(1)[0][0]
    
    def detect_emotion(self, frame):
        if self.model is None:
            self.load_model()
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(rgb_frame)
            
            if not results.detections:
                return None
            
            face = results.detections[0]
            bbox = face.location_data.relative_bounding_box
            
            h, w, _ = frame.shape
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            # Extract and preprocess face
            face_roi = frame[y:y+height, x:x+width]
            if face_roi.size == 0:
                return None
            
            # Preprocessing
            face_roi = cv2.resize(face_roi, (48, 48))
            face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            face_roi = cv2.equalizeHist(face_roi)
            face_roi = face_roi.astype('float32') / 255.0
            face_roi = np.expand_dims(face_roi, axis=[0, -1])
            
            # Get prediction
            predictions = self.model.predict(face_roi, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            emotion = self.emotions[emotion_idx]
            
            # Smooth predictions
            smoothed_emotion = self._smooth_predictions(emotion)
            
            return {
                'emotion': smoothed_emotion,
                'confidence': float(predictions[emotion_idx]),
                'bbox': (x, y, width, height)
            }
            
        except Exception as e:
            print(f"Error in emotion detection: {str(e)}")
            return None 