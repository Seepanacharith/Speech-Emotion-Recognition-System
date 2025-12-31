import os
import numpy as np
import tensorflow as tf
from src.features import extract_mfcc

# Emotion labels (update order ONLY if your training used a different order)
EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised"
]

def predict_emotion(file_path):
    # Load model inside function to avoid conflicts
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'emotion_model.h5')
    model = tf.keras.models.load_model(model_path, compile=False)
    
    mfcc = extract_mfcc(file_path)

    # shape: (40,) -> (40, 1)
    mfcc = np.expand_dims(mfcc, axis=-1)

    # shape: (40, 1) -> (1, 40, 1)
    mfcc = np.expand_dims(mfcc, axis=0)

    prediction = model.predict(mfcc, verbose=0)
    probabilities = prediction[0]
    
    # Get top 3 emotions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    top_emotions = [EMOTIONS[i] for i in top_indices]
    top_probabilities = [probabilities[i] for i in top_indices]
    
    return top_emotions, top_probabilities
