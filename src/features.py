import librosa
import numpy as np

def extract_mfcc(file_path, augment=False):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    
    if augment:
        # Add random noise
        noise = np.random.randn(len(audio)) * 0.005
        audio = audio + noise
        
        # Random pitch shift
        if np.random.rand() > 0.5:
            audio = librosa.effects.pitch_shift(audio, sr=sr, n_steps=np.random.uniform(-2, 2))
        
        # Random time stretch
        if np.random.rand() > 0.5:
            audio = librosa.effects.time_stretch(audio, rate=np.random.uniform(0.8, 1.2))
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    mfcc = np.mean(mfcc.T, axis=0)
    return mfcc

