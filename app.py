import streamlit as st
import tempfile
import matplotlib.pyplot as plt
import librosa
import numpy as np
from src.predict import predict_emotion

st.set_page_config(page_title="Emotion Recognition", layout="centered")

st.title("🎤 Speech Emotion Classification")

# Function to generate the waveform plot
def visualize_waveform(file_path):
    # Load the audio file
    data, sr = librosa.load(file_path)
    
    # Create the plot using matplotlib
    fig, ax = plt.subplots(figsize=(10, 4))
    time = np.linspace(0, len(data)/sr, len(data))
    ax.plot(time, data)
    ax.set_title("Input Audio Waveform", size=15)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    
    return fig

audio_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if audio_file is not None:
    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp:
        audio_file.seek(0)
        temp.write(audio_file.read())
        temp_path = temp.name

    # Display Audio Player
    audio_file.seek(0)
    st.audio(audio_file)

    # --- NEW: Display Waveform ---
    st.subheader("Audio Visualization")
    with st.spinner("Generating waveform..."):
        fig = visualize_waveform(temp_path)
        st.pyplot(fig)
        plt.close(fig)
    # -----------------------------

    if st.button("Detect Emotion"):
        emotions, probabilities = predict_emotion(temp_path)
        st.success(f"Primary Emotion: {emotions[0].title()} (Confidence: {probabilities[0]:.1%})")
        st.info(f"Other possibilities: {emotions[1].title()} ({probabilities[1]:.1%}), {emotions[2].title()} ({probabilities[2]:.1%})")

        # --- Feature Analysis ---
        st.subheader("Feature Analysis")
        st.write("The emotion recognition analyzes the audio's frequency patterns over time. The spectrogram below shows how the sound changes - brighter areas indicate stronger frequencies that help identify emotions like happiness or sadness.")
        with st.spinner("Generating spectrogram..."):
            data, sr = librosa.load(temp_path)
            S = librosa.feature.melspectrogram(y=data, sr=sr, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            fig, ax = plt.subplots(figsize=(10, 4))
            img = ax.imshow(S_dB, aspect='auto', origin='lower', extent=[0, len(data)/sr, 0, sr/2], cmap='viridis')
            ax.set_title('Audio Spectrogram (Frequency Patterns)')
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel('Frequency (Hz)')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            st.pyplot(fig)
            plt.close(fig)
        # -----------------------
