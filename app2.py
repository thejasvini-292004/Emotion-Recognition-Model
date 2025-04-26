import streamlit as st
import numpy as np
import pickle
import librosa

# Set the title
st.title("🎙️ Emotion Recognition from Speech")

# Load your trained emotion recognition model
model_path = r"C:\Users\luffy\emotion_model.pkl"
label_encoder_path = r"C:\Users\luffy\label_encoder.pkl"

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)

except FileNotFoundError:
    st.error("Model or label encoder file not found. Please check the path.")
    st.stop()

except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# Define audio feature extraction function
def extract_audio_features(filename):
    y, sr = librosa.load(filename, duration=3, offset=0.5)
    mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    stft = np.abs(librosa.stft(y))
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=y, sr=sr).T, axis=0)
    f0, _, _ = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
    pitch = np.nanmean(f0) if f0 is not None else 0
    zcr = np.mean(librosa.feature.zero_crossing_rate(y).T, axis=0)
    rms = np.mean(librosa.feature.rms(y=y).T, axis=0)
    cent = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr).T, axis=0)
    bw = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr).T, axis=0)
    return np.hstack([mfcc, chroma, mel, pitch, zcr, rms, cent, bw])

# File uploader for audio
uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format='audio/wav')

    try:
        # Extract features
        features = extract_audio_features(uploaded_file)
        features = features.reshape(1, -1)

        # Predict
        probabilities = model.predict_proba(features)[0]
        predicted_label = np.argmax(probabilities)
        predicted_emotion = label_encoder.inverse_transform([predicted_label])[0]

        # Output
        st.subheader(f"🧠 Predicted Emotion: `{predicted_emotion.upper()}`")
        st.markdown("### Emotion Embeddings (Confidence Scores):")

        # Display emotion probabilities
        for idx, prob in enumerate(probabilities):
            emotion = label_encoder.inverse_transform([idx])[0]
            st.write(f"**{emotion.capitalize()}**: {prob:.4f}")

    except Exception as e:
        st.error(f"Error processing audio: {e}")
