import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
import librosa
import numpy as np
import os
import gdown

try:
    from transformers import AutoFeatureExtractor as ASTFeatureExtractor
except ImportError:
    from transformers import ASTFeatureExtractor

from transformers import ASTForAudioClassification

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SR = 16000
DURATION = 8
SAMPLES = SR * DURATION

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

GENRE_EMOJI = {
    "blues": "🎸",
    "classical": "🎻",
    "country": "🤠",
    "disco": "🪩",
    "hiphop": "🎤",
    "jazz": "🎷",
    "metal": "🤘",
    "pop": "🌟",
    "reggae": "🌴",
    "rock": "🎸",
}

MODEL_URL = "https://drive.google.com/uc?id=1FehevePB4ZWQRh9jCqFbMs6p9-O10juM"
MODEL_PATH = "best_model.pth"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

# ─────────────────────────────────────────────
# PAGE CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #c850c0 0%, #e8445a 45%, #ff6b35 100%) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL LOADER
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593"
    )

    model = ASTForAudioClassification.from_pretrained(
        "MIT/ast-finetuned-audioset-10-10-0.4593",
        num_labels=10,
        ignore_mismatched_sizes=True
    )

    # Download model from Google Drive if not exists
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading trained model... please wait"):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    return model, feature_extractor


model, feature_extractor = load_model()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-bottom:24px;">
    <span style="font-size:46px; font-weight:800; color:white;">
        🎵 Music Genre Classifier
    </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILE UPLOAD
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your WAV file here",
    type=["wav"]
)

# ─────────────────────────────────────────────
# MAIN LOGIC
# ─────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner("Analyzing your track..."):
        waveform, sr = librosa.load(uploaded_file, sr=SR, mono=True)
        waveform = torch.tensor(waveform)

        if len(waveform) < SAMPLES:
            waveform = F.pad(waveform, (0, SAMPLES - len(waveform)))
        else:
            waveform = waveform[:SAMPLES]

        waveform = waveform / (waveform.abs().max() + 1e-6)

        inputs = feature_extractor(
            waveform.numpy(),
            sampling_rate=SR,
            return_tensors="pt",
            padding=True
        )

        with torch.no_grad():
            logits = model(inputs["input_values"]).logits
            probs = F.softmax(logits, dim=1)[0].numpy()
            pred = int(probs.argmax())

    genre = GENRES[pred]
    confidence = float(probs[pred]) * 100
    emoji = GENRE_EMOJI[genre]

    st.markdown(f"""
    <div style="
        background: rgba(255,255,255,0.18);
        padding: 30px;
        border-radius: 20px;
        text-align: center;
        color: white;
        margin-top: 20px;
    ">
        <div style="font-size:70px;">{emoji}</div>
        <div style="font-size:42px; font-weight:800;">
            {genre.upper()}
        </div>
        <div style="font-size:18px; margin-top:10px;">
            Confidence: {confidence:.2f}%
        </div>
    </div>
    """, unsafe_allow_html=True)