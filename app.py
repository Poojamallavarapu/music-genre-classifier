import streamlit as st
import streamlit.components.v1 as components
import torch
import torch.nn.functional as F
import librosa
import numpy as np
try:
    from transformers import AutoFeatureExtractor as ASTFeatureExtractor
except ImportError:
    from transformers import ASTFeatureExtractor
from transformers import ASTForAudioClassification

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
SR       = 16000
DURATION = 8
SAMPLES  = SR * DURATION
GENRES   = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]
GENRE_EMOJI = {
    "blues":     "🎸",
    "classical": "🎻",
    "country":   "🤠",
    "disco":     "🪩",
    "hiphop":    "🎤",
    "jazz":      "🎷",
    "metal":     "🤘",
    "pop":       "🌟",
    "reggae":    "🌴",
    "rock":      "🎸",
}

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

# ─────────────────────────────────────────────
# PAGE-LEVEL CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap');

* { font-family: 'Outfit', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #c850c0 0%, #e8445a 45%, #ff6b35 100%) !important;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 640px !important;
}

/* Upload widget */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.12) !important;
    border: 2px dashed rgba(255,255,255,0.5) !important;
    border-radius: 18px !important;
    padding: 4px !important;
}
[data-testid="stFileUploader"] * { color: white !important; }
[data-testid="stFileUploaderDropzone"] { background: transparent !important; }

/* Browse button → "Upload" */
[data-testid="stFileUploaderDropzoneInstructions"] + div button,
[data-testid="stFileUploader"] button {
    background: white !important;
    color: #e8445a !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 15px !important;
    padding: 8px 22px !important;
    cursor: pointer !important;
    font-family: 'Outfit', sans-serif !important;
}
[data-testid="stFileUploader"] button::before {
    content: "Upload";
}
[data-testid="stFileUploader"] button span {
    display: none !important;
}

/* Spinner */
.stSpinner > div { border-top-color: white !important; }

/* iframe transparent bg */
iframe { background: transparent !important; }
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
    model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
    model.eval()
    return model, feature_extractor

model, feature_extractor = load_model()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-bottom:24px;">
    <span style="font-size:46px; font-weight:800; color:white;
                 letter-spacing:-1.5px; text-shadow:0 4px 20px rgba(0,0,0,0.2);
                 font-family:'Outfit',sans-serif;">
        🎵 Music Genre Classifier
    </span>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# UPLOAD
# ─────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Drop your WAV file here",
    type=["wav"],
    label_visibility="collapsed"
)

# ─────────────────────────────────────────────
# HTML BUILDER  (runs inside st.components → proper iframe render)
# ─────────────────────────────────────────────
def build_html(
    emoji="🎵",
    track_name="No Track Loaded",
    sub_text="UPLOAD A WAV FILE TO BEGIN",
    time_cur="0:00",
    time_tot="0:00",
    prog_pct=0,
    idle=True,
    result_genre=None,
    confidence=None,
    top5=None,
):
    ctrl_opacity = "0.35" if idle else "1"

    # ── Result section ──
    result_html = ""
    if not idle and result_genre:
        bars = ""
        if top5:
            top_val = top5[0][1] * 100
            for g, p in top5:
                bar_w = (p * 100 / max(top_val, 0.001)) * 100
                is_top = g == result_genre
                name_style = "font-weight:800;color:white;" if is_top else "color:rgba(255,255,255,0.7);"
                pct_style  = "font-weight:800;color:white;" if is_top else "color:rgba(255,255,255,0.65);"
                bars += f"""
                <div style="display:flex;align-items:center;gap:12px;margin-bottom:10px;">
                    <div style="font-size:16px;width:80px;text-align:right;
                                flex-shrink:0;{name_style}">{g}</div>
                    <div style="flex:1;height:7px;background:rgba(255,255,255,0.2);
                                border-radius:7px;overflow:hidden;">
                        <div style="width:{bar_w:.1f}%;height:100%;background:white;
                                    border-radius:7px;"></div>
                    </div>
                    <div style="font-size:15px;width:42px;text-align:right;
                                flex-shrink:0;{pct_style}">{p*100:.0f}%</div>
                </div>"""

        result_html = f"""
        <div style="background:rgba(255,255,255,0.18);border:1px solid rgba(255,255,255,0.32);
                    border-radius:24px;padding:30px 36px;text-align:center;margin-top:20px;">
            <div style="color:rgba(255,255,255,0.8);font-size:15px;font-weight:600;
                        letter-spacing:2.5px;text-transform:uppercase;margin-bottom:12px;">
                Predicted Genre
            </div>
            <div style="font-size:64px;margin-bottom:8px;">{emoji}</div>
            <div style="color:white;font-size:52px;font-weight:800;letter-spacing:-2px;
                        text-shadow:0 2px 16px rgba(0,0,0,0.2);line-height:1;margin-bottom:8px;">
                {result_genre.upper()}
            </div>
            <div style="color:rgba(255,255,255,0.72);font-size:18px;font-weight:500;margin-bottom:28px;">
                {confidence:.1f}% confidence
            </div>
            <div style="color:rgba(255,255,255,0.7);font-size:14px;font-weight:600;
                        letter-spacing:2px;text-transform:uppercase;
                        margin-bottom:14px;text-align:left;">
                Top Predictions
            </div>
            {bars}
        </div>"""

    return f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&display=swap" rel="stylesheet">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; font-family:'Outfit',sans-serif; }}
  html, body {{ background: transparent; padding:0; }}

  .card {{
    background: rgba(255,255,255,0.14);
    backdrop-filter: blur(24px);
    -webkit-backdrop-filter: blur(24px);
    border: 1px solid rgba(255,255,255,0.28);
    border-radius: 28px;
    padding: 38px 44px 34px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.2);
  }}

  .album-art {{
    width: 190px;
    height: 190px;
    background: linear-gradient(135deg,rgba(255,255,255,0.28),rgba(255,255,255,0.07));
    border: 1.5px solid rgba(255,255,255,0.35);
    border-radius: 22px;
    margin: 0 auto 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 80px;
    box-shadow: 0 12px 40px rgba(0,0,0,0.18);
  }}

  .track-title {{
    color: white;
    font-size: 30px;
    font-weight: 800;
    text-align: center;
    letter-spacing: -0.5px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.2);
    margin-bottom: 6px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }}

  .track-sub {{
    color: rgba(255,255,255,0.72);
    font-size: 16px;
    font-weight: 500;
    text-align: center;
    margin-bottom: 30px;
    letter-spacing: 0.5px;
  }}

  .time-row {{
    display: flex;
    justify-content: space-between;
    color: rgba(255,255,255,0.82);
    font-size: 15px;
    font-weight: 600;
    margin-bottom: 12px;
  }}

  .progress-track {{
    width: 100%;
    height: 5px;
    background: rgba(255,255,255,0.25);
    border-radius: 5px;
    position: relative;
    overflow: visible;
    margin-bottom: 32px;
  }}

  .progress-fill {{
    height: 100%;
    background: white;
    border-radius: 5px;
    position: relative;
    width: {prog_pct:.1f}%;
  }}

  .progress-thumb {{
    width: 16px;
    height: 16px;
    background: white;
    border-radius: 50%;
    position: absolute;
    right: -8px;
    top: -6px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
  }}

  .controls {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 38px;
    opacity: {ctrl_opacity};
  }}

  .ctrl {{
    color: rgba(255,255,255,0.85);
    font-size: 26px;
    cursor: pointer;
    user-select: none;
    line-height: 1;
  }}

  .ctrl-main {{
    width: 60px;
    height: 60px;
    background: rgba(255,255,255,0.2);
    border: 2px solid rgba(255,255,255,0.45);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    color: white;
  }}
</style>
</head>
<body>

<div class="card">
  <div class="album-art">{emoji}</div>
  <div class="track-title">{track_name}</div>
  <div class="track-sub">{sub_text}</div>

  <div class="time-row">
    <span>{time_cur}</span>
    <span>{time_tot}</span>
  </div>

  <div class="progress-track">
    <div class="progress-fill">
      <div class="progress-thumb"></div>
    </div>
  </div>

  <div class="controls">
    <div class="ctrl">&#8249;</div>
    <div class="ctrl">&#9654;</div>
    <div class="ctrl-main">&#9646;&#9646;</div>
    <div class="ctrl">&#9632;</div>
    <div class="ctrl">&#8250;</div>
  </div>
</div>

{result_html}

<div style="text-align:center;margin-top:22px;color:rgba(255,255,255,0.4);
            font-size:14px;letter-spacing:1px;">
  Messy Mashup &nbsp;·&nbsp; T12026
</div>

</body>
</html>"""


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if uploaded_file is not None:
    with st.spinner("Analyzing your track..."):
        waveform, sr = librosa.load(uploaded_file, sr=SR, mono=True)
        waveform     = torch.tensor(waveform)
        duration_sec = len(waveform) / SR

        if len(waveform) < SAMPLES:
            waveform = F.pad(waveform, (0, SAMPLES - len(waveform)))
        else:
            waveform = waveform[:SAMPLES]

        waveform = waveform / (waveform.abs().max() + 1e-6)

        inputs = feature_extractor(
            waveform.numpy(), sampling_rate=SR,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = model(inputs["input_values"]).logits
            probs  = F.softmax(logits, dim=1)[0].numpy()
            pred   = int(probs.argmax())

    genre      = GENRES[pred]
    emoji      = GENRE_EMOJI[genre]
    confidence = float(probs[pred]) * 100
    total_m    = int(duration_sec) // 60
    total_s    = int(duration_sec) % 60
    prog_pct   = min(30 / max(duration_sec, 1) * 100, 100)
    track_name = uploaded_file.name.replace(".wav", "").replace("_", " ").title()
    top5_idx   = probs.argsort()[::-1][:5]
    top5       = [(GENRES[i], float(probs[i])) for i in top5_idx]

    html = build_html(
        emoji        = emoji,
        track_name   = track_name,
        sub_text     = f"DETECTED GENRE · {genre.upper()}",
        time_cur     = "0:30",
        time_tot     = f"{total_m}:{total_s:02d}",
        prog_pct     = prog_pct,
        idle         = False,
        result_genre = genre,
        confidence   = confidence,
        top5         = top5,
    )
    components.html(html, height=860, scrolling=False)

else:
    html = build_html(idle=True)
    components.html(html, height=430, scrolling=False)