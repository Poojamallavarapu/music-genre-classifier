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
SR       = 16000
DURATION = 8
SAMPLES  = SR * DURATION

GENRES = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

GENRE_EMOJI = {
    "blues":     "🎸", "classical": "🎻", "country":   "🤠",
    "disco":     "🪩", "hiphop":    "🎤", "jazz":      "🎷",
    "metal":     "🤘", "pop":       "🌟", "reggae":    "🌴",
    "rock":      "🎸",
}

GENRE_COLOR = {
    "blues":     "#4facfe", "classical": "#a78bfa", "country":   "#fbbf24",
    "disco":     "#f472b6", "hiphop":    "#34d399", "jazz":      "#fb923c",
    "metal":     "#94a3b8", "pop":       "#f9a8d4", "reggae":    "#4ade80",
    "rock":      "#f87171",
}

MODEL_URL  = "https://drive.google.com/uc?id=1FehevePB4ZWQRh9jCqFbMs6p9-O10juM"
MODEL_PATH = "best_model.pth"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="SoundID — Music Genre Classifier",
    page_icon="🎵",
    layout="centered"
)

# ─────────────────────────────────────────────
# PAGE CSS — dark luxury, neon accents
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

* { font-family: 'DM Sans', sans-serif !important; box-sizing: border-box; }

.stApp {
    background: linear-gradient(135deg, #c850c0 0%, #e8445a 45%, #ff6b35 100%) !important;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }

.block-container {
    padding-top: 0 !important;
    padding-bottom: 2rem !important;
    max-width: 680px !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1.5px dashed rgba(255,255,255,0.15) !important;
    border-radius: 20px !important;
    transition: border-color 0.3s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(255,255,255,0.35) !important;
}
[data-testid="stFileUploader"] * { color: rgba(255,255,255,0.7) !important; }

/* Upload button */
[data-testid="stFileUploader"] button {
    background: white !important;
    color: #080808 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 14px !important;
    padding: 8px 20px !important;
    font-family: 'DM Sans', sans-serif !important;
}
[data-testid="stFileUploader"] button::before { content: "Upload"; }
[data-testid="stFileUploader"] button span { display: none !important; }

/* Spinner */
.stSpinner > div { border-top-color: #fff !important; }

iframe { background: transparent !important; border: none !important; }
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
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights..."):
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, feature_extractor

model, feature_extractor = load_model()

# ─────────────────────────────────────────────
# HTML BUILDER
# ─────────────────────────────────────────────
def build_html(idle=True, emoji="🎵", track_name="", genre=None,
               confidence=None, top5=None, color="#ffffff",
               time_cur="0:00", time_tot="0:00", prog_pct=0):

    # ── waveform bars (random heights for visual)
    np.random.seed(42)
    bars_html = ""
    for i in range(48):
        h     = np.random.randint(20, 100) if not idle else np.random.randint(10, 40)
        delay = round(i * 0.04, 2)
        anim  = f"animation: wave {round(np.random.uniform(0.8,1.6),2)}s {delay}s ease-in-out infinite alternate;" if not idle else ""
        bars_html += f'<div class="bar" style="height:{h}px;{anim}"></div>'

    # ── top 5 bars
    result_section = ""
    if not idle and genre and top5:
        top_val = top5[0][1] * 100
        bars5   = ""
        for g, p in top5:
            pct   = p * 100
            w     = (pct / max(top_val, 0.001)) * 100
            gc    = GENRE_COLOR.get(g, "#ffffff")
            bold  = "font-weight:700;color:#fff;" if g == genre else "color:rgba(255,255,255,0.5);"
            bars5 += f"""
            <div style="display:flex;align-items:center;gap:14px;margin-bottom:12px;">
                <div style="font-size:15px;width:72px;text-align:right;
                            flex-shrink:0;letter-spacing:0.3px;{bold}">{g}</div>
                <div style="flex:1;height:6px;background:rgba(255,255,255,0.08);
                            border-radius:6px;overflow:hidden;">
                    <div style="width:{w:.1f}%;height:100%;border-radius:6px;
                                background:{gc};transition:width 1s ease;"></div>
                </div>
                <div style="font-size:14px;width:38px;text-align:right;
                            flex-shrink:0;{bold}">{pct:.0f}%</div>
            </div>"""

        result_section = f"""
        <!-- RESULT CARD -->
        <div class="result-card" style="--accent:{color};">
            <div class="vinyl-wrap">
                <div class="vinyl">
                    <div class="vinyl-label">{emoji}</div>
                    <div class="vinyl-groove g1"></div>
                    <div class="vinyl-groove g2"></div>
                    <div class="vinyl-groove g3"></div>
                    <div class="vinyl-center"></div>
                </div>
                <div class="vinyl-shadow"></div>
            </div>

            <div class="genre-info">
                <div class="genre-tag" style="color:{color};border-color:{color}30;background:{color}12;">
                    DETECTED
                </div>
                <div class="genre-name" style="color:{color};">{genre.upper()}</div>
                <div class="conf-text">{confidence:.1f}% confidence</div>

                <div class="divider"></div>

                <div class="top-label">TOP PREDICTIONS</div>
                {bars5}
            </div>
        </div>"""

    # ── idle vinyl
    idle_vinyl = ""
    if idle:
        idle_vinyl = """
        <div style="display:flex;justify-content:center;margin:32px 0 24px;">
            <div class="vinyl idle-vinyl">
                <div class="vinyl-label" style="font-size:40px;">🎵</div>
                <div class="vinyl-groove g1"></div>
                <div class="vinyl-groove g2"></div>
                <div class="vinyl-groove g3"></div>
                <div class="vinyl-center"></div>
            </div>
        </div>"""

    ctrl_op = "1" if not idle else "0.3"

    return f"""<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after {{ margin:0; padding:0; box-sizing:border-box; }}
  html, body {{ background:transparent; font-family:'DM Sans',sans-serif; color:white; }}

  /* ── PLAYER CARD ── */
  .player {{
    background: rgba(0,0,0,0.45);
    border: 1px solid rgba(255,255,255,0.22);
    border-radius: 28px;
    padding: 32px 36px 28px;
    box-shadow: 0 32px 80px rgba(0,0,0,0.6);
    position: relative;
    overflow: hidden;
  }}

  .player::before {{
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, rgba(255,255,255,0.03) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
  }}

  /* ── TRACK INFO ── */
  .track-name {{
    font-family: 'Syne', sans-serif;
    font-size: 26px;
    font-weight: 800;
    color: white;
    letter-spacing: -0.5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    margin-bottom: 4px;
  }}

  .track-meta {{
    font-size: 13px;
    color: rgba(255,255,255,0.38);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-bottom: 28px;
    font-weight: 500;
  }}

  /* ── WAVEFORM ── */
  .waveform {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 3px;
    height: 80px;
    margin-bottom: 22px;
    overflow: hidden;
  }}

  .bar {{
    width: 3px;
    border-radius: 3px;
    background: rgba(255,255,255,0.18);
    flex-shrink: 0;
    transition: height 0.4s ease;
  }}

  {''.join(['.bar { background: rgba(255,255,255,0.55); }' if not idle else ''])}

  @keyframes wave {{
    from {{ transform: scaleY(0.4); opacity: 0.5; }}
    to   {{ transform: scaleY(1);   opacity: 1;   }}
  }}

  /* ── PROGRESS ── */
  .time-row {{
    display: flex;
    justify-content: space-between;
    font-size: 13px;
    color: rgba(255,255,255,0.45);
    font-weight: 500;
    margin-bottom: 10px;
    letter-spacing: 0.5px;
  }}

  .prog-track {{
    width: 100%;
    height: 3px;
    background: rgba(255,255,255,0.1);
    border-radius: 3px;
    overflow: visible;
    position: relative;
    margin-bottom: 30px;
    cursor: pointer;
  }}

  .prog-fill {{
    height: 100%;
    background: white;
    border-radius: 3px;
    width: {prog_pct:.1f}%;
    position: relative;
  }}

  .prog-thumb {{
    width: 13px; height: 13px;
    background: white;
    border-radius: 50%;
    position: absolute;
    right: -6px; top: -5px;
    box-shadow: 0 0 0 3px rgba(255,255,255,0.15);
  }}

  /* ── CONTROLS ── */
  .controls {{
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 32px;
    opacity: {ctrl_op};
  }}

  .ctrl {{ font-size: 22px; color: rgba(255,255,255,0.6); cursor:pointer; user-select:none; line-height:1; }}
  .ctrl:hover {{ color: white; }}

  .ctrl-play {{
    width: 58px; height: 58px;
    background: white;
    border-radius: 50%;
    display: flex; align-items:center; justify-content:center;
    font-size: 22px; color: #111;
    box-shadow: 0 8px 32px rgba(255,255,255,0.15);
    cursor: pointer;
  }}

  /* ── RESULT CARD ── */
  .result-card {{
    display: flex;
    gap: 32px;
    align-items: flex-start;
    background: rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 28px;
    padding: 32px 36px;
    margin-top: 18px;
    box-shadow: 0 24px 60px rgba(0,0,0,0.5);
    position: relative;
    overflow: hidden;
  }}

  .result-card::after {{
    content: '';
    position: absolute;
    bottom: -60px; left: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, var(--accent, #fff)18 0%, transparent 70%);
    opacity: 0.07;
    border-radius: 50%;
    pointer-events: none;
  }}

  /* ── VINYL RECORD ── */
  .vinyl-wrap {{
    flex-shrink: 0;
    position: relative;
    width: 160px; height: 160px;
  }}

  .vinyl {{
    width: 160px; height: 160px;
    background: radial-gradient(circle at 50% 50%,
      #2a2a2a 0%, #1a1a1a 30%, #111 60%, #0a0a0a 100%);
    border-radius: 50%;
    display: flex; align-items:center; justify-content:center;
    position: relative;
    box-shadow: 0 12px 40px rgba(0,0,0,0.6),
                inset 0 1px 0 rgba(255,255,255,0.06);
    animation: spin 4s linear infinite;
  }}

  .idle-vinyl {{ animation: none !important; opacity: 0.5; }}

  @keyframes spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(360deg); }} }}

  .vinyl-label {{
    font-size: 36px;
    position: relative; z-index: 10;
    filter: drop-shadow(0 2px 6px rgba(0,0,0,0.5));
    animation: counter-spin 4s linear infinite;
  }}

  .idle-vinyl .vinyl-label {{ animation: none !important; }}

  @keyframes counter-spin {{ from {{ transform: rotate(0deg); }} to {{ transform: rotate(-360deg); }} }}

  .vinyl-groove {{
    position: absolute;
    border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.04);
    top: 50%; left: 50%;
    transform: translate(-50%,-50%);
  }}
  .g1 {{ width: 130px; height: 130px; }}
  .g2 {{ width: 100px; height: 100px; }}
  .g3 {{ width:  70px; height:  70px; }}

  .vinyl-center {{
    position: absolute;
    width: 14px; height: 14px;
    background: #333;
    border-radius: 50%;
    border: 2px solid rgba(255,255,255,0.15);
    top: 50%; left: 50%;
    transform: translate(-50%,-50%);
    z-index: 11;
  }}

  .vinyl-shadow {{
    position: absolute;
    bottom: -12px; left: 50%;
    transform: translateX(-50%);
    width: 130px; height: 20px;
    background: rgba(0,0,0,0.5);
    border-radius: 50%;
    filter: blur(10px);
  }}

  /* ── GENRE INFO ── */
  .genre-info {{ flex: 1; min-width: 0; }}

  .genre-tag {{
    display: inline-block;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid;
    margin-bottom: 10px;
  }}

  .genre-name {{
    font-family: 'Syne', sans-serif;
    font-size: 44px;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1;
    margin-bottom: 6px;
  }}

  .conf-text {{
    font-size: 14px;
    color: rgba(255,255,255,0.45);
    font-weight: 400;
    margin-bottom: 20px;
  }}

  .divider {{
    width: 100%; height: 1px;
    background: rgba(255,255,255,0.06);
    margin-bottom: 18px;
  }}

  .top-label {{
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 2px;
    color: rgba(255,255,255,0.3);
    text-transform: uppercase;
    margin-bottom: 14px;
  }}

  /* ── IDLE VINYL ── */
  .idle-vinyl {{
    width: 140px; height: 140px;
  }}

  /* ── FOOTER ── */
  .footer {{
    text-align: center;
    margin-top: 24px;
    font-size: 12px;
    color: rgba(255,255,255,0.2);
    letter-spacing: 1.5px;
    text-transform: uppercase;
    font-weight: 500;
  }}
</style>
</head>
<body>

<!-- PLAYER CARD -->
<div class="player">

  <div class="track-name">{track_name if track_name else "No Track Loaded"}</div>
  <div class="track-meta">{"Upload a WAV to begin"}</div>

  <!-- Waveform visualizer -->
  <div class="waveform">{bars_html}</div>

  <!-- Progress -->
  <div class="time-row">
    <span>{time_cur}</span>
    <span>{time_tot}</span>
  </div>
  <div class="prog-track">
    <div class="prog-fill">
      <div class="prog-thumb"></div>
    </div>
  </div>

  <!-- Controls -->
  <div class="controls">
    <div class="ctrl">&#9664;&#9664;</div>
    <div class="ctrl">&#9664;</div>
    <div class="ctrl-play">&#9646;&#9646;</div>
    <div class="ctrl">&#9654;</div>
    <div class="ctrl">&#9654;&#9654;</div>
  </div>

</div>

{idle_vinyl}
{result_section}



</body>
</html>"""


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div style="padding: 48px 0 28px; text-align:center;">
    <div style="font-family:'Syne',sans-serif; font-size:52px; font-weight:800;
                color:white; letter-spacing:-2px; line-height:1; margin-bottom:10px;">
        SoundID
    </div>
    <div style="font-size:16px; color:rgba(255,255,255,0.4); font-weight:400;">
        Drop a track. Know its genre.
    </div>
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

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

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
    color      = GENRE_COLOR[genre]
    confidence = float(probs[pred]) * 100
    total_m    = int(duration_sec) // 60
    total_s    = int(duration_sec) % 60
    prog_pct   = min(30 / max(duration_sec, 1) * 100, 100)
    track_name = uploaded_file.name.replace(".wav","").replace("_"," ").title()
    top5_idx   = probs.argsort()[::-1][:5]
    top5       = [(GENRES[i], float(probs[i])) for i in top5_idx]

    html = build_html(
        idle        = False,
        emoji       = emoji,
        track_name  = track_name,
        genre       = genre,
        confidence  = confidence,
        top5        = top5,
        color       = color,
        time_cur    = "0:30",
        time_tot    = f"{total_m}:{total_s:02d}",
        prog_pct    = prog_pct,
    )
    components.html(html, height=900, scrolling=False)

else:
    html = build_html(idle=True)
    components.html(html, height=520, scrolling=False)
