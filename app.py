import streamlit as st
from fastai.vision.all import *
from src.utils import MODELS_DIR

LEARNER_PATH = MODELS_DIR / "resnet34_export_20251023-222020.pkl"
learn = load_learner(LEARNER_PATH)

st.title("🏀🏐⚾ Sports Balls Classifier")
file = st.file_uploader("Upload!", type=["jpg","jpeg","png"])

if file:
    img = PILImage.create(file)
    st.image(img.to_thumb(512,512))
    pred, pred_idx, probs = learn.predict(img)
    st.subheader(f"Predict: **{pred}**")
    st.write({learn.dls.vocab[i]: float(probs[i]) for i in range(len(probs))})
