import streamlit as st
from fastai.vision.all import *
import pandas as pd
from src.utils import MODELS_DIR

LEARNER_PATH = MODELS_DIR / "resnet34_export_20251023-222020.pkl"
learn = load_learner(LEARNER_PATH)

st.title("🏀🏐⚾ Sports Balls Classifier")
file = st.file_uploader("Upload!", type=["jpg","jpeg","png"])

if file:
    img = PILImage.create(file)
    st.image(img.to_thumb(512,512))
    pred, pred_idx, probs = learn.predict(img)

    df_probs = pd.DataFrame({
        "Class": learn.dls.vocab,
        "Probability": [100 * float(p) for p in probs]
    }).sort_values(by="Probability", ascending=False)

    st.subheader(f"Predict: **{pred}**")

    st.dataframe(
        df_probs,
        column_config={
            "Probability": st.column_config.ProgressColumn(
                "Probability",
                help="Model confidence",
                format="%.2f %%",  
                min_value=0.0,
                max_value=100.0,
            )
        },
        hide_index=True,
        use_container_width=True
    )
