from fastai.vision.all import *
from utils import MODELS_DIR

LEARNER_PATH = MODELS_DIR / "resnet34_export_20251023-222020.pkl"

def predict_image(img_path: str):
    learn = load_learner(LEARNER_PATH)  # carrega .pkl exportado
    pred, pred_idx, probs = learn.predict(img_path)
    return str(pred), float(probs[pred_idx])

if __name__ == "__main__":
    import sys
    img = sys.argv[1]
    label, conf = predict_image(img)
    print(f"Predição: {label} ({conf:.3f})")
