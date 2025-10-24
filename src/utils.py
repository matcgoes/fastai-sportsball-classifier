from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
MODELS_DIR = ROOT / "models"

for d in (DATA_PROCESSED, REPORTS_DIR, MODELS_DIR):
    d.mkdir(parents=True, exist_ok=True)

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_current_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)

def plot_loss_per_epoch(learn):
    train_losses = [v[0] for v in learn.recorder.values]
    valid_losses = [v[1] for v in learn.recorder.values]

    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_losses, "-o", label="train", color="blue", linewidth=2, markersize=5)
    plt.plot(epochs, valid_losses, "-s", label="valid", color="red", linewidth=2, markersize=5)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Train vs Validation Loss per Epoch")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(epochs)
