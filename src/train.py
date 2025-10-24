from fastai.vision.all import *
from fastai.callback.tracker import SaveModelCallback, EarlyStoppingCallback, CSVLogger
from datetime import datetime as dt
from pathlib import Path
import matplotlib.pyplot as plt
from utils import DATA_PROCESSED, REPORTS_DIR, MODELS_DIR, set_seed, save_current_fig, plot_loss_per_epoch
import warnings
warnings.filterwarnings("ignore")

def make_dls(base_path: Path, use_physical_split=False, img_size=224):
    if use_physical_split:
        dls = ImageDataLoaders.from_folder(
            base_path,
            train="train",
            valid="valid",
            item_tfms=Resize(img_size),
            batch_tfms=aug_transforms(),
            bs=16,
            num_workers=4
        )
    else:
        dls = ImageDataLoaders.from_folder(
            base_path,
            train=".",
            valid_pct=0.2,
            seed=42,
            item_tfms=Resize(img_size),
            batch_tfms=aug_transforms(),
            bs=16,
            num_workers=4
        )
    return dls


def main():
    set_seed(42)

    # ==== Main configs  ====
    use_physical_split = True  # train and valid already split
    freeze_epochs = 1  # head layer epoch
    total_epochs = 50  # total epochs (freeze + unfreeze)
    # base_lr = 1e-4  # base learning rate
    patience = 15  # early stopping
    img_size = 224  # image size

    base_path = DATA_PROCESSED
    dls = make_dls(base_path, use_physical_split, img_size)
    print("Classes:", dls.vocab)
    print("#classes:", dls.c)

    # show batch
    dls.show_batch(max_n=9, nrows=3, figsize=(7, 8))
    save_current_fig(REPORTS_DIR / "show_batch.png")
    plt.close()

    # Learner + initial fit
    learn = cnn_learner(dls, resnet34, metrics=accuracy)

    # ensure checkpoints directed to <PROJECT_ROOT>/models
    # (fastai saves in dls.path / 'models' by default)
    learn.path = MODELS_DIR.parent
    learn.model_dir = MODELS_DIR.name

    # Callbacks
    cbs = [
        SaveModelCallback(monitor="accuracy", fname="best_model"),
        EarlyStoppingCallback(monitor="accuracy", patience=patience),
        CSVLogger(),
    ]

    # ==== Fit ====
    print(f"Initializing fitting ({total_epochs} epochs).")
    learn.fine_tune(total_epochs, freeze_epochs=freeze_epochs, cbs=cbs)

    # Plot per batch steps
    learn.recorder.plot_loss()
    save_current_fig(REPORTS_DIR / "train_val_loss.png")
    plt.close()

    # Plot per epoch
    plot_loss_per_epoch(learn)
    save_current_fig(REPORTS_DIR / "train_val_loss_epochs.png")
    plt.close()

    # ==== Evaluate ====
    interp = ClassificationInterpretation.from_learner(learn)
    interp.plot_confusion_matrix(figsize=(7, 7))
    save_current_fig(REPORTS_DIR / "confusion_matrix.png")
    plt.close()

    # Get top losses
    interp.plot_top_losses(9, nrows=3, figsize=(10, 10))
    save_current_fig(REPORTS_DIR / "top_losses.png")
    plt.close()

    acc = float(learn.validate()[1])
    print(f"Evaluated Accuracy: {acc:.4f}")

    # ==== Save final model ====
    learn.load("best_model")
    ts = dt.now().strftime("%Y%m%d-%H%M%S")
    learn.save(f"resnet34_best_{ts}")  # saving model weights
    learn.export(MODELS_DIR / f"resnet34_export_{ts}.pkl")  # save final model

    print(f"Model exported: {MODELS_DIR / f'resnet34_export_{ts}.pkl'}")


if __name__ == "__main__":
    main()
