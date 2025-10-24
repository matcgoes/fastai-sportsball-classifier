from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

ROOT = Path(__file__).resolve().parents[1]   # sobe 1 níveis
DATA = ROOT / "data"
RAW = DATA / "raw"
OUT_TRAIN = DATA / "processed" / "train"
OUT_VALID = DATA / "processed" / "valid"

def rename_folder_files(dir=RAW):
    for folder in dir.iterdir():
        name_folder = folder.name
        for i, file in enumerate(sorted(folder.iterdir()), start=1):
            if file.is_file() and not file.name.startswith('.'):
                new_extension = file.suffix
                new_name = f"{name_folder}_{i}{new_extension}"
                new_filepath = folder / new_name
                try:
                    file.rename(new_filepath)
                except FileExistsError:
                    print(f'File {file} already exsists.')
                    pass


def main(test_size=0.2, seed=42, move=False):
    classes = [d for d in RAW.iterdir() if d.is_dir()]
    for c in classes:
        imgs = list(c.glob("*"))
        tr, va = train_test_split(imgs, test_size=test_size, random_state=seed)
        (OUT_TRAIN/c.name).mkdir(parents=True, exist_ok=True)
        (OUT_VALID/c.name).mkdir(parents=True, exist_ok=True)
        op = shutil.move if move else shutil.copy
        for x in tr: op(str(x), OUT_TRAIN/c.name/x.name)
        for x in va: op(str(x), OUT_VALID/c.name/x.name)
    print("Done!")

if __name__ == "__main__":
    rename_folder_files()
    main()
