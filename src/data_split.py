import os
import shutil
import random
from tqdm import tqdm

SOURCE_DIR = "E:\\Work\\Plant_disease_detection_DL\\plantvillage dataset\\color"
OUTPUT_DIR = "dataset"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

def split_dataset():
    classes = os.listdir(SOURCE_DIR)

    for cls in classes:
        class_path = os.path.join(SOURCE_DIR, cls)
        images = os.listdir(class_path)
        random.shuffle(images)

        train_split = int(len(images) * TRAIN_RATIO)
        val_split = int(len(images) * (TRAIN_RATIO + VAL_RATIO))

        train_imgs = images[:train_split]
        val_imgs = images[train_split:val_split]
        test_imgs = images[val_split:]

        for split, split_imgs in zip(
            ["train", "val", "test"],
            [train_imgs, val_imgs, test_imgs]
        ):
            split_path = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(split_path, exist_ok=True)

            for img in tqdm(split_imgs, desc=f"{split}-{cls}"):
                src = os.path.join(class_path, img)
                dst = os.path.join(split_path, img)
                shutil.copyfile(src, dst)

if __name__ == "__main__":
    split_dataset()