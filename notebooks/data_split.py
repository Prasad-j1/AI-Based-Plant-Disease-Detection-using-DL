import os
import shutil
import random

SOURCE_DIR = r'E:\Work\Plant_disease_detection_DL\plantvillage dataset\color'
DEST_DIR = r"E:\Work\Plant_disease_detection_DL\dataset"

TRAIN_DIR = os.path.join(DEST_DIR, "train")
VAL_DIR = os.path.join(DEST_DIR, "val")

print("🔄 Starting dataset split...")
print("TRAIN_DIR:", TRAIN_DIR)
print("VAL_DIR:", VAL_DIR)

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(VAL_DIR, exist_ok=True)

SPLIT_RATIO = 0.8  # 80% train, 20% validation

for class_name in os.listdir(SOURCE_DIR):
    class_path = os.path.join(SOURCE_DIR, class_name)
    
    if not os.path.isdir(class_path):
        continue

    images = os.listdir(class_path)
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)

    train_images = images[:split_index]
    val_images = images[split_index:]

    os.makedirs(os.path.join(TRAIN_DIR, class_name), exist_ok=True)
    os.makedirs(os.path.join(VAL_DIR, class_name), exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(TRAIN_DIR, class_name, img)
        )

    for img in val_images:
        shutil.copy(
            os.path.join(class_path, img),
            os.path.join(VAL_DIR, class_name, img)
        )

print("✅ Dataset split completed successfully")