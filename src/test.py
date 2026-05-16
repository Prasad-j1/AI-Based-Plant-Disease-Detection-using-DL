import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

TRAIN_DIR = "dataset/train"

datagen = ImageDataGenerator(rescale=1./255)

train_generator = datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

class_indices = train_generator.class_indices
print("Class Indices (Label -> Index):")
print(class_indices)

index_to_class = {v: k for k, v in class_indices.items()}
print("\nReversed Mapping (Index -> Label):")
print(index_to_class)

with open("src/config.py", "a") as f:
    f.write(f"INDEX_TO_CLASS = {index_to_class}\n")

print("\nconfig.py updated successfully.")