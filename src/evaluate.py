import tensorflow as tf
from config import *
from sklearn.metrics import classification_report
import numpy as np

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    shuffle=False
)

model = tf.keras.models.load_model(MODEL_SAVE_PATH)

loss, acc = model.evaluate(test_ds)
print(f"Test Accuracy: {acc:.4f}")

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_pred = model.predict(test_ds)

print(classification_report(
    np.argmax(y_true, axis=1),
    np.argmax(y_pred, axis=1)
))