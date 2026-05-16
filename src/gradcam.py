# import tensorflow as tf
# import numpy as np
# import cv2
# from src.config import IMG_SIZE, MODEL_SAVE_PATH

# model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# # def make_gradcam_heatmap(img_array, model, pred_index=None):

# #     # Get EfficientNet base model
# #     base_model = model.get_layer("efficientnetb0")

# #     # Get last conv layer
# #     last_conv_layer = base_model.get_layer("top_conv")

# #     # Create model that outputs:
# #     # 1) last conv layer output
# #     # 2) final predictions
# #     grad_model = tf.keras.models.Model(
# #         inputs=model.inputs,
# #         outputs=[last_conv_layer.output, model.output]
# #     )

# #     with tf.GradientTape() as tape:
# #         conv_outputs, predictions = grad_model(img_array)

# #         if pred_index is None:
# #             pred_index = tf.argmax(predictions[0])

# #         class_channel = predictions[:, pred_index]

# #     grads = tape.gradient(class_channel, conv_outputs)

# #     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

# #     conv_outputs = conv_outputs[0]
# #     heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
# #     heatmap = tf.squeeze(heatmap)

# #     heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)

# #     return heatmap.numpy()


# # def save_gradcam_image(img_path, heatmap, output_path, alpha=0.4):

# #     img = cv2.imread(img_path)
# #     img = cv2.resize(img, (224, 224))

# #     heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
# #     heatmap = np.uint8(255 * heatmap)

# #     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

# #     superimposed_img = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)

# #     cv2.imwrite(output_path, superimposed_img)

# #     return output_path


# # --------------------------------
# # GradCAM
# # --------------------------------

# import tensorflow as tf
# import numpy as np
# import cv2
# from src.config import IMG_SIZE, MODEL_SAVE_PATH

# model = tf.keras.models.load_model(MODEL_SAVE_PATH)


# def make_gradcam_heatmap(image_path, save_path):

#     img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
#     img_array = tf.keras.utils.img_to_array(img)
#     img_array = tf.expand_dims(img_array, axis=0)
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

#     # Get backbone
#     base_model = model.get_layer("efficientnetb0")

#     # Build model that outputs last conv feature maps
#     conv_model = tf.keras.models.Model(
#         inputs=base_model.input,
#         outputs=base_model.get_layer("top_conv").output
#     )

#     with tf.GradientTape() as tape:

#         # Forward pass through augmentation
#         x = model.get_layer("sequential")(img_array, training=False)

#         # Get conv feature maps
#         conv_outputs = conv_model(x)

#         tape.watch(conv_outputs)

#         # Continue forward manually
#         x = model.get_layer("global_average_pooling2d")(conv_outputs)
#         x = model.get_layer("batch_normalization")(x, training=False)
#         x = model.get_layer("dropout")(x, training=False)
#         predictions = model.get_layer("dense")(x)

#         predicted_class = tf.argmax(predictions[0])
#         loss = predictions[:, predicted_class]

#     grads = tape.gradient(loss, conv_outputs)

#     pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

#     conv_outputs = conv_outputs[0]
#     heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

#     heatmap = tf.maximum(heatmap, 0)

#     max_val = tf.reduce_max(heatmap)
#     if max_val != 0:
#         heatmap /= max_val

#     heatmap = heatmap.numpy()

#     # Ensure valid shape
#     if heatmap.ndim != 2:
#         heatmap = np.squeeze(heatmap)

#     heatmap = cv2.resize(heatmap.astype(np.float32), IMG_SIZE)
#     heatmap = np.uint8(255 * heatmap)
#     heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

#     original = cv2.imread(image_path)
#     original = cv2.resize(original, IMG_SIZE)

#     superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

#     cv2.imwrite(save_path, superimposed)

#     return save_path


import tensorflow as tf
import numpy as np
import cv2
from src.config import *

# --------------------------------
# Load Model
# --------------------------------
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Extract parts
augmentation = model.get_layer("sequential")
base_model = model.get_layer("efficientnetb0")
gap = model.get_layer("global_average_pooling2d")
bn = model.get_layer("batch_normalization")
dropout = model.get_layer("dropout")
classifier = model.get_layer("dense")

last_conv_layer = base_model.get_layer("top_conv")


# # --------------------------------
# # GradCAM
# # --------------------------------

def generate_gradcam(image_path, save_path):

    # Load image
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, axis=0)

    # EfficientNet preprocessing
    img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_array = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:

        # Augmentation
        x = augmentation(img_array, training=False)

        # Backbone forward
        conv_outputs = base_model(x, training=False)

        # Watch real conv outputs
        tape.watch(conv_outputs)

        # Classification head (manual forward)
        x = gap(conv_outputs)
        x = bn(x, training=False)
        x = dropout(x, training=False)
        predictions = classifier(x)

        pred_index = tf.argmax(predictions[0])
        loss = predictions[:, pred_index]

    # Compute gradients w.r.t real conv output
    grads = tape.gradient(loss, conv_outputs)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_sum(conv_outputs * pooled_grads, axis=-1)

    heatmap = tf.maximum(heatmap, 0)

    max_val = tf.reduce_max(heatmap)
    if max_val > 0:
        heatmap /= max_val

    heatmap = heatmap.numpy()

    # Resize & overlay
    heatmap = cv2.resize(heatmap.astype(np.float32), IMG_SIZE)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(image_path)
    original = cv2.resize(original, IMG_SIZE)

    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(save_path, superimposed)

    return save_path