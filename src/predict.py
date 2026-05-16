# src/predict.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # adds src folder
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))  # adds root folder

from config import *  # now works even if run directly
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model(MODEL_SAVE_PATH)

def predict_image(image_path):
    img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions)
    confidence = np.max(predictions)

    return predicted_class, confidence
    # print(f"Predicted Class: {INDEX_TO_CLASS[predicted_class]}, Confidence: {confidence:.2f}",predicted_class)

# predict_image(r"dataset\test\Blueberry___healthy\0a0b8f78-df2d-4cfc-becf-cde10fa2766b___RS_HL 5487.JPG")  # Test the function with a sample image

print("Prediction module loaded successfully.")


# from config import *
# import tensorflow as tf
# import numpy as np
# import os
# from src.gradcam import make_gradcam_heatmap, save_gradcam_image
# from src.config import *

# model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# # IMPORTANT: For EfficientNetB0
# # LAST_CONV_LAYER = "top_conv"


# # def predict_image(image_path):

# #     img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
# #     img_array = tf.keras.utils.img_to_array(img)
# #     img_array = tf.expand_dims(img_array, 0)

# #     # 
# #     # img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)
# #     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

# #     predictions = model.predict(img_array)
# #     predicted_class = np.argmax(predictions)
# #     confidence = np.max(predictions)

# #     # ==========================
# #     # Grad-CAM
# #     # ==========================
# #     # heatmap = make_gradcam_heatmap(
# #     #     img_array,
# #     #     model,
# #     #     LAST_CONV_LAYER,
# #     #     predicted_class
# #     # )
# #     heatmap = make_gradcam_heatmap(img_array, model, predicted_class)

# #     gradcam_output_path = os.path.join(
# #         "app", "static", "uploads", "gradcam_" + os.path.basename(image_path)
# #     )

# #     save_gradcam_image(image_path, heatmap, gradcam_output_path)

# #     return predicted_class, confidence, "uploads/" + os.path.basename(gradcam_output_path)



# from src.config import *
# from src.gradcam import make_gradcam_heatmap
# import tensorflow as tf
# import numpy as np
# import os

# model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# def predict_image(image_path):

#     img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
#     img_array = tf.keras.utils.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

#     predictions = model.predict(img_array)

#     predicted_class = np.argmax(predictions)
#     confidence = np.max(predictions)

#     # ----------------------------
#     # Grad-CAM Save Path
#     # ----------------------------
#     filename = os.path.basename(image_path)
#     gradcam_filename = "gradcam_" + filename

#     gradcam_path = os.path.join(
#         os.path.dirname(image_path),
#         gradcam_filename
#     )

#     # Generate Grad-CAM image
#     make_gradcam_heatmap(image_path, gradcam_path)

#     return predicted_class, confidence, "uploads/" + gradcam_filename




# import tensorflow as tf
# import numpy as np
# from src.config import *

# model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# def predict_image(image_path):
#     img = tf.keras.utils.load_img(image_path, target_size=IMG_SIZE)
#     img_array = tf.keras.utils.img_to_array(img)
#     img_array = tf.expand_dims(img_array, 0)
#     img_array = tf.keras.applications.efficientnet.preprocess_input(img_array)

#     predictions = model.predict(img_array)
#     predicted_class = np.argmax(predictions)
#     confidence = np.max(predictions)

#     return predicted_class, confidence