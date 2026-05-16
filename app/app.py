
import os
import json
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from src.predict import predict_image
from src.gradcam import generate_gradcam
from src.config import *


# ==============================
# CLIP IMPORTS (NEW)
# ==============================
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import torch

print("All files loaded successfully. Starting the Flask app...")

app = Flask(__name__, static_folder='static')

# ==============================
# CONFIGURATION
# ==============================

UPLOAD_FOLDER = os.path.join("app","static", "uploads")
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==============================
# LOAD CLIP MODEL (NEW)
# ==============================

clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ==============================
# CLIP VALIDATION FUNCTION (NEW)
# ==============================

def validate_image_with_clip(image_path):
    image = Image.open(image_path)

    labels = [
        "a photo of a plant leaf",
        "a diseased plant leaf",
        "a car",
        "a human",
        "an animal",
        "a random object"
    ]

    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)
    outputs = clip_model(**inputs)

    probs = outputs.logits_per_image.softmax(dim=1)

    probs_list = probs[0].tolist()
    max_index = probs_list.index(max(probs_list))
    detected_label = labels[max_index]

    plant_score = max(probs_list[0], probs_list[1])

    return plant_score, detected_label

# ==============================
# LOAD DISEASE INFORMATION JSON
# ==============================

print("Testing path and JSON loading...")

here = os.getcwd()

json_path = os.path.join("Data_information", "disease_info.json")

with open(json_path, 'r', encoding='utf-8') as f:
    disease_info = json.load(f)

print("Successfully loaded JSON data.")

# ==============================
# ROUTES
# ==============================

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")


@app.route("/predict", methods=["GET"])
def predict_page():
    return render_template("predict.html")

@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return redirect(url_for("predict_page"))

    file = request.files["file"]

    if file.filename == "":
        return redirect(url_for("predict_page"))

    filename = secure_filename(file.filename)

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    # ==============================
    # CLIP VALIDATION (NEW 🔥)
    # ==============================

    plant_score, detected_label = validate_image_with_clip(filepath)

    if plant_score < 0.6:
        return render_template(
            "error.html",
            image_path=url_for('static', filename='uploads/' + filename),
            detected_label=detected_label
        )

    # ==============================
    # GRADCAM
    # ==============================

    gradcam_filename = "gradcam_" + filename
    gradcam_path = os.path.join(app.config["UPLOAD_FOLDER"], gradcam_filename)

    generate_gradcam(filepath, gradcam_path)

    # ==============================
    # PREDICTION
    # ==============================

    predicted_class, confidence = predict_image(filepath)

    predicted_class_name = INDEX_TO_CLASS.get(predicted_class, "Unknown")

    confidence_str = f"{confidence * 100:.2f}%"

    # ==============================
    # GET DISEASE INFO FROM JSON
    # ==============================

    info = disease_info.get(predicted_class_name, {})

    # ==============================
    # RENDER TEMPLATE
    # ==============================

    return render_template(
        "result.html",

        image_path=url_for('static', filename='uploads/' + filename),
        gradcam_path=url_for('static', filename='uploads/' + gradcam_filename),

        prediction=predicted_class_name,
        confidence=confidence_str,

        disease_details=info
    )

# =============================
# Dashboard route (NEW)
# =============================

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

# ==============================
# RUN APP
# ==============================

if __name__ == "__main__":
    app.run(debug=False)