from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load image (you can replace with your uploaded image)
image = Image.open(r"C:\Users\joshi\Pictures\plant disease detection\images.jpg")


# 🔥 IMPORTANT CHANGE HERE
texts = [
    "a photo of a plant leaf",
    "a photo of a diseased plant leaf",
    'a photo of a animal',
    "a photo of a car",
    "a photo of a human",
    "a random object"
]

inputs = processor(text=texts, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)

# Print results
for label, prob in zip(texts, probs[0]):
    print(f"{label}: {prob.item():.4f}")

# Decision logic
if probs[0][0] > 0.6 or probs[0][1] > 0.6:
    print("✅ Valid Plant Leaf Image → Proceed to disease model")
else:
    print("❌ Invalid Image → Reject")