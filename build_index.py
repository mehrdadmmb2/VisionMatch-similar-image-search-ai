import os
from PIL import Image
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel

# Settings
IMAGE_DIR = "images/"
INDEX_FILE = "index.faiss"
PATHS_FILE = "image_paths.npy"

# Loading model
print("📦 Loading CLIP model...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Preparing images
embeddings = []
image_paths = []

print("🖼️ Starting image processing...")
for filename in os.listdir(IMAGE_DIR):
    filepath = os.path.join(IMAGE_DIR, filename)
    if filename.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):
        try:
            image = Image.open(filepath).convert("RGB")
            inputs = processor(images=image, return_tensors="pt")
            with torch.no_grad():
                outputs = model.get_image_features(**inputs)
            vector = outputs[0].numpy()
            embeddings.append(vector)
            image_paths.append(filepath)
            print(f"✅ Processed: {filename}")
        except Exception as e:
            print(f"❌ Error in {filename}: {e}")

# Convert to array
embedding_matrix = np.array(embeddings).astype("float32")

# Create and save FAISS index
print("📦 Creating FAISS index...")
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # Euclidean distance
index.add(embedding_matrix)
faiss.write_index(index, INDEX_FILE)

# Save image paths
np.save(PATHS_FILE, np.array(image_paths))

print("🎉 Everything is ready! Now run the server and test with Postman.")
