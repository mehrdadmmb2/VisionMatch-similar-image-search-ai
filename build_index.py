import os
from PIL import Image
import numpy as np
import faiss
import torch
from transformers import CLIPProcessor, CLIPModel

# تنظیمات
IMAGE_DIR = "images/"
INDEX_FILE = "index.faiss"
PATHS_FILE = "image_paths.npy"

# بارگذاری مدل
print("📦 بارگذاری مدل CLIP...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# آماده‌سازی تصاویر
embeddings = []
image_paths = []

print("🖼️ شروع پردازش تصاویر...")
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
            print(f"✅ پردازش شد: {filename}")
        except Exception as e:
            print(f"❌ خطا در {filename}: {e}")

# تبدیل به آرایه
embedding_matrix = np.array(embeddings).astype("float32")

# ساخت و ذخیره ایندکس FAISS
print("📦 ساخت FAISS index...")
index = faiss.IndexFlatL2(embedding_matrix.shape[1])  # فاصله اقلیدسی
index.add(embedding_matrix)
faiss.write_index(index, INDEX_FILE)

# ذخیره مسیر عکس‌ها
np.save(PATHS_FILE, np.array(image_paths))

print("🎉 همه‌چیز آماده‌ست! حالا سرور رو اجرا کن و با Postman تست کن.")
