from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import faiss
import io
import os
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



@app.get("/")
def read_index():
    return FileResponse("static/index.html")

if not os.path.exists("images"):
    os.makedirs("images") 

app.mount("/images", StaticFiles(directory="images"), name="images")

# Load model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load FAISS index
index = faiss.read_index("index.faiss")
image_paths = np.load("image_paths.npy")

def get_image_embedding(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs.detach().numpy()[0]

@app.post("/search")
async def search_image(file: UploadFile = File(...)):
    image_bytes = await file.read()
    query_vector = get_image_embedding(image_bytes).reshape(1, -1)

    D, I = index.search(query_vector, k=5)
    results = [image_paths[i] for i in I[0]]
    return JSONResponse(content={"results": results})
