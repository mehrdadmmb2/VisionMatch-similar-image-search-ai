# VisionMatch – Similar Image Search AI 🔍🧠

VisionMatch is an open-source AI-powered tool for finding visually similar images.  
Just upload an image, and VisionMatch will search your dataset to find the closest matches based on visual features.

---

## 🖼️ Demo

![screenshot](images/demo_screenshot.png) <!-- Optional: you can add your own screenshot -->

---

## ✨ Features

- Upload your images manually
- Uses powerful vision transformers (CLIP) for embedding generation
- Lightweight FastAPI backend
- Easy-to-use web frontend
- Works locally, no cloud dependency

---

## 🚀 Getting Started

### 🔧 Requirements

- Python 3.8+
- `pip install -r requirements.txt`

### 📝 Indexing Your Image Dataset

Before running the API, make sure to index your image dataset by running the following script:

```bash
python build_index.py
```

### ▶️ Run the server

```bash
uvicorn server:app --reload
```

### 🌐 Open your browser

- Frontend: `index.html` (open in browser manually)
- API Endpoint: `http://localhost:8000/search`

---

## 🧪 How It Works

1. Upload an image via the web UI (or Postman)
2. The backend converts the image to an embedding using a vision model
3. It compares it to preprocessed image vectors in your dataset
4. It returns the most visually similar images

---

## 📁 Folder Structure

```bash
.
├── server.py                # FastAPI backend
├── index.html              # Frontend (simple uploader)
├── static/images/          # Image dataset
├── embeddings.pkl          # Pre-computed image vectors
├── build_index.py          # Dataset updater
├── requirements.txt        # Python dependencies
└── LICENSE                 # MIT License
```

---

## 🫶 Support This Project

If you like this project and want to support future development:

### ☕ Buy me a coffee (crypto)

- **Tether (USDT, TRC20):** `TRzxqih3wSjPkb8EmrF7TzAvSquGhf1wwo`
- **Bitcoin (BTC):** `bc1qxk0h9rdpgnh7yyc59uxndkrr2ndjglwtv3z72j`

Or simply ⭐ **star this repo** — it helps a lot!

---

## 🧑‍💻 Author

Made with ❤️ by [mehrdadmmb2](https://github.com/mehrdadmmb2)

---

## 🪪 License

This project is licensed under the [MIT License](LICENSE)
