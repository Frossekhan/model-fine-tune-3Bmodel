# Getting Started - 3 Simple Steps

## 1️⃣ Push to Git
```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin <your-repo-url>
git push -u origin main
```

## 2️⃣ Clone and Run (Any Machine)
```bash
git clone <your-repo-url>
cd model-fine-tune-3Bmodel

# Windows
run.bat

# Linux/Mac
chmod +x run.sh
./run.sh
```

## 3️⃣ Access the App
- Open http://localhost:8000
- API docs at http://localhost:8000/docs

## That's It! 🎉

The app will:
- Auto-install all dependencies
- Download the 3B model (~6GB, first run only)
- Start the server on port 8000
- Load your fine-tuned LoRA adapter

## What You Get

✅ **Qwen2.5-3B-Instruct** - 3B parameter language model  
✅ **LoRA Fine-tuned** - Trained on Amazon reviews  
✅ **Sentiment Analysis** - Built-in Naive Bayes classifier  
✅ **RAG Ready** - ChromaDB for document retrieval  
✅ **Streaming API** - Real-time responses  
✅ **Dockerized** - Runs anywhere with Docker  

## Need Help?

See `README_DOCKER.md` for detailed documentation.