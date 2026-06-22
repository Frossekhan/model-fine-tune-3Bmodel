# Enterprise AI Assistant - Docker Quick Start

A simple, production-ready AI assistant with Qwen2.5-3B-Instruct model and LoRA fine-tuning.

## 🚀 One-Command Setup

### Windows
```bash
run.bat
```

### Linux/Mac
```bash
chmod +x run.sh
./run.sh
```

That's it! The script will:
1. ✅ Check Docker installation
2. ✅ Create necessary directories
3. ✅ Build the Docker image
4. ✅ Start the application
5. ✅ Download the 3B model (first run only)

## 📊 Access the Application

- **API:** http://localhost:8000
- **Web UI:** http://localhost:8000
- **Metrics:** http://localhost:8000/metrics

## 🛠️ Manual Commands

If you prefer to run commands manually:

```bash
# Build the image
docker-compose build

# Start the application
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the application
docker-compose down
```

## 📁 Project Structure

```
model-fine-tune-3Bmodel/
├── app/                      # Application code
│   ├── main.py              # FastAPI server
│   ├── config.py            # Configuration
│   └── services/            # AI services
├── models/                  # Model storage (gitignored)
│   ├── lora-qwen3b-amazon-valid/  # Your fine-tuned LoRA
│   └── sentiment/           # Sentiment classifier
├── training/                # Training scripts
│   └── examples/            # Training datasets
├── Dockerfile               # Docker image definition
├── docker-compose.yml       # Docker Compose config
├── run.sh / run.bat         # One-click run scripts
├── requirements.txt         # Python dependencies
└── .env.example            # Environment template
```

## 🔧 Configuration

1. Copy `.env.example` to `.env`
2. Edit `.env` if you need to customize settings
3. Run `run.bat` or `./run.sh`

## 🎯 Model Details

- **Base Model:** Qwen/Qwen2.5-3B-Instruct (3B parameters)
- **Fine-tuning:** LoRA with r=8, targeting q_proj and v_proj
- **Trainable Parameters:** ~10-15M (only 0.3% of total)
- **Task:** Amazon review sentiment analysis

## 📦 What's Included

- ✅ FastAPI backend with streaming support
- ✅ Qwen2.5-3B-Instruct base model
- ✅ LoRA fine-tuned adapter (Amazon reviews)
- ✅ RAG (Retrieval-Augmented Generation) with ChromaDB
- ✅ Sentiment analysis (Naive Bayes classifier)
- ✅ Redis & MongoDB support (optional)
- ✅ Prometheus metrics
- ✅ CORS enabled
- ✅ Health checks

## 🐳 Docker Requirements

- Docker 20.10+
- Docker Compose 1.29+
- (Optional) NVIDIA GPU with CUDA for faster inference

## 📝 Notes

- First run downloads ~6GB model files
- Model files are cached in Docker volumes
- GPU acceleration is optional (CPU works too, just slower)
- Redis and MongoDB are optional - app works without them

## 🔍 Troubleshooting

**Port already in use?**
```bash
# Change port in docker-compose.yml or .env
API_PORT=8001
```

**Out of memory?**
```bash
# Reduce batch size or use CPU mode in docker-compose.yml
# Remove the 'deploy' section for CPU-only mode
```

**Model download slow?**
```bash
# Add your Hugging Face token to .env for faster downloads
HF_TOKEN=your_token_here
```

## 📄 License

MIT