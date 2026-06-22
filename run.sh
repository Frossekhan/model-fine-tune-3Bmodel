#!/bin/bash

# Enterprise AI Assistant - One-Click Run Script
# This script builds and runs the entire application

echo "========================================="
echo "Enterprise AI Assistant - Quick Start"
echo "========================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker found"
echo ""

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models/sentiment models/lora-qwen3b-amazon-valid chroma_store training/examples
echo "✅ Directories created"
echo ""

# Build and start the application
echo "🔨 Building Docker image..."
docker-compose build

echo ""
echo "🚀 Starting the application..."
docker-compose up -d

echo ""
echo "========================================="
echo "✅ Application is starting!"
echo "========================================="
echo ""
echo "📊 Access the application:"
echo "   - API: http://localhost:8000"
echo "   - Metrics: http://localhost:8000/metrics"
echo ""
echo "📝 View logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 Stop the application:"
echo "   docker-compose down"
echo ""
echo "⏳ First run will download the 3B model (~6GB). This may take a few minutes..."
echo "   Check progress with: docker-compose logs -f"
echo ""