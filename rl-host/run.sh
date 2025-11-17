#!/bin/bash

# Restaurant RL Host - Quick Start Script

echo "=========================================="
echo "Restaurant RL Host - Starting Server"
echo "=========================================="
echo ""

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "✅ uv found - using fast installation"
    USE_UV=true
else
    echo "⚠️  uv not found - using pip (slower)"
    echo "   Install uv for faster package management: curl -LsSf https://astral.sh/uv/install.sh | sh"
    USE_UV=false
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "⚠️  Virtual environment not found. Creating one..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
if [ "$USE_UV" = true ]; then
    uv pip install -r requirements.in
else
    pip install -q -r requirements.txt
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data saved_models logs/tensorboard frontend

# Check if trained model exists
if [ ! -f "saved_models/best_host.zip" ]; then
    echo ""
    echo "⚠️  No trained model found!"
    echo "The AI host will not be available until you train a model."
    echo ""
    echo "To train a model, run:"
    echo "  python backend/training/train.py --episodes 100"
    echo ""
    read -p "Do you want to train a model now? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🎓 Starting quick training (100 episodes)..."
        python backend/training/train.py --episodes 100 --timesteps 200
    fi
fi

echo ""
echo "=========================================="
echo "🚀 Starting FastAPI Server"
echo "=========================================="
echo ""
echo "Server will be available at:"
echo "  👉 http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
cd "$(dirname "$0")"
python backend/app.py
