# 🚀 Quick Start Guide

Get up and running with Restaurant RL Host in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- (Optional but recommended) [uv](https://github.com/astral-sh/uv) for fast package management

## Installation

### Step 1: Install uv (Recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2: Clone and navigate

```bash
cd rl-host
```

### Step 3: Install dependencies

**With uv (fast):**
```bash
uv pip install -r requirements.in
```

**With pip:**
```bash
pip install -r requirements.txt
```

### Step 4: Create directories

```bash
mkdir -p data saved_models logs/tensorboard
```

## Running Without Training

You can run the system immediately without a trained model:

```bash
./run.sh
```

Or manually:

```bash
python backend/app.py
```

Then open: **http://localhost:8000**

> **Note:** The AI recommendations will not be available until you train a model.

## Quick Training (10 minutes)

Train a simple model to test the system:

```bash
python backend/training/train.py --episodes 100 --timesteps 200
```

This creates a basic trained model at `saved_models/best_host.zip`.

## Full Training (1-2 hours)

For production-quality results:

```bash
python backend/training/train.py --episodes 1000 --timesteps 500
```

## Verify Installation

Run the test suite:

```bash
python test_system.py
```

Expected output:
```
✅ Configuration: All tests passed!
✅ Restaurant Models: All tests passed!
✅ Data Generator: All tests passed!
✅ RL Environment: All tests passed!
```

## Using the Application

### 1. Start the Server

```bash
python backend/app.py
# or
./run.sh
```

### 2. Open Web Interface

Navigate to **http://localhost:8000**

### 3. Try These Actions

**Manual Mode:**
- Click "➕ Add Customer" to manually add customers
- Click on tables to see their status
- Use "Get AI Recommendation" to see what the AI suggests

**Simulation Mode:**
- Select a scenario type (Normal, Rush, Slow)
- Set the speed multiplier (1x - 10x)
- Click "▶️ Start Simulation"
- Watch the AI manage the restaurant in real-time!

## Common Commands

### Training
```bash
# Quick training
python backend/training/train.py --episodes 100

# Full training
python backend/training/train.py --episodes 1000

# Custom training
python backend/training/train.py --episodes 500 --timesteps 300 --save-path my_model
```

### Evaluation
```bash
# Evaluate trained model
python backend/training/evaluate.py saved_models/best_host.zip

# Compare with baseline
python backend/training/evaluate.py saved_models/best_host.zip --compare

# Test on rush hour scenario
python backend/training/evaluate.py saved_models/best_host.zip --scenario rush
```

### Development
```bash
# Run tests
python test_system.py

# Format code
black backend/

# Start with auto-reload
uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000
```

## Troubleshooting

### "ModuleNotFoundError"
- Make sure you installed dependencies: `uv pip install -r requirements.in`
- Verify Python version: `python --version` (should be 3.8+)

### "Cannot connect to server"
- Check if port 8000 is available: `lsof -i :8000`
- Try a different port: `python backend/app.py` (then edit to use different port)

### "No trained model found"
- This is normal if you haven't trained yet
- The system will work, but AI decisions won't be available
- Train a model: `python backend/training/train.py --episodes 100`

### WebSocket disconnects frequently
- Check your firewall settings
- Try using `localhost` instead of `0.0.0.0`
- Check browser console for errors

## Next Steps

1. **Read the full README.md** for detailed documentation
2. **Experiment with different scenarios** to see how the AI adapts
3. **Train longer for better performance** (try 1000+ episodes)
4. **Modify the reward function** in `backend/utils/config.py` to customize behavior
5. **Add new features** - the code is modular and easy to extend!

## Performance Targets

A well-trained model should achieve:
- ✅ Average wait time < 10 minutes
- ✅ Customer satisfaction rate > 85%
- ✅ Table utilization > 75%
- ✅ Lost customers < 5%

## Support

- Check `README.md` for comprehensive documentation
- Review code comments for implementation details
- Run `python test_system.py` to verify installation

---

**Happy hosting! 🍽️**
