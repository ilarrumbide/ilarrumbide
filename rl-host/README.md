# 🍽️ Restaurant RL Host

An AI-powered restaurant host management system using Reinforcement Learning to optimize table assignments and customer satisfaction in real-time.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![Stable-Baselines3](https://img.shields.io/badge/Stable--Baselines3-2.2+-orange.svg)
![uv](https://img.shields.io/badge/uv-enabled-blueviolet.svg)

> **⚡ Quick Start:** New to the project? Check out [QUICKSTART.md](QUICKSTART.md) for a 5-minute setup guide!

## 🎮 Features

- **Reinforcement Learning Agent**: PPO-based AI that learns to be the perfect restaurant host
- **Real-time Visualization**: Retro game-style top-down restaurant view
- **WebSocket Integration**: Live updates and interactive controls
- **Synthetic Data Generation**: Realistic customer arrival patterns
- **Performance Metrics**: Track satisfaction rate, wait times, and table efficiency
- **Multiple Scenarios**: Normal flow, rush hours, slow periods

## 🏗️ Project Structure

```
rl-host/
├── backend/
│   ├── app.py                      # FastAPI server with WebSocket
│   ├── models/
│   │   ├── restaurant.py           # Domain models (Table, CustomerGroup, Restaurant)
│   │   ├── rl_environment.py       # Gymnasium environment
│   │   └── rl_agent.py             # PPO agent wrapper
│   ├── training/
│   │   ├── train.py                # Training script
│   │   ├── data_generator.py       # Synthetic data generation
│   │   └── evaluate.py             # Model evaluation
│   └── utils/
│       ├── config.py               # Configuration management
│       └── helpers.py              # Utility functions
├── frontend/
│   ├── index.html                  # Main UI
│   ├── style.css                   # Retro game styling
│   └── app.js                      # Canvas visualization + WebSocket client
├── data/
│   └── synthetic_dataset.json      # Training data
├── saved_models/
│   └── best_host.zip               # Trained model
├── requirements.in                 # uv requirements specification
├── requirements.txt                # Generated from requirements.in
└── pyproject.toml                  # Project configuration
```

## 🚀 Quick Start

### Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

**Option 1: Using uv (Recommended)**

1. **Install uv** (if not already installed)
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Navigate to project directory**
```bash
cd rl-host
```

3. **Install dependencies with uv**
```bash
uv pip install -r requirements.in
```

**Option 2: Using pip**

```bash
cd rl-host
pip install -r requirements.txt
```

3. **Create necessary directories**
```bash
mkdir -p data saved_models logs/tensorboard
```

### Training the Model

Train the RL agent with synthetic data:

```bash
python backend/training/train.py --episodes 1000 --timesteps 500
```

**Training Options:**
- `--episodes`: Number of training episodes (default: 1000)
- `--timesteps`: Timesteps per episode (default: 500)
- `--save-path`: Path to save the model (default: saved_models/best_host)
- `--dataset`: Path to pre-generated dataset (optional)
- `--verbose`: Verbosity level (0, 1, or 2)

**Quick Training (for testing):**
```bash
python backend/training/train.py --episodes 100 --timesteps 200
```

### Evaluating the Model

Evaluate a trained model:

```bash
python backend/training/evaluate.py saved_models/best_host.zip --episodes 20
```

**Evaluation Options:**
- `--episodes`: Number of evaluation episodes
- `--scenario`: Scenario type (slow, normal, rush, mixed)
- `--render`: Render episodes to console
- `--compare`: Compare with random baseline

**Example with comparison:**
```bash
python backend/training/evaluate.py saved_models/best_host.zip --episodes 20 --compare
```

### Running the Application

Start the web server:

```bash
cd restaurant-rl-host
python backend/app.py
```

Or using uvicorn directly:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser to: **http://localhost:8000**

## 🎯 How It Works

### RL Environment

**State Space** (continuous):
- Table occupancy (normalized)
- Table capacities
- Queue length and composition
- Customer preferences distribution
- Time encoding (cyclical)
- Available table combinations

**Action Space** (discrete):
- Do nothing (wait for better opportunity)
- Seat at specific table (1-10)
- Suggest wait time (5, 10, 15, 20, 30 min)
- Seat at combined tables

**Reward Function**:
- `+10` for seating in preferred zone
- `+5` for seating in alternative zone
- `-1` per minute of wait time
- `-20` for customer leaving (impatience)
- `+3` for efficient table usage (>80% capacity)
- `-5` for inefficient usage (<50% capacity)
- `+15` for successfully handling large groups (>6)
- `-10` for unnecessary waiting

### Restaurant Configuration

**Default Layout**:
- 10 tables total
- 5 inside tables
- 5 outside tables
- Special zones: Window (2 tables), Bar (1 table)
- Capacities: [2, 2, 4, 4, 4, 5, 5, 6, 6, 8]
- Tables can be combined if distance ≤ 1.5 units

**Customer Attributes**:
- Group size: 1-15 people
- Zone preferences: inside, outside, window, bar
- Patience: 5-45 minutes
- Dining duration: 30-180 minutes
- Mood: evolves based on wait time

## 📊 Performance Targets

The AI aims to achieve:
- ✅ Average wait time < 10 minutes
- ✅ Customer satisfaction rate > 85%
- ✅ Table utilization efficiency > 75%
- ✅ Lost customers < 5% of total arrivals

## 🎨 Frontend Features

### Retro Game Style UI
- Pixel-art inspired design
- 8-bit style fonts (Press Start 2P, VT323)
- Neon color palette with glow effects
- Smooth animations

### Interactive Controls
- **Start/Stop Simulation**: Run automated scenarios
- **AI Recommendations**: Get AI's suggested action
- **Manual Actions**: Add customers and seat them manually
- **Real-time Stats**: Monitor performance metrics
- **Event Log**: Track all system events

### Visualization
- Top-down restaurant layout
- Color-coded tables (green=available, red=occupied, yellow=cleaning)
- Customer queue with mood indicators
- Zone highlighting
- Live statistics dashboard

## 🧪 API Endpoints

### REST API

- `GET /api/state` - Get current restaurant state
- `POST /api/reset` - Reset restaurant
- `POST /api/customer/add` - Add customer to queue
- `POST /api/customer/seat` - Manually seat customer
- `GET /api/ai/decision` - Get AI recommendation
- `POST /api/ai/execute` - Execute AI action
- `POST /api/simulation/start` - Start simulation
- `POST /api/simulation/stop` - Stop simulation

### WebSocket

Connect to `/ws` for real-time updates:

**Incoming messages:**
- `initial_state` - Initial restaurant state
- `state_update` - State changed
- `customer_arrived` - New customer
- `customer_seated` - Customer seated
- `customer_left` - Customer left
- `simulation_started/stopped` - Simulation status

**Outgoing messages:**
- `request_state` - Request current state
- `ping` - Keep alive

## 🔧 Configuration

Edit `backend/utils/config.py` to customize:

```python
DEFAULT_CONFIG = {
    "restaurant": {
        "total_tables": 10,
        "table_capacities": [2, 2, 4, 4, 4, 5, 5, 6, 6, 8],
        # ... more options
    },
    "rl_model": {
        "learning_rate": 3e-4,
        "batch_size": 64,
        # ... more options
    },
    "rewards": {
        "seat_preferred": 10,
        "customer_left": -20,
        # ... more options
    }
}
```

## 📈 Training Tips

1. **Start Small**: Begin with 100-200 episodes for testing
2. **Curriculum Learning**: The training script automatically increases difficulty
3. **Monitor Progress**: Check TensorBoard logs: `tensorboard --logdir logs/tensorboard`
4. **Save Checkpoints**: Models are saved every 10 batches
5. **Evaluate Often**: Run evaluation to check performance

## 🐛 Troubleshooting

**Issue: Training is slow**
- Reduce `--episodes` or `--timesteps`
- Use smaller batch size in config
- Check CPU/GPU usage

**Issue: Model not improving**
- Adjust learning rate in config
- Generate more diverse training scenarios
- Check reward function balance

**Issue: WebSocket disconnects**
- Check firewall settings
- Verify port 8000 is available
- Look at browser console for errors

**Issue: Frontend not updating**
- Hard refresh browser (Ctrl+F5)
- Check WebSocket connection status
- Verify backend is running

## 🎓 Advanced Features

### Custom Scenarios

Create custom test scenarios:

```python
from backend.training.data_generator import CustomerDataGenerator

generator = CustomerDataGenerator(seed=42)
scenario = generator.generate_scenario(
    duration_minutes=480,
    start_hour=18,  # Dinner service
    scenario_type="rush"
)
```

### A/B Testing

Compare different strategies:

```python
from backend.training.evaluate import compare_with_baseline

compare_with_baseline('saved_models/best_host.zip', n_episodes=20)
```

### Multi-Restaurant Support

Extend the system to handle multiple restaurant layouts by modifying the configuration.

## 📝 License

MIT License - feel free to use this project for learning and development.

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional RL algorithms (DQN, A2C, SAC)
- More sophisticated reward functions
- Mobile-responsive design
- Sound effects and animations
- Historical analytics dashboard
- Reservation system integration

## 🙏 Acknowledgments

- **Stable-Baselines3**: PPO implementation
- **Gymnasium**: RL environment framework
- **FastAPI**: Modern web framework
- **Press Start 2P & VT323**: Retro fonts

---

**Built with ❤️ using Reinforcement Learning**

For questions or issues, please open a GitHub issue.
