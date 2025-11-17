"""Configuration settings for the restaurant RL host system."""

import yaml
from pathlib import Path
from typing import Dict, List, Tuple

# Default configuration
DEFAULT_CONFIG = {
    "restaurant": {
        "total_tables": 10,
        "layout": "standard",
        "zones": {
            "inside": {
                "tables": 5,
                "coordinates": [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]
            },
            "outside": {
                "tables": 5,
                "coordinates": [(0, 1), (1, 1), (2, 1), (3, 1), (4, 1)]
            }
        },
        "table_capacities": [2, 2, 4, 4, 4, 5, 5, 6, 6, 8],
        "combine_distance_threshold": 1.5
    },
    "rl_model": {
        "algorithm": "PPO",
        "learning_rate": 3e-4,
        "batch_size": 64,
        "training_episodes": 10000,
        "update_frequency": 100,
        "evaluation_frequency": 100,
        "early_stopping_patience": 20
    },
    "simulation": {
        "speed": 1.0,
        "day_duration_hours": 14,
        "start_hour": 8,
        "end_hour": 22
    },
    "customer": {
        "patience_mean": 20,
        "patience_std": 10,
        "dining_duration_mean": 60,
        "dining_duration_std": 20
    },
    "rewards": {
        "seat_preferred": 10,
        "seat_alternative": 5,
        "wait_per_minute": -1,
        "customer_left": -20,
        "efficient_usage": 3,
        "inefficient_usage": -5,
        "large_group_success": 15,
        "unnecessary_wait": -10
    }
}


class Config:
    """Configuration manager for the restaurant RL system."""

    def __init__(self, config_path: str = None):
        """Initialize configuration."""
        self.config = DEFAULT_CONFIG.copy()

        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                custom_config = yaml.safe_load(f)
                self._update_config(custom_config)

    def _update_config(self, custom_config: Dict):
        """Recursively update configuration."""
        def update_dict(d, u):
            for k, v in u.items():
                if isinstance(v, dict):
                    d[k] = update_dict(d.get(k, {}), v)
                else:
                    d[k] = v
            return d

        self.config = update_dict(self.config, custom_config)

    def get(self, *keys):
        """Get configuration value by keys."""
        value = self.config
        for key in keys:
            value = value[key]
        return value

    @property
    def restaurant_config(self) -> Dict:
        return self.config["restaurant"]

    @property
    def rl_config(self) -> Dict:
        return self.config["rl_model"]

    @property
    def simulation_config(self) -> Dict:
        return self.config["simulation"]

    @property
    def customer_config(self) -> Dict:
        return self.config["customer"]

    @property
    def rewards_config(self) -> Dict:
        return self.config["rewards"]


# Global config instance
config = Config()
