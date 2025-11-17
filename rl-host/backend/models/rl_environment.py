"""Gymnasium environment for restaurant host RL agent."""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
import math

from backend.models.restaurant import Restaurant, Table, CustomerGroup, TableStatus, CustomerMood
from backend.utils.config import config
from backend.utils.helpers import encode_time_cyclical, calculate_table_efficiency


class RestaurantEnv(gym.Env):
    """Custom Gymnasium environment for restaurant host optimization."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, tables: List[Table], max_queue_size: int = 20):
        """Initialize the restaurant environment.

        Args:
            tables: List of Table objects
            max_queue_size: Maximum size of waiting queue for state space
        """
        super().__init__()

        self.restaurant = Restaurant(tables)
        self.max_queue_size = max_queue_size
        self.num_tables = len(tables)

        # Configuration
        self.rewards = config.rewards_config

        # Define observation space
        # State components:
        # - Table occupancy (num_tables)
        # - Table capacity normalized (num_tables)
        # - Queue length (1)
        # - Queue composition: avg size, avg wait time, avg patience (3)
        # - Zone preference distribution (4 zones)
        # - Time encoding (2: sin, cos)
        # - Available combinations count (1)
        # - Current group features if exists (5: size, zone_pref_encoded, wait_time, patience, mood)

        obs_dim = (
            self.num_tables +  # occupancy
            self.num_tables +  # capacity
            1 +  # queue length
            3 +  # queue stats
            4 +  # zone preferences
            2 +  # time
            1 +  # combinations
            5    # current group features
        )

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Define action space
        # Actions:
        # 0: Do nothing (wait)
        # 1-10: Seat at table 1-10 (single table)
        # 11-20: Suggest wait time (5, 10, 15, 20, 30 min)
        # 21+: Seat at table combinations (top common combinations)

        max_actions = 1 + self.num_tables + 10 + 20  # wait + single tables + wait suggestions + combinations
        self.action_space = spaces.Discrete(max_actions)

        # Episode tracking
        self.current_step = 0
        self.max_steps = 500
        self.pending_groups: List[CustomerGroup] = []
        self.current_group_index = 0

        # Metrics
        self.episode_reward = 0.0
        self.customers_served = 0
        self.customers_lost = 0

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        self.restaurant.reset()
        self.current_step = 0
        self.current_group_index = 0
        self.episode_reward = 0.0
        self.customers_served = 0
        self.customers_lost = 0

        # Initialize with empty pending groups (will be set externally or generated)
        if options and "customer_schedule" in options:
            self.pending_groups = options["customer_schedule"]
        else:
            self.pending_groups = []

        obs = self._get_observation()
        info = self._get_info()

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step within the environment."""
        reward = 0.0
        self.current_step += 1

        # Process action
        action_reward, action_info = self._process_action(action)
        reward += action_reward

        # Update restaurant state
        self.restaurant.update(self.current_step)

        # Add new arrivals from pending groups
        while (self.current_group_index < len(self.pending_groups) and
               self.pending_groups[self.current_group_index].arrival_time <= self.current_step):
            group = self.pending_groups[self.current_group_index]
            self.restaurant.add_to_queue(group)
            self.current_group_index += 1

        # Check for customers who left
        initial_lost = self.customers_lost
        self.customers_lost = self.restaurant.total_customers_lost
        if self.customers_lost > initial_lost:
            reward += self.rewards["customer_left"] * (self.customers_lost - initial_lost)

        # Check episode termination
        terminated = self.current_step >= self.max_steps
        truncated = False

        # Get new observation
        obs = self._get_observation()
        info = self._get_info()
        info.update(action_info)

        self.episode_reward += reward

        return obs, reward, terminated, truncated, info

    def _process_action(self, action: int) -> Tuple[float, Dict]:
        """Process the selected action and return reward."""
        reward = 0.0
        info = {"action_type": "none", "action_success": False}

        # No customers waiting
        if not self.restaurant.waiting_queue:
            return 0.0, info

        current_group = self.restaurant.waiting_queue[0]

        # Action 0: Do nothing
        if action == 0:
            info["action_type"] = "wait"
            info["action_success"] = True
            # Small penalty for waiting when tables might be available
            if self._has_suitable_table(current_group):
                reward += self.rewards["unnecessary_wait"] * 0.1
            return reward, info

        # Actions 1-num_tables: Seat at single table
        elif 1 <= action <= self.num_tables:
            table_id = action - 1
            info["action_type"] = "seat_single"
            info["table_id"] = table_id

            if table_id in self.restaurant.tables:
                table = self.restaurant.tables[table_id]

                if table.is_available and table.capacity >= current_group.size:
                    success = self.restaurant.seat_group(current_group, [table_id])

                    if success:
                        info["action_success"] = True
                        self.customers_served += current_group.size

                        # Reward for seating
                        if table.zone == current_group.zone_preference:
                            reward += self.rewards["seat_preferred"]
                        elif table.zone in current_group.alternative_zones:
                            reward += self.rewards["seat_alternative"]

                        # Efficiency reward
                        efficiency = calculate_table_efficiency(table.capacity, current_group.size)
                        if efficiency >= 0.8:
                            reward += self.rewards["efficient_usage"]
                        elif efficiency < 0.5:
                            reward += self.rewards["inefficient_usage"]

                        # Large group bonus
                        if current_group.size > 6:
                            reward += self.rewards["large_group_success"]

                        # Wait time penalty
                        wait_time = current_group.current_wait_time
                        reward += self.rewards["wait_per_minute"] * wait_time

            return reward, info

        # Actions for wait suggestions (not directly seating, just informative)
        elif self.num_tables + 1 <= action <= self.num_tables + 10:
            info["action_type"] = "suggest_wait"
            wait_options = [5, 10, 15, 20, 30]
            wait_index = action - self.num_tables - 1
            if wait_index < len(wait_options):
                info["suggested_wait"] = wait_options[wait_index]
                info["action_success"] = True
            return reward, info

        # Actions for table combinations
        else:
            info["action_type"] = "seat_combination"
            combinations = self.restaurant.get_combinable_tables(current_group.size)

            if combinations:
                combo_index = action - (self.num_tables + 11)
                if combo_index < len(combinations):
                    table_ids = combinations[combo_index]
                    info["table_ids"] = table_ids

                    success = self.restaurant.seat_group(current_group, table_ids)

                    if success:
                        info["action_success"] = True
                        self.customers_served += current_group.size

                        # Check zone match
                        zones = [self.restaurant.tables[tid].zone for tid in table_ids]
                        if current_group.zone_preference in zones:
                            reward += self.rewards["seat_preferred"]
                        else:
                            reward += self.rewards["seat_alternative"] * 0.5

                        # Large group bonus
                        if current_group.size > 6:
                            reward += self.rewards["large_group_success"]

                        # Wait time penalty
                        wait_time = current_group.current_wait_time
                        reward += self.rewards["wait_per_minute"] * wait_time

            return reward, info

    def _has_suitable_table(self, group: CustomerGroup) -> bool:
        """Check if there's a suitable table for the group."""
        available = self.restaurant.get_available_tables()
        for table in available:
            if table.capacity >= group.size:
                return True

        # Check combinations
        combinations = self.restaurant.get_combinable_tables(group.size)
        return len(combinations) > 0

    def _get_observation(self) -> np.ndarray:
        """Get current observation state."""
        obs = []

        # Table occupancy
        for table in self.restaurant.tables.values():
            obs.append(1.0 if table.status == TableStatus.OCCUPIED else 0.0)

        # Table capacity (normalized)
        max_capacity = max(t.capacity for t in self.restaurant.tables.values())
        for table in self.restaurant.tables.values():
            obs.append(table.capacity / max_capacity)

        # Queue length (normalized)
        obs.append(min(len(self.restaurant.waiting_queue) / self.max_queue_size, 1.0))

        # Queue statistics
        if self.restaurant.waiting_queue:
            avg_size = np.mean([g.size for g in self.restaurant.waiting_queue]) / 15.0  # normalize
            avg_wait = np.mean([self.current_step - g.arrival_time for g in self.restaurant.waiting_queue]) / 60.0
            avg_patience = np.mean([g.patience_minutes for g in self.restaurant.waiting_queue]) / 60.0
            obs.extend([avg_size, min(avg_wait, 1.0), min(avg_patience, 1.0)])
        else:
            obs.extend([0.0, 0.0, 0.0])

        # Zone preference distribution
        zone_counts = {"inside": 0, "outside": 0, "window": 0, "bar": 0}
        if self.restaurant.waiting_queue:
            for group in self.restaurant.waiting_queue:
                if group.zone_preference in zone_counts:
                    zone_counts[group.zone_preference] += 1
            total = len(self.restaurant.waiting_queue)
            obs.extend([zone_counts[z] / total for z in ["inside", "outside", "window", "bar"]])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])

        # Time encoding (normalized to 0-1 range, then sin/cos)
        hour = (self.current_step / 60.0) % 24
        sin_time, cos_time = encode_time_cyclical(int(hour))
        obs.extend([(sin_time + 1) / 2, (cos_time + 1) / 2])  # normalize to 0-1

        # Available combinations count
        if self.restaurant.waiting_queue:
            combinations = self.restaurant.get_combinable_tables(self.restaurant.waiting_queue[0].size)
            obs.append(min(len(combinations) / 10.0, 1.0))
        else:
            obs.append(0.0)

        # Current group features (if exists)
        if self.restaurant.waiting_queue:
            group = self.restaurant.waiting_queue[0]
            obs.append(group.size / 15.0)  # normalized size

            # Zone preference encoded
            zone_encoding = {"inside": 0.25, "outside": 0.5, "window": 0.75, "bar": 1.0}
            obs.append(zone_encoding.get(group.zone_preference, 0.0))

            # Wait time
            wait_time = self.current_step - group.arrival_time
            obs.append(min(wait_time / 60.0, 1.0))

            # Patience
            obs.append(min(group.patience_minutes / 60.0, 1.0))

            # Mood score
            obs.append(group.mood_score)
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])

        return np.array(obs, dtype=np.float32)

    def _get_info(self) -> Dict:
        """Get additional information about current state."""
        return {
            "current_step": self.current_step,
            "queue_length": len(self.restaurant.waiting_queue),
            "seated_groups": len(self.restaurant.seated_groups),
            "customers_served": self.customers_served,
            "customers_lost": self.customers_lost,
            "episode_reward": self.episode_reward,
            "available_tables": len(self.restaurant.get_available_tables())
        }

    def render(self):
        """Render the environment (optional)."""
        if self.render_mode == "human":
            print(f"\n=== Step {self.current_step} ===")
            print(f"Queue: {len(self.restaurant.waiting_queue)}")
            print(f"Seated: {len(self.restaurant.seated_groups)}")
            print(f"Available tables: {len(self.restaurant.get_available_tables())}")

    def set_customer_schedule(self, groups: List[CustomerGroup]):
        """Set the customer arrival schedule for the episode."""
        self.pending_groups = sorted(groups, key=lambda g: g.arrival_time)
        self.current_group_index = 0
