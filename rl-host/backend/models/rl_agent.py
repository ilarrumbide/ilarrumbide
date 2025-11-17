"""RL Agent for restaurant host using PPO algorithm."""

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from typing import Optional, Dict, List
import numpy as np
from pathlib import Path

from backend.models.rl_environment import RestaurantEnv
from backend.utils.config import config


class TrainingCallback(BaseCallback):
    """Custom callback for training progress tracking."""

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0

    def _on_step(self) -> bool:
        """Called at each step."""
        self.current_episode_reward += self.locals["rewards"][0]
        self.current_episode_length += 1

        # Check if episode ended
        if self.locals["dones"][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)

            if self.verbose > 0 and len(self.episode_rewards) % 10 == 0:
                mean_reward = np.mean(self.episode_rewards[-10:])
                mean_length = np.mean(self.episode_lengths[-10:])
                print(f"Episode {len(self.episode_rewards)}: "
                      f"Mean Reward = {mean_reward:.2f}, "
                      f"Mean Length = {mean_length:.0f}")

            self.current_episode_reward = 0
            self.current_episode_length = 0

        return True


class RestaurantRLAgent:
    """Wrapper for the restaurant RL agent using PPO."""

    def __init__(
        self,
        env: RestaurantEnv,
        learning_rate: float = 3e-4,
        batch_size: int = 64,
        n_steps: int = 2048,
        verbose: int = 1
    ):
        """Initialize the RL agent.

        Args:
            env: Restaurant environment
            learning_rate: Learning rate for PPO
            batch_size: Batch size for training
            n_steps: Number of steps to run for each environment per update
            verbose: Verbosity level
        """
        self.env = env
        self.verbose = verbose

        # Create vectorized environment
        self.vec_env = DummyVecEnv([lambda: env])

        # Initialize PPO model
        self.model = PPO(
            "MlpPolicy",
            self.vec_env,
            learning_rate=learning_rate,
            batch_size=batch_size,
            n_steps=n_steps,
            verbose=verbose,
            tensorboard_log="./logs/tensorboard/"
        )

        self.training_callback = TrainingCallback(verbose=verbose)

    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 10000,
        save_path: str = "saved_models/restaurant_host",
        early_stopping_patience: int = 20
    ) -> Dict:
        """Train the agent.

        Args:
            total_timesteps: Total timesteps to train
            eval_freq: Frequency of evaluation
            save_path: Path to save best model
            early_stopping_patience: Patience for early stopping

        Returns:
            Training statistics
        """
        print(f"Starting training for {total_timesteps} timesteps...")

        # Create save directory
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)

        # Train the model
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=self.training_callback,
            progress_bar=True
        )

        # Save final model
        self.model.save(save_path)
        print(f"Model saved to {save_path}")

        # Return statistics
        return {
            "episode_rewards": self.training_callback.episode_rewards,
            "episode_lengths": self.training_callback.episode_lengths,
            "mean_reward": np.mean(self.training_callback.episode_rewards[-100:]) if self.training_callback.episode_rewards else 0,
            "total_episodes": len(self.training_callback.episode_rewards)
        }

    def predict(self, observation, deterministic: bool = True):
        """Predict action for given observation.

        Args:
            observation: Current state observation
            deterministic: Whether to use deterministic policy

        Returns:
            Predicted action and additional info
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """Save the model."""
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load a trained model."""
        self.model = PPO.load(path, env=self.vec_env)
        print(f"Model loaded from {path}")

    @staticmethod
    def load_agent(path: str, env: RestaurantEnv) -> 'RestaurantRLAgent':
        """Load a trained agent from file.

        Args:
            path: Path to saved model
            env: Environment to use

        Returns:
            Loaded agent
        """
        agent = RestaurantRLAgent(env)
        agent.load(path)
        return agent

    def evaluate(
        self,
        n_episodes: int = 10,
        render: bool = False
    ) -> Dict:
        """Evaluate the agent.

        Args:
            n_episodes: Number of episodes to evaluate
            render: Whether to render the environment

        Returns:
            Evaluation metrics
        """
        episode_rewards = []
        episode_lengths = []
        customers_served_list = []
        customers_lost_list = []

        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0

            while not done:
                action = self.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

                episode_reward += reward
                episode_length += 1

                if render:
                    self.env.render()

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            customers_served_list.append(info["customers_served"])
            customers_lost_list.append(info["customers_lost"])

            if self.verbose > 0:
                print(f"Episode {episode + 1}/{n_episodes}: "
                      f"Reward = {episode_reward:.2f}, "
                      f"Served = {info['customers_served']}, "
                      f"Lost = {info['customers_lost']}")

        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "mean_length": np.mean(episode_lengths),
            "mean_served": np.mean(customers_served_list),
            "mean_lost": np.mean(customers_lost_list),
            "episode_rewards": episode_rewards
        }

    def get_action_explanation(self, action: int) -> str:
        """Get human-readable explanation of action.

        Args:
            action: Action number

        Returns:
            Explanation string
        """
        num_tables = len(self.env.restaurant.tables)

        if action == 0:
            return "Wait - No action taken"
        elif 1 <= action <= num_tables:
            table_id = action - 1
            return f"Seat at Table {table_id}"
        elif num_tables + 1 <= action <= num_tables + 10:
            wait_options = [5, 10, 15, 20, 30]
            wait_index = action - num_tables - 1
            if wait_index < len(wait_options):
                return f"Suggest {wait_options[wait_index]} minute wait"
            return "Invalid wait suggestion"
        else:
            combo_index = action - (num_tables + 11)
            if self.env.restaurant.waiting_queue:
                combinations = self.env.restaurant.get_combinable_tables(
                    self.env.restaurant.waiting_queue[0].size
                )
                if combo_index < len(combinations):
                    table_ids = combinations[combo_index]
                    return f"Seat at combined tables {table_ids}"
            return f"Seat at table combination {combo_index}"
