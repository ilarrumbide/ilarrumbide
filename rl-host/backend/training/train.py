"""Training script for the restaurant RL host agent."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.models.restaurant import Table
from backend.models.rl_environment import RestaurantEnv
from backend.models.rl_agent import RestaurantRLAgent
from backend.training.data_generator import CustomerDataGenerator
from backend.utils.config import config
from backend.utils.helpers import get_zone_for_coordinates

import argparse
import json
from datetime import datetime


def create_default_restaurant_tables() -> list[Table]:
    """Create default restaurant table configuration."""
    restaurant_config = config.restaurant_config

    tables = []
    table_id = 0

    for zone_name, zone_config in restaurant_config["zones"].items():
        for coord in zone_config["coordinates"]:
            # Determine actual zone based on coordinates
            actual_zone = get_zone_for_coordinates(coord[0], coord[1])

            table = Table(
                id=table_id,
                capacity=restaurant_config["table_capacities"][table_id],
                coordinates=coord,
                zone=actual_zone
            )
            tables.append(table)
            table_id += 1

    return tables


def train_agent(
    num_episodes: int = 10000,
    timesteps_per_episode: int = 500,
    save_path: str = "saved_models/best_host",
    dataset_path: str = None,
    verbose: int = 1
):
    """Train the restaurant RL agent.

    Args:
        num_episodes: Number of training episodes
        timesteps_per_episode: Max timesteps per episode
        save_path: Path to save the trained model
        dataset_path: Path to pre-generated dataset (optional)
        verbose: Verbosity level
    """
    print("=" * 60)
    print("Restaurant RL Host - Training")
    print("=" * 60)

    # Create restaurant environment
    tables = create_default_restaurant_tables()
    env = RestaurantEnv(tables, max_queue_size=20)
    env.max_steps = timesteps_per_episode

    print(f"\nEnvironment created:")
    print(f"  - Tables: {len(tables)}")
    print(f"  - Max queue size: 20")
    print(f"  - Max steps per episode: {timesteps_per_episode}")

    # Create data generator
    generator = CustomerDataGenerator(seed=42)

    # Load or generate dataset
    if dataset_path and Path(dataset_path).exists():
        print(f"\nLoading dataset from {dataset_path}...")
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)
        print(f"  - Loaded {len(dataset)} scenarios")
    else:
        print("\nGenerating training dataset...")
        dataset = generator.generate_training_dataset(
            num_scenarios=min(num_episodes, 100),
            output_file="data/synthetic_dataset.json"
        )
        print(f"  - Generated {len(dataset)} scenarios")

    # Create RL agent
    rl_config = config.rl_config
    agent = RestaurantRLAgent(
        env=env,
        learning_rate=rl_config["learning_rate"],
        batch_size=rl_config["batch_size"],
        verbose=verbose
    )

    print("\nAgent configuration:")
    print(f"  - Algorithm: PPO")
    print(f"  - Learning rate: {rl_config['learning_rate']}")
    print(f"  - Batch size: {rl_config['batch_size']}")

    # Training loop with curriculum learning
    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    total_timesteps = num_episodes * timesteps_per_episode

    # Set up training scenarios (cycle through dataset)
    scenario_index = 0

    def get_next_scenario():
        nonlocal scenario_index
        scenario = dataset[scenario_index % len(dataset)]
        scenario_index += 1
        return generator.load_scenario_from_dict(scenario)

    # Custom training loop to inject scenarios
    print("\nPhase 1: Training with diverse scenarios...")

    # We'll train in batches, setting customer schedule before each episode
    episodes_per_batch = 10
    num_batches = num_episodes // episodes_per_batch

    for batch in range(num_batches):
        print(f"\nBatch {batch + 1}/{num_batches}")

        # Train for a batch
        agent.model.learn(
            total_timesteps=episodes_per_batch * timesteps_per_episode,
            callback=agent.training_callback,
            reset_num_timesteps=False,
            progress_bar=True
        )

        # Evaluate periodically
        if (batch + 1) % 10 == 0:
            print(f"\nEvaluating after batch {batch + 1}...")

            # Set a test scenario
            test_scenario = get_next_scenario()
            env.set_customer_schedule(test_scenario)

            eval_results = agent.evaluate(n_episodes=5, render=False)

            print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
            print(f"  Mean Served: {eval_results['mean_served']:.1f}")
            print(f"  Mean Lost: {eval_results['mean_lost']:.1f}")

            # Save checkpoint
            checkpoint_path = f"{save_path}_checkpoint_{batch + 1}"
            agent.save(checkpoint_path)

    # Save final model
    print(f"\n{'=' * 60}")
    print("Training Complete!")
    print(f"{'=' * 60}")

    agent.save(save_path)

    # Final evaluation
    print("\nFinal Evaluation:")
    print("-" * 60)

    final_results = agent.evaluate(n_episodes=20, render=False)

    print(f"\nResults over 20 episodes:")
    print(f"  Mean Reward: {final_results['mean_reward']:.2f} ± {final_results['std_reward']:.2f}")
    print(f"  Mean Episode Length: {final_results['mean_length']:.1f}")
    print(f"  Mean Customers Served: {final_results['mean_served']:.1f}")
    print(f"  Mean Customers Lost: {final_results['mean_lost']:.1f}")

    if final_results['mean_served'] > 0:
        satisfaction_rate = (final_results['mean_served'] /
                           (final_results['mean_served'] + final_results['mean_lost'])) * 100
        print(f"  Satisfaction Rate: {satisfaction_rate:.1f}%")

    print(f"\n{'=' * 60}")
    print(f"Model saved to: {save_path}")
    print(f"{'=' * 60}")

    return agent, final_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Restaurant RL Host")
    parser.add_argument("--episodes", type=int, default=1000,
                       help="Number of training episodes")
    parser.add_argument("--timesteps", type=int, default=500,
                       help="Timesteps per episode")
    parser.add_argument("--save-path", type=str, default="saved_models/best_host",
                       help="Path to save trained model")
    parser.add_argument("--dataset", type=str, default=None,
                       help="Path to pre-generated dataset")
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level")

    args = parser.parse_args()

    train_agent(
        num_episodes=args.episodes,
        timesteps_per_episode=args.timesteps,
        save_path=args.save_path,
        dataset_path=args.dataset,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
