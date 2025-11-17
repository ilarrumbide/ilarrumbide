"""Evaluation script for the trained restaurant RL host agent."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.models.restaurant import Table
from backend.models.rl_environment import RestaurantEnv
from backend.models.rl_agent import RestaurantRLAgent
from backend.training.data_generator import CustomerDataGenerator, generate_quick_scenario
from backend.utils.config import config
from backend.utils.helpers import get_zone_for_coordinates

import argparse
import json
import numpy as np
from datetime import datetime


def create_default_restaurant_tables() -> list[Table]:
    """Create default restaurant table configuration."""
    restaurant_config = config.restaurant_config

    tables = []
    table_id = 0

    for zone_name, zone_config in restaurant_config["zones"].items():
        for coord in zone_config["coordinates"]:
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


def evaluate_agent(
    model_path: str,
    n_episodes: int = 20,
    scenario_type: str = "normal",
    render: bool = False,
    verbose: int = 1
):
    """Evaluate a trained agent.

    Args:
        model_path: Path to the trained model
        n_episodes: Number of episodes to evaluate
        scenario_type: Type of scenario to test
        render: Whether to render episodes
        verbose: Verbosity level
    """
    print("=" * 60)
    print("Restaurant RL Host - Evaluation")
    print("=" * 60)

    # Create environment
    tables = create_default_restaurant_tables()
    env = RestaurantEnv(tables, max_queue_size=20)

    print(f"\nEnvironment: {len(tables)} tables")

    # Load agent
    print(f"\nLoading model from: {model_path}")
    agent = RestaurantRLAgent.load_agent(model_path, env)

    # Create data generator
    generator = CustomerDataGenerator(seed=123)

    print(f"\nEvaluating on {n_episodes} episodes ({scenario_type} scenarios)...")
    print("-" * 60)

    # Track detailed metrics
    all_rewards = []
    all_served = []
    all_lost = []
    all_wait_times = []
    all_efficiencies = []

    for episode in range(n_episodes):
        # Generate scenario
        scenario = generator.generate_scenario(
            duration_minutes=480,
            start_hour=11,
            scenario_type=scenario_type
        )

        env.set_customer_schedule(scenario)

        # Run episode
        obs, info = env.reset()
        done = False
        episode_reward = 0
        step = 0
        wait_times = []

        while not done:
            action = agent.predict(obs, deterministic=True)

            if render and verbose > 1:
                action_explanation = agent.get_action_explanation(action)
                print(f"Step {step}: {action_explanation}")

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            episode_reward += reward
            step += 1

            # Track wait times for served customers
            for group in env.restaurant.seated_groups.values():
                if group.seated_time is not None:
                    wait_time = group.seated_time - group.arrival_time
                    wait_times.append(wait_time)

        # Calculate efficiency
        total_capacity = sum(t.capacity for t in env.restaurant.tables.values())
        total_seated = len(env.restaurant.seated_groups) + len(env.restaurant.departed_groups)
        efficiency = (total_seated / total_capacity * 100) if total_capacity > 0 else 0

        # Store metrics
        all_rewards.append(episode_reward)
        all_served.append(info["customers_served"])
        all_lost.append(info["customers_lost"])
        all_efficiencies.append(efficiency)

        if wait_times:
            all_wait_times.extend(wait_times)

        if verbose > 0:
            print(f"Episode {episode + 1:2d}: "
                  f"Reward={episode_reward:6.1f}, "
                  f"Served={info['customers_served']:2d}, "
                  f"Lost={info['customers_lost']:2d}, "
                  f"Efficiency={efficiency:.1f}%")

    # Calculate summary statistics
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"\nPerformance Metrics ({n_episodes} episodes):")
    print(f"  Mean Reward: {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean Customers Served: {np.mean(all_served):.1f} ± {np.std(all_served):.1f}")
    print(f"  Mean Customers Lost: {np.mean(all_lost):.1f} ± {np.std(all_lost):.1f}")

    if sum(all_served) > 0:
        total_served = sum(all_served)
        total_lost = sum(all_lost)
        satisfaction_rate = (total_served / (total_served + total_lost)) * 100
        print(f"  Overall Satisfaction Rate: {satisfaction_rate:.1f}%")

    if all_wait_times:
        print(f"  Mean Wait Time: {np.mean(all_wait_times):.1f} min ± {np.std(all_wait_times):.1f}")
        print(f"  Median Wait Time: {np.median(all_wait_times):.1f} min")
        print(f"  Max Wait Time: {np.max(all_wait_times):.1f} min")

    print(f"  Mean Table Efficiency: {np.mean(all_efficiencies):.1f}%")

    # Performance targets
    print("\n" + "-" * 60)
    print("Target Achievement:")
    avg_wait = np.mean(all_wait_times) if all_wait_times else 0
    satisfaction = satisfaction_rate if sum(all_served) > 0 else 0
    efficiency = np.mean(all_efficiencies)

    print(f"  {'✓' if avg_wait < 10 else '✗'} Average wait time < 10 min: {avg_wait:.1f} min")
    print(f"  {'✓' if satisfaction > 85 else '✗'} Satisfaction rate > 85%: {satisfaction:.1f}%")
    print(f"  {'✓' if efficiency > 75 else '✗'} Table efficiency > 75%: {efficiency:.1f}%")
    print(f"  {'✓' if np.mean(all_lost) < 2 else '✗'} Lost customers < 2 per episode: {np.mean(all_lost):.1f}")

    print("\n" + "=" * 60)

    return {
        "mean_reward": np.mean(all_rewards),
        "mean_served": np.mean(all_served),
        "mean_lost": np.mean(all_lost),
        "satisfaction_rate": satisfaction,
        "mean_wait_time": avg_wait,
        "efficiency": efficiency
    }


def compare_with_baseline(model_path: str, n_episodes: int = 10):
    """Compare trained agent with random baseline.

    Args:
        model_path: Path to trained model
        n_episodes: Number of episodes to compare
    """
    print("\n" + "=" * 60)
    print("Comparing with Random Baseline")
    print("=" * 60)

    # Evaluate trained agent
    print("\n[1] Evaluating Trained Agent...")
    agent_results = evaluate_agent(model_path, n_episodes, verbose=0)

    # Evaluate random baseline
    print("\n[2] Evaluating Random Baseline...")

    tables = create_default_restaurant_tables()
    env = RestaurantEnv(tables)
    generator = CustomerDataGenerator(seed=123)

    baseline_served = []
    baseline_lost = []

    for episode in range(n_episodes):
        scenario = generator.generate_scenario(duration_minutes=480)
        env.set_customer_schedule(scenario)

        obs, info = env.reset()
        done = False

        while not done:
            # Random action
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

        baseline_served.append(info["customers_served"])
        baseline_lost.append(info["customers_lost"])

    baseline_satisfaction = (sum(baseline_served) /
                            (sum(baseline_served) + sum(baseline_lost))) * 100

    # Print comparison
    print("\n" + "=" * 60)
    print("Comparison Results")
    print("=" * 60)

    print(f"\n{'Metric':<25} {'Trained Agent':<15} {'Random Baseline':<15} {'Improvement'}")
    print("-" * 70)

    improvement_served = ((agent_results['mean_served'] - np.mean(baseline_served)) /
                          np.mean(baseline_served) * 100)
    print(f"{'Customers Served':<25} {agent_results['mean_served']:<15.1f} "
          f"{np.mean(baseline_served):<15.1f} {improvement_served:+.1f}%")

    improvement_lost = ((np.mean(baseline_lost) - agent_results['mean_lost']) /
                        np.mean(baseline_lost) * 100)
    print(f"{'Customers Lost':<25} {agent_results['mean_lost']:<15.1f} "
          f"{np.mean(baseline_lost):<15.1f} {improvement_lost:+.1f}%")

    improvement_sat = agent_results['satisfaction_rate'] - baseline_satisfaction
    print(f"{'Satisfaction Rate':<25} {agent_results['satisfaction_rate']:<15.1f}% "
          f"{baseline_satisfaction:<15.1f}% {improvement_sat:+.1f}%")

    print("\n" + "=" * 60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate Restaurant RL Host")
    parser.add_argument("model_path", type=str,
                       help="Path to trained model")
    parser.add_argument("--episodes", type=int, default=20,
                       help="Number of evaluation episodes")
    parser.add_argument("--scenario", type=str, default="normal",
                       choices=["slow", "normal", "rush", "mixed"],
                       help="Scenario type")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes")
    parser.add_argument("--compare", action="store_true",
                       help="Compare with random baseline")
    parser.add_argument("--verbose", type=int, default=1,
                       help="Verbosity level")

    args = parser.parse_args()

    # Evaluate
    evaluate_agent(
        args.model_path,
        n_episodes=args.episodes,
        scenario_type=args.scenario,
        render=args.render,
        verbose=args.verbose
    )

    # Compare with baseline if requested
    if args.compare:
        compare_with_baseline(args.model_path, n_episodes=10)


if __name__ == "__main__":
    main()
