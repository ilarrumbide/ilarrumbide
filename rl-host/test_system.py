"""Quick test script to verify the system works correctly."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.models.restaurant import Table, CustomerGroup, Restaurant
from backend.models.rl_environment import RestaurantEnv
from backend.training.data_generator import CustomerDataGenerator, generate_quick_scenario
from backend.utils.config import config
from backend.utils.helpers import get_zone_for_coordinates

import numpy as np


def test_restaurant_models():
    """Test basic restaurant models."""
    print("\n" + "="*60)
    print("Testing Restaurant Models")
    print("="*60)

    # Create tables
    tables = []
    for i in range(5):
        table = Table(
            id=i,
            capacity=[2, 4, 4, 6, 8][i],
            coordinates=(i, 0),
            zone="inside"
        )
        tables.append(table)

    print(f"✓ Created {len(tables)} tables")

    # Create restaurant
    restaurant = Restaurant(tables)
    print(f"✓ Created restaurant with {len(restaurant.tables)} tables")

    # Create customer group
    group = CustomerGroup(
        id=1,
        size=4,
        arrival_time=0,
        zone_preference="inside",
        alternative_zones=["outside"],
        patience_minutes=20,
        expected_dining_minutes=60
    )

    print(f"✓ Created customer group (size: {group.size})")

    # Add to queue
    restaurant.add_to_queue(group)
    print(f"✓ Added customer to queue (queue length: {len(restaurant.waiting_queue)})")

    # Seat customer
    success = restaurant.seat_group(group, [1])
    print(f"✓ Seated customer: {success}")

    assert success, "Failed to seat customer"
    assert len(restaurant.seated_groups) == 1, "Customer not in seated groups"
    assert len(restaurant.waiting_queue) == 0, "Queue should be empty"

    print("\n✅ Restaurant Models: All tests passed!")
    return True


def test_data_generator():
    """Test synthetic data generation."""
    print("\n" + "="*60)
    print("Testing Data Generator")
    print("="*60)

    generator = CustomerDataGenerator(seed=42)
    print("✓ Created data generator")

    # Generate single group
    group = generator.generate_customer_group(arrival_time=0, time_of_day=12)
    print(f"✓ Generated customer group (size: {group.size}, zone: {group.zone_preference})")

    # Generate scenario
    scenario = generator.generate_scenario(
        duration_minutes=120,
        scenario_type="normal"
    )
    print(f"✓ Generated scenario with {len(scenario)} customers")

    assert len(scenario) > 0, "Scenario should have customers"

    # Generate dataset
    dataset = generator.generate_training_dataset(num_scenarios=5)
    print(f"✓ Generated training dataset with {len(dataset)} scenarios")

    assert len(dataset) == 5, "Should have 5 scenarios"

    print("\n✅ Data Generator: All tests passed!")
    return True


def test_rl_environment():
    """Test RL environment."""
    print("\n" + "="*60)
    print("Testing RL Environment")
    print("="*60)

    # Create tables
    tables = []
    capacities = [2, 2, 4, 4, 4, 5, 5, 6, 6, 8]

    for i in range(10):
        x = i % 5
        y = i // 5
        zone = get_zone_for_coordinates(x, y)

        table = Table(
            id=i,
            capacity=capacities[i],
            coordinates=(x, y),
            zone=zone
        )
        tables.append(table)

    print(f"✓ Created {len(tables)} tables")

    # Create environment
    env = RestaurantEnv(tables)
    print(f"✓ Created RL environment")

    # Test observation space
    print(f"  - Observation space shape: {env.observation_space.shape}")
    print(f"  - Action space: {env.action_space.n} actions")

    # Generate test scenario
    scenario = generate_quick_scenario(num_customers=10, seed=42)
    env.set_customer_schedule(scenario)
    print(f"✓ Set customer schedule ({len(scenario)} customers)")

    # Reset environment
    obs, info = env.reset()
    print(f"✓ Reset environment")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Info keys: {list(info.keys())}")

    # Take random actions
    total_reward = 0
    for i in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    print(f"✓ Executed {i+1} steps (total reward: {total_reward:.2f})")

    print("\n✅ RL Environment: All tests passed!")
    return True


def test_configuration():
    """Test configuration system."""
    print("\n" + "="*60)
    print("Testing Configuration")
    print("="*60)

    # Test config loading
    restaurant_config = config.restaurant_config
    print(f"✓ Loaded restaurant config")
    print(f"  - Total tables: {restaurant_config['total_tables']}")

    rl_config = config.rl_config
    print(f"✓ Loaded RL config")
    print(f"  - Algorithm: {rl_config['algorithm']}")
    print(f"  - Learning rate: {rl_config['learning_rate']}")

    rewards_config = config.rewards_config
    print(f"✓ Loaded rewards config")
    print(f"  - Seat preferred: {rewards_config['seat_preferred']}")

    print("\n✅ Configuration: All tests passed!")
    return True


def run_all_tests():
    """Run all system tests."""
    print("\n" + "="*70)
    print(" " * 15 + "RESTAURANT RL HOST - SYSTEM TESTS")
    print("="*70)

    tests = [
        ("Configuration", test_configuration),
        ("Restaurant Models", test_restaurant_models),
        ("Data Generator", test_data_generator),
        ("RL Environment", test_rl_environment),
    ]

    results = []

    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name}: FAILED")
            print(f"   Error: {str(e)}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{name:<30} {status}")

    print("="*70)
    print(f"Results: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("  1. Train the model: python backend/training/train.py --episodes 100")
        print("  2. Run the server: python backend/app.py")
        print("  3. Open browser: http://localhost:8000")
        return True
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
