"""Generate synthetic customer data for training the RL agent."""

import numpy as np
import json
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import random

from backend.models.restaurant import CustomerGroup, CustomerMood
from backend.utils.helpers import weighted_random_choice, get_rush_hour_multiplier
from backend.utils.config import config


class CustomerDataGenerator:
    """Generate realistic customer arrival patterns and preferences."""

    def __init__(self, seed: int = None):
        """Initialize the data generator.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        self.customer_config = config.customer_config
        self.next_id = 1

        # Zone preferences distribution
        self.zone_weights = {
            "inside": 0.40,
            "outside": 0.30,
            "window": 0.20,
            "bar": 0.10
        }

        # Group size distribution
        self.size_distribution = {
            1: 0.10,
            2: 0.35,
            3: 0.15,
            4: 0.15,
            5: 0.10,
            6: 0.05,
            7: 0.04,
            8: 0.03,
            9: 0.02,
            10: 0.01
        }

        # Special requirements pool
        self.special_requirements_pool = [
            "wheelchair",
            "highchair",
            "quiet",
            "allergy_friendly",
            "celebration",
            "business_meeting"
        ]

    def generate_group_size(self) -> int:
        """Generate a group size based on distribution."""
        sizes = list(self.size_distribution.keys())
        weights = list(self.size_distribution.values())
        return weighted_random_choice(sizes, weights)

    def generate_zone_preference(self) -> Tuple[str, List[str]]:
        """Generate zone preference and alternatives.

        Returns:
            Tuple of (primary_zone, alternative_zones)
        """
        zones = list(self.zone_weights.keys())
        weights = list(self.zone_weights.values())
        primary = weighted_random_choice(zones, weights)

        # Generate alternatives (1-2 other zones)
        alternatives = [z for z in zones if z != primary]
        num_alternatives = random.randint(1, 2)
        alternatives = random.sample(alternatives, num_alternatives)

        return primary, alternatives

    def generate_patience(self, group_size: int, time_of_day: int) -> float:
        """Generate patience in minutes based on group characteristics.

        Args:
            group_size: Size of the group
            time_of_day: Hour of the day (0-23)

        Returns:
            Patience in minutes
        """
        # Base patience from config
        base_mean = self.customer_config["patience_mean"]
        base_std = self.customer_config["patience_std"]

        # Adjustments
        # Larger groups tend to be more patient
        size_adjustment = (group_size - 2) * 2

        # Rush hour makes people less patient
        rush_multiplier = get_rush_hour_multiplier(time_of_day)
        rush_adjustment = -5 if rush_multiplier > 1.5 else 0

        # Weekend makes people more patient
        # (We'll assume this is controlled externally)

        adjusted_mean = base_mean + size_adjustment + rush_adjustment
        patience = np.random.normal(adjusted_mean, base_std)

        return max(5.0, min(45.0, patience))  # Clamp to 5-45 minutes

    def generate_dining_duration(self, group_size: int, time_of_day: int) -> float:
        """Generate expected dining duration in minutes.

        Args:
            group_size: Size of the group
            time_of_day: Hour of the day (0-23)

        Returns:
            Dining duration in minutes
        """
        # Breakfast: shorter (30-60 min)
        if 7 <= time_of_day < 11:
            base_mean = 40
            base_std = 10

        # Lunch: medium (45-75 min)
        elif 11 <= time_of_day < 15:
            base_mean = 60
            base_std = 15

        # Dinner: longer (60-120 min)
        elif 17 <= time_of_day < 23:
            base_mean = 90
            base_std = 20

        # Off-peak: variable
        else:
            base_mean = 60
            base_std = 15

        # Larger groups take longer
        size_adjustment = (group_size - 2) * 5

        adjusted_mean = base_mean + size_adjustment
        duration = np.random.normal(adjusted_mean, base_std)

        return max(30.0, min(180.0, duration))  # Clamp to 30-180 minutes

    def generate_special_requirements(self, group_size: int) -> List[str]:
        """Generate special requirements for the group.

        Args:
            group_size: Size of the group

        Returns:
            List of special requirements
        """
        requirements = []

        # 20% chance of having special requirements
        if random.random() < 0.2:
            # Larger groups more likely to have requirements
            num_requirements = 1 if group_size <= 4 else random.randint(1, 2)

            requirements = random.sample(
                self.special_requirements_pool,
                min(num_requirements, len(self.special_requirements_pool))
            )

        return requirements

    def generate_customer_group(
        self,
        arrival_time: float,
        time_of_day: int
    ) -> CustomerGroup:
        """Generate a single customer group.

        Args:
            arrival_time: Arrival time in minutes from start
            time_of_day: Hour of the day (0-23)

        Returns:
            CustomerGroup object
        """
        group_size = self.generate_group_size()
        zone_pref, alternatives = self.generate_zone_preference()
        patience = self.generate_patience(group_size, time_of_day)
        dining_duration = self.generate_dining_duration(group_size, time_of_day)
        special_reqs = self.generate_special_requirements(group_size)

        group = CustomerGroup(
            id=self.next_id,
            size=group_size,
            arrival_time=arrival_time,
            zone_preference=zone_pref,
            alternative_zones=alternatives,
            patience_minutes=patience,
            expected_dining_minutes=dining_duration,
            special_requirements=special_reqs,
            willing_to_wait=random.random() < 0.85  # 85% willing to wait
        )

        self.next_id += 1
        return group

    def generate_scenario(
        self,
        duration_minutes: int = 480,
        start_hour: int = 8,
        scenario_type: str = "normal"
    ) -> List[CustomerGroup]:
        """Generate a complete scenario of customer arrivals.

        Args:
            duration_minutes: Duration of the scenario in minutes (default 8 hours)
            start_hour: Starting hour of the day (0-23)
            scenario_type: Type of scenario ('normal', 'rush', 'slow', 'mixed')

        Returns:
            List of CustomerGroup objects sorted by arrival time
        """
        groups = []
        current_minute = 0

        while current_minute < duration_minutes:
            # Calculate current hour
            current_hour = (start_hour + current_minute // 60) % 24

            # Determine arrival rate based on scenario type and time
            base_rate = self._get_base_rate(scenario_type)
            rush_multiplier = get_rush_hour_multiplier(current_hour)
            arrival_rate = base_rate * rush_multiplier

            # Generate inter-arrival time (exponential distribution)
            if arrival_rate > 0:
                inter_arrival = np.random.exponential(1.0 / arrival_rate)
            else:
                inter_arrival = 60  # Default to 1 hour if rate is 0

            current_minute += inter_arrival

            if current_minute < duration_minutes:
                group = self.generate_customer_group(current_minute, current_hour)
                groups.append(group)

        return sorted(groups, key=lambda g: g.arrival_time)

    def _get_base_rate(self, scenario_type: str) -> float:
        """Get base arrival rate for scenario type.

        Args:
            scenario_type: Type of scenario

        Returns:
            Base arrival rate (customers per minute)
        """
        rates = {
            "slow": 0.05,      # ~3 per hour
            "normal": 0.1,     # ~6 per hour
            "rush": 0.2,       # ~12 per hour
            "mixed": 0.1       # ~6 per hour (will vary with time)
        }
        return rates.get(scenario_type, 0.1)

    def generate_training_dataset(
        self,
        num_scenarios: int = 100,
        output_file: str = None
    ) -> List[Dict]:
        """Generate a dataset of scenarios for training.

        Args:
            num_scenarios: Number of scenarios to generate
            output_file: Optional file path to save the dataset

        Returns:
            List of scenario dictionaries
        """
        dataset = []

        scenario_types = ["slow", "normal", "rush", "mixed"]
        durations = [240, 360, 480]  # 4, 6, 8 hours

        for i in range(num_scenarios):
            scenario_type = random.choice(scenario_types)
            duration = random.choice(durations)
            start_hour = random.choice([8, 11, 17])  # Breakfast, lunch, or dinner

            groups = self.generate_scenario(duration, start_hour, scenario_type)

            scenario_dict = {
                "scenario_id": f"scenario_{i:04d}",
                "scenario_type": scenario_type,
                "duration_minutes": duration,
                "start_hour": start_hour,
                "num_customers": len(groups),
                "customers": [
                    {
                        "id": g.id,
                        "arrival_minute": g.arrival_time,
                        "group_size": g.size,
                        "zone_preference": g.zone_preference,
                        "alternatives": g.alternative_zones,
                        "patience_minutes": g.patience_minutes,
                        "dining_duration": g.expected_dining_minutes,
                        "special_requirements": g.special_requirements,
                        "willing_to_wait": g.willing_to_wait
                    }
                    for g in groups
                ]
            }

            dataset.append(scenario_dict)

        if output_file:
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            print(f"Dataset saved to {output_file}")

        return dataset

    def load_scenario_from_dict(self, scenario_dict: Dict) -> List[CustomerGroup]:
        """Load a scenario from a dictionary.

        Args:
            scenario_dict: Dictionary containing scenario data

        Returns:
            List of CustomerGroup objects
        """
        groups = []

        for customer_data in scenario_dict["customers"]:
            group = CustomerGroup(
                id=customer_data["id"],
                size=customer_data["group_size"],
                arrival_time=customer_data["arrival_minute"],
                zone_preference=customer_data["zone_preference"],
                alternative_zones=customer_data["alternatives"],
                patience_minutes=customer_data["patience_minutes"],
                expected_dining_minutes=customer_data["dining_duration"],
                special_requirements=customer_data.get("special_requirements", []),
                willing_to_wait=customer_data.get("willing_to_wait", True)
            )
            groups.append(group)

        return sorted(groups, key=lambda g: g.arrival_time)


# Convenience function
def generate_quick_scenario(num_customers: int = 20, seed: int = None) -> List[CustomerGroup]:
    """Quickly generate a scenario with specified number of customers.

    Args:
        num_customers: Approximate number of customers
        seed: Random seed

    Returns:
        List of CustomerGroup objects
    """
    generator = CustomerDataGenerator(seed=seed)

    # Estimate duration based on desired number of customers
    # Assume ~6 customers per hour on average
    duration = int(num_customers / 6 * 60)

    return generator.generate_scenario(duration_minutes=duration, scenario_type="normal")
