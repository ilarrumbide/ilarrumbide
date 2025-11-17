"""Utility functions for the restaurant RL host system."""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Tuple
import math


def calculate_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """Calculate Euclidean distance between two coordinates."""
    return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2)


def encode_time_cyclical(hour: int) -> Tuple[float, float]:
    """Encode time as sin/cos for cyclical representation."""
    hour_normalized = hour / 24.0
    return (
        math.sin(2 * math.pi * hour_normalized),
        math.cos(2 * math.pi * hour_normalized)
    )


def calculate_table_efficiency(table_capacity: int, group_size: int) -> float:
    """Calculate how efficiently a table is being used."""
    if table_capacity == 0:
        return 0.0
    return group_size / table_capacity


def get_zone_for_coordinates(x: int, y: int) -> str:
    """Determine zone based on coordinates."""
    if y == 0:
        if x <= 1:
            return "window"
        elif x == 4:
            return "bar"
        else:
            return "inside"
    elif y == 1:
        return "outside"
    return "inside"


def calculate_mood_score(wait_time: float, patience: float) -> float:
    """Calculate customer mood score (0-1, where 1 is happy)."""
    if patience == 0:
        return 0.0
    ratio = wait_time / patience
    if ratio <= 0.5:
        return 1.0
    elif ratio <= 1.0:
        return 1.0 - (ratio - 0.5) * 2
    else:
        return max(0.0, 1.0 - ratio)


def get_mood_emoji(mood_score: float) -> str:
    """Get emoji representation of mood."""
    if mood_score >= 0.8:
        return "😊"
    elif mood_score >= 0.6:
        return "🙂"
    elif mood_score >= 0.4:
        return "😐"
    elif mood_score >= 0.2:
        return "😟"
    else:
        return "😠"


def weighted_random_choice(choices: List, weights: List) -> any:
    """Make a weighted random choice."""
    return np.random.choice(choices, p=np.array(weights) / sum(weights))


def poisson_arrivals(rate: float, duration_minutes: int) -> List[int]:
    """Generate customer arrival times using Poisson process."""
    arrivals = []
    current_time = 0

    while current_time < duration_minutes:
        # Time until next arrival (exponential distribution)
        inter_arrival = np.random.exponential(1.0 / rate)
        current_time += inter_arrival

        if current_time < duration_minutes:
            arrivals.append(int(current_time))

    return sorted(arrivals)


def get_rush_hour_multiplier(hour: int) -> float:
    """Get arrival rate multiplier based on time of day."""
    # Breakfast rush (7-10am)
    if 7 <= hour < 10:
        return 1.5
    # Lunch rush (12-2pm)
    elif 12 <= hour < 14:
        return 2.0
    # Dinner rush (6-10pm)
    elif 18 <= hour < 22:
        return 2.5
    # Off-peak
    else:
        return 0.5


def format_time_minutes(minutes: float) -> str:
    """Format minutes as human-readable time."""
    hours = int(minutes // 60)
    mins = int(minutes % 60)
    return f"{hours}h {mins}m" if hours > 0 else f"{mins}m"


def normalize_array(arr: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Normalize array to [min_val, max_val] range."""
    arr_min = arr.min()
    arr_max = arr.max()

    if arr_max - arr_min == 0:
        return np.full_like(arr, (min_val + max_val) / 2, dtype=float)

    normalized = (arr - arr_min) / (arr_max - arr_min)
    return normalized * (max_val - min_val) + min_val
