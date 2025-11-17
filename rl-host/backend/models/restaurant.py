"""Restaurant domain models: Table, CustomerGroup, Restaurant."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict
from enum import Enum
from datetime import datetime
import numpy as np

from backend.utils.helpers import calculate_distance, get_zone_for_coordinates, calculate_mood_score


class TableStatus(Enum):
    """Table status enumeration."""
    AVAILABLE = "available"
    OCCUPIED = "occupied"
    CLEANING = "cleaning"
    COMBINED = "combined"
    MAINTENANCE = "maintenance"


class CustomerMood(Enum):
    """Customer mood states."""
    HAPPY = "happy"
    CONTENT = "content"
    NEUTRAL = "neutral"
    IMPATIENT = "impatient"
    ANGRY = "angry"
    LEFT = "left"


@dataclass
class Table:
    """Represents a restaurant table."""
    id: int
    capacity: int
    coordinates: Tuple[float, float]
    zone: str
    status: TableStatus = TableStatus.AVAILABLE
    current_group_id: Optional[int] = None
    occupied_since: Optional[float] = None
    combined_with: List[int] = field(default_factory=list)

    @property
    def is_available(self) -> bool:
        """Check if table is available for seating."""
        return self.status == TableStatus.AVAILABLE

    @property
    def effective_capacity(self) -> int:
        """Get effective capacity including combined tables."""
        return self.capacity

    def occupy(self, group_id: int, time: float):
        """Mark table as occupied."""
        self.status = TableStatus.OCCUPIED
        self.current_group_id = group_id
        self.occupied_since = time

    def free(self):
        """Free the table."""
        self.status = TableStatus.AVAILABLE
        self.current_group_id = None
        self.occupied_since = None
        self.combined_with = []

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "capacity": self.capacity,
            "coordinates": self.coordinates,
            "zone": self.zone,
            "status": self.status.value,
            "current_group_id": self.current_group_id,
            "occupied_since": self.occupied_since,
            "combined_with": self.combined_with
        }


@dataclass
class CustomerGroup:
    """Represents a group of customers."""
    id: int
    size: int
    arrival_time: float
    zone_preference: str
    alternative_zones: List[str]
    patience_minutes: float
    expected_dining_minutes: float
    special_requirements: List[str] = field(default_factory=list)
    willing_to_wait: bool = True
    mood: CustomerMood = CustomerMood.HAPPY
    wait_start_time: Optional[float] = None
    seated_time: Optional[float] = None
    table_ids: List[int] = field(default_factory=list)
    departure_time: Optional[float] = None

    @property
    def current_wait_time(self) -> float:
        """Get current wait time in minutes."""
        if self.wait_start_time is None:
            return 0.0
        if self.seated_time is not None:
            return self.seated_time - self.wait_start_time
        return 0.0  # Will be calculated with current time externally

    @property
    def mood_score(self) -> float:
        """Get mood score (0-1)."""
        wait_time = self.current_wait_time
        return calculate_mood_score(wait_time, self.patience_minutes)

    def update_mood(self, current_time: float):
        """Update customer mood based on wait time."""
        if self.seated_time is not None:
            return  # Already seated

        wait_time = current_time - self.arrival_time
        mood_score = calculate_mood_score(wait_time, self.patience_minutes)

        if mood_score >= 0.8:
            self.mood = CustomerMood.HAPPY
        elif mood_score >= 0.6:
            self.mood = CustomerMood.CONTENT
        elif mood_score >= 0.4:
            self.mood = CustomerMood.NEUTRAL
        elif mood_score >= 0.2:
            self.mood = CustomerMood.IMPATIENT
        else:
            if wait_time > self.patience_minutes:
                self.mood = CustomerMood.LEFT
            else:
                self.mood = CustomerMood.ANGRY

    def seat(self, table_ids: List[int], current_time: float):
        """Seat the customer group."""
        self.table_ids = table_ids
        self.seated_time = current_time
        self.departure_time = current_time + self.expected_dining_minutes

    def has_left(self, current_time: float) -> bool:
        """Check if customer has left due to impatience."""
        if self.seated_time is not None:
            return False
        wait_time = current_time - self.arrival_time
        return wait_time > self.patience_minutes

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "size": self.size,
            "arrival_time": self.arrival_time,
            "zone_preference": self.zone_preference,
            "alternative_zones": self.alternative_zones,
            "patience_minutes": self.patience_minutes,
            "expected_dining_minutes": self.expected_dining_minutes,
            "special_requirements": self.special_requirements,
            "willing_to_wait": self.willing_to_wait,
            "mood": self.mood.value,
            "wait_start_time": self.wait_start_time,
            "seated_time": self.seated_time,
            "table_ids": self.table_ids,
            "departure_time": self.departure_time
        }


class Restaurant:
    """Represents the restaurant with tables and customer management."""

    def __init__(self, tables: List[Table]):
        """Initialize restaurant with tables."""
        self.tables = {table.id: table for table in tables}
        self.waiting_queue: List[CustomerGroup] = []
        self.seated_groups: Dict[int, CustomerGroup] = {}
        self.departed_groups: List[CustomerGroup] = []
        self.current_time: float = 0.0
        self.next_group_id: int = 1

        # Statistics
        self.total_customers_served: int = 0
        self.total_customers_lost: int = 0
        self.total_wait_time: float = 0.0
        self.total_dining_time: float = 0.0

    def add_to_queue(self, group: CustomerGroup):
        """Add customer group to waiting queue."""
        group.wait_start_time = self.current_time
        self.waiting_queue.append(group)

    def seat_group(self, group: CustomerGroup, table_ids: List[int]) -> bool:
        """Seat a customer group at specified tables."""
        # Verify tables are available and capacity is sufficient
        total_capacity = 0
        for table_id in table_ids:
            table = self.tables.get(table_id)
            if not table or not table.is_available:
                return False
            total_capacity += table.capacity

        if total_capacity < group.size:
            return False

        # Seat the group
        group.seat(table_ids, self.current_time)

        # Occupy tables
        for table_id in table_ids:
            self.tables[table_id].occupy(group.id, self.current_time)

        # Move from queue to seated
        if group in self.waiting_queue:
            self.waiting_queue.remove(group)

        self.seated_groups[group.id] = group
        self.total_customers_served += group.size

        return True

    def free_tables(self, group_id: int):
        """Free tables when group departs."""
        group = self.seated_groups.get(group_id)
        if not group:
            return

        for table_id in group.table_ids:
            if table_id in self.tables:
                self.tables[table_id].free()

        # Move to departed
        del self.seated_groups[group_id]
        self.departed_groups.append(group)

    def update(self, current_time: float):
        """Update restaurant state."""
        self.current_time = current_time

        # Update waiting customers mood
        for group in self.waiting_queue[:]:
            group.update_mood(current_time)
            if group.has_left(current_time):
                self.waiting_queue.remove(group)
                self.departed_groups.append(group)
                self.total_customers_lost += group.size

        # Check for groups that should depart
        for group_id, group in list(self.seated_groups.items()):
            if group.departure_time and current_time >= group.departure_time:
                self.free_tables(group_id)

    def get_available_tables(self, zone: str = None) -> List[Table]:
        """Get list of available tables, optionally filtered by zone."""
        available = [t for t in self.tables.values() if t.is_available]
        if zone:
            available = [t for t in available if t.zone == zone]
        return available

    def get_combinable_tables(self, min_capacity: int) -> List[List[int]]:
        """Find combinations of tables that can be combined."""
        available = self.get_available_tables()
        combinations = []

        # Single tables
        for table in available:
            if table.capacity >= min_capacity:
                combinations.append([table.id])

        # Pairs of adjacent tables
        for i, table1 in enumerate(available):
            for table2 in available[i+1:]:
                distance = calculate_distance(table1.coordinates, table2.coordinates)
                if distance <= 1.5:  # Configurable threshold
                    combined_capacity = table1.capacity + table2.capacity
                    if combined_capacity >= min_capacity:
                        combinations.append([table1.id, table2.id])

        return combinations

    def get_state_dict(self) -> Dict:
        """Get current restaurant state as dictionary."""
        return {
            "current_time": self.current_time,
            "tables": [table.to_dict() for table in self.tables.values()],
            "waiting_queue": [group.to_dict() for group in self.waiting_queue],
            "seated_groups": [group.to_dict() for group in self.seated_groups.values()],
            "statistics": {
                "total_served": self.total_customers_served,
                "total_lost": self.total_customers_lost,
                "current_occupancy": len(self.seated_groups),
                "queue_length": len(self.waiting_queue)
            }
        }

    def reset(self):
        """Reset restaurant to initial state."""
        for table in self.tables.values():
            table.free()

        self.waiting_queue = []
        self.seated_groups = {}
        self.departed_groups = []
        self.current_time = 0.0
        self.next_group_id = 1
        self.total_customers_served = 0
        self.total_customers_lost = 0
        self.total_wait_time = 0.0
        self.total_dining_time = 0.0
