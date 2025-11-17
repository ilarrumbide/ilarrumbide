"""FastAPI server for Restaurant RL Host with WebSocket support."""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import asyncio
import json
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.models.restaurant import Restaurant, Table, CustomerGroup, TableStatus
from backend.models.rl_environment import RestaurantEnv
from backend.models.rl_agent import RestaurantRLAgent
from backend.training.data_generator import CustomerDataGenerator, generate_quick_scenario
from backend.utils.config import config
from backend.utils.helpers import get_zone_for_coordinates

# Initialize FastAPI app
app = FastAPI(title="Restaurant RL Host", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
restaurant_state = {
    "restaurant": None,
    "env": None,
    "agent": None,
    "simulation_running": False,
    "current_scenario": [],
    "current_time": 0.0,
    "speed_multiplier": 1.0
}

# WebSocket connections
active_connections: List[WebSocket] = []


# Pydantic models for API
class CustomerGroupCreate(BaseModel):
    size: int
    zone_preference: str
    alternative_zones: List[str] = []
    patience_minutes: float = 20.0
    expected_dining_minutes: float = 60.0
    special_requirements: List[str] = []


class SeatCustomerRequest(BaseModel):
    group_id: int
    table_ids: List[int]


class SimulationConfig(BaseModel):
    duration_minutes: int = 480
    scenario_type: str = "normal"
    speed_multiplier: float = 1.0


# Helper functions
def create_default_tables() -> List[Table]:
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


def initialize_restaurant():
    """Initialize restaurant and environment."""
    tables = create_default_tables()
    restaurant = Restaurant(tables)
    env = RestaurantEnv(tables)

    # Try to load trained agent
    model_path = Path("saved_models/best_host.zip")
    agent = None

    if model_path.exists():
        try:
            agent = RestaurantRLAgent.load_agent(str(model_path), env)
            print("Loaded trained agent successfully")
        except Exception as e:
            print(f"Could not load trained agent: {e}")
            agent = None
    else:
        print("No trained model found. AI decisions will be unavailable.")

    restaurant_state["restaurant"] = restaurant
    restaurant_state["env"] = env
    restaurant_state["agent"] = agent
    restaurant_state["current_time"] = 0.0


# WebSocket connection manager
async def broadcast_message(message: Dict):
    """Broadcast message to all connected clients."""
    disconnected = []
    for connection in active_connections:
        try:
            await connection.send_json(message)
        except Exception:
            disconnected.append(connection)

    # Remove disconnected clients
    for conn in disconnected:
        if conn in active_connections:
            active_connections.remove(conn)


# API Endpoints
@app.on_event("startup")
async def startup_event():
    """Initialize on startup."""
    initialize_restaurant()
    print("Restaurant RL Host server started!")


@app.get("/")
async def root():
    """Serve the main page."""
    return FileResponse("frontend/index.html")


@app.post("/api/reset")
async def reset_restaurant():
    """Reset the restaurant to initial state."""
    if restaurant_state["restaurant"]:
        restaurant_state["restaurant"].reset()
        restaurant_state["current_time"] = 0.0
        restaurant_state["current_scenario"] = []
        restaurant_state["simulation_running"] = False

        await broadcast_message({
            "type": "restaurant_reset",
            "state": restaurant_state["restaurant"].get_state_dict()
        })

        return {"status": "success", "message": "Restaurant reset"}

    raise HTTPException(status_code=500, detail="Restaurant not initialized")


@app.get("/api/state")
async def get_restaurant_state():
    """Get current restaurant state."""
    if restaurant_state["restaurant"]:
        return {
            "status": "success",
            "state": restaurant_state["restaurant"].get_state_dict(),
            "current_time": restaurant_state["current_time"],
            "simulation_running": restaurant_state["simulation_running"]
        }

    raise HTTPException(status_code=500, detail="Restaurant not initialized")


@app.post("/api/customer/add")
async def add_customer(customer: CustomerGroupCreate):
    """Manually add a customer to the queue."""
    restaurant = restaurant_state["restaurant"]

    if not restaurant:
        raise HTTPException(status_code=500, detail="Restaurant not initialized")

    # Create customer group
    group = CustomerGroup(
        id=restaurant.next_group_id,
        size=customer.size,
        arrival_time=restaurant_state["current_time"],
        zone_preference=customer.zone_preference,
        alternative_zones=customer.alternative_zones,
        patience_minutes=customer.patience_minutes,
        expected_dining_minutes=customer.expected_dining_minutes,
        special_requirements=customer.special_requirements
    )

    restaurant.next_group_id += 1
    restaurant.add_to_queue(group)

    await broadcast_message({
        "type": "customer_arrived",
        "group": group.to_dict()
    })

    return {"status": "success", "group_id": group.id}


@app.post("/api/customer/seat")
async def seat_customer(request: SeatCustomerRequest):
    """Manually seat a customer."""
    restaurant = restaurant_state["restaurant"]

    if not restaurant:
        raise HTTPException(status_code=500, detail="Restaurant not initialized")

    # Find the group
    group = None
    for g in restaurant.waiting_queue:
        if g.id == request.group_id:
            group = g
            break

    if not group:
        raise HTTPException(status_code=404, detail="Customer group not found")

    # Attempt to seat
    success = restaurant.seat_group(group, request.table_ids)

    if success:
        await broadcast_message({
            "type": "customer_seated",
            "group_id": group.id,
            "table_ids": request.table_ids
        })

        return {"status": "success", "message": "Customer seated"}
    else:
        raise HTTPException(status_code=400, detail="Could not seat customer at specified tables")


@app.get("/api/ai/decision")
async def get_ai_decision():
    """Get AI recommendation for next action."""
    restaurant = restaurant_state["restaurant"]
    env = restaurant_state["env"]
    agent = restaurant_state["agent"]

    if not agent:
        raise HTTPException(status_code=503, detail="AI agent not available")

    if not restaurant.waiting_queue:
        return {
            "status": "success",
            "decision": "No customers waiting",
            "action": None
        }

    # Get current observation
    obs = env._get_observation()

    # Get AI prediction
    action = agent.predict(obs, deterministic=True)

    # Get explanation
    explanation = agent.get_action_explanation(int(action))

    return {
        "status": "success",
        "decision": explanation,
        "action": int(action),
        "current_group": restaurant.waiting_queue[0].to_dict()
    }


@app.post("/api/ai/execute")
async def execute_ai_decision():
    """Execute AI's recommended action."""
    restaurant = restaurant_state["restaurant"]
    env = restaurant_state["env"]
    agent = restaurant_state["agent"]

    if not agent:
        raise HTTPException(status_code=503, detail="AI agent not available")

    if not restaurant.waiting_queue:
        return {"status": "success", "message": "No customers waiting"}

    # Get and execute action
    obs = env._get_observation()
    action = agent.predict(obs, deterministic=True)

    # Execute through environment
    _, reward, _, _, info = env.step(int(action))

    # Broadcast update
    await broadcast_message({
        "type": "ai_action_executed",
        "action": agent.get_action_explanation(int(action)),
        "reward": float(reward),
        "info": info
    })

    return {
        "status": "success",
        "action": agent.get_action_explanation(int(action)),
        "reward": float(reward)
    }


@app.post("/api/simulation/start")
async def start_simulation(config: SimulationConfig):
    """Start automated simulation."""
    restaurant_state["simulation_running"] = True
    restaurant_state["speed_multiplier"] = config.speed_multiplier

    # Generate scenario
    generator = CustomerDataGenerator()
    scenario = generator.generate_scenario(
        duration_minutes=config.duration_minutes,
        scenario_type=config.scenario_type
    )

    restaurant_state["current_scenario"] = scenario

    # Reset restaurant
    restaurant_state["restaurant"].reset()
    restaurant_state["current_time"] = 0.0

    if restaurant_state["env"]:
        restaurant_state["env"].set_customer_schedule(scenario)

    await broadcast_message({
        "type": "simulation_started",
        "duration": config.duration_minutes,
        "scenario_type": config.scenario_type,
        "num_customers": len(scenario)
    })

    return {
        "status": "success",
        "message": "Simulation started",
        "num_customers": len(scenario)
    }


@app.post("/api/simulation/stop")
async def stop_simulation():
    """Stop the simulation."""
    restaurant_state["simulation_running"] = False

    await broadcast_message({
        "type": "simulation_stopped"
    })

    return {"status": "success", "message": "Simulation stopped"}


# WebSocket endpoint
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for real-time updates."""
    await websocket.accept()
    active_connections.append(websocket)

    try:
        # Send initial state
        if restaurant_state["restaurant"]:
            await websocket.send_json({
                "type": "initial_state",
                "state": restaurant_state["restaurant"].get_state_dict()
            })

        # Keep connection alive and handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            # Handle different message types
            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

            elif message.get("type") == "request_state":
                if restaurant_state["restaurant"]:
                    await websocket.send_json({
                        "type": "state_update",
                        "state": restaurant_state["restaurant"].get_state_dict(),
                        "current_time": restaurant_state["current_time"]
                    })

    except WebSocketDisconnect:
        active_connections.remove(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        if websocket in active_connections:
            active_connections.remove(websocket)


# Simulation loop (called periodically)
async def simulation_loop():
    """Main simulation loop."""
    while True:
        if restaurant_state["simulation_running"]:
            restaurant = restaurant_state["restaurant"]
            agent = restaurant_state["agent"]

            if restaurant and agent:
                # Advance time
                restaurant_state["current_time"] += 1.0 * restaurant_state["speed_multiplier"]

                # Add arrivals
                for group in restaurant_state["current_scenario"]:
                    if (group.arrival_time <= restaurant_state["current_time"] and
                        group not in restaurant.waiting_queue and
                        group.id not in restaurant.seated_groups):

                        restaurant.add_to_queue(group)

                        await broadcast_message({
                            "type": "customer_arrived",
                            "group": group.to_dict(),
                            "current_time": restaurant_state["current_time"]
                        })

                # Update restaurant
                restaurant.update(restaurant_state["current_time"])

                # AI makes decisions if there are customers waiting
                if restaurant.waiting_queue and agent:
                    obs = restaurant_state["env"]._get_observation()
                    action = agent.predict(obs, deterministic=True)

                    # Execute action
                    _, reward, _, _, info = restaurant_state["env"].step(int(action))

                    if info.get("action_success"):
                        await broadcast_message({
                            "type": "state_update",
                            "state": restaurant.get_state_dict(),
                            "current_time": restaurant_state["current_time"],
                            "last_action": agent.get_action_explanation(int(action))
                        })

        await asyncio.sleep(1.0 / restaurant_state["speed_multiplier"])


# Start simulation loop on startup
@app.on_event("startup")
async def start_simulation_loop():
    """Start the simulation loop."""
    asyncio.create_task(simulation_loop())


# Serve static files
app.mount("/static", StaticFiles(directory="frontend"), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
