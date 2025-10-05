# Robot Simulation Backend

Flask backend for multi-robot navigation simulation using ir-sim.

## Quick Start

### 1. Install Dependencies

```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Add Map Images

Create a `maps/` directory and add your map images:

```bash
mkdir maps
# Copy your map files:
# - hm3d_1.png through hm3d_9.png
# - cave.png
```

### 3. Run the Server

```bash
python app.py
```

Server will start at `http://localhost:5000`

## Project Structure

```
backend/
├── app.py                      # Flask API server
├── config_manager.py           # YAML configuration manager
├── simulation_runner.py        # Simulation logic
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── maps/                       # Map images directory
│   ├── hm3d_1.png
│   ├── hm3d_2.png
│   └── ...
└── config/                     # Generated at runtime
    └── simulation_config.yaml  # Simulation parameters
```

## API Endpoints

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "service": "robot-simulation-backend"
}
```

### GET /api/get-map
Retrieve a map image by name.

**Query Parameters:**
- `map_name` (required): Name of the map file (e.g., "hm3d_1.png")

**Response:**
- PNG image file

**Example:**
```bash
curl "http://localhost:5000/api/get-map?map_name=hm3d_1.png" --output map.png
```

### POST /api/start-simulation
Start robot simulation with provided parameters.

**Request Body:**
```json
{
  "map_name": "hm3d_1.png",
  "robot_count": 3,
  "robot_position": {
    "x": 400,
    "y": 300
  },
  "survivors": [
    {"x": 100, "y": 150},
    {"x": 200, "y": 250},
    {"x": 300, "y": 350}
  ]
}
```

**Response:**
Server-Sent Events (SSE) stream with simulation frames:
```
data: {"frame": "base64_encoded_image_data"}
data: {"frame": "base64_encoded_image_data"}
...
data: {"status": "completed"}
```

**Example:**
```bash
curl -X POST http://localhost:5000/api/start-simulation \
  -H "Content-Type: application/json" \
  -d '{"map_name":"hm3d_1.png","robot_count":1,"robot_position":{"x":400,"y":300},"survivors":[]}'
```

### POST /api/pause-simulation
Pause or resume the current simulation.

**Request Body:**
```json
{
  "paused": true
}
```

**Response:**
```json
{
  "status": "paused"
}
```

### POST /api/stop-simulation
Stop the current simulation.

**Response:**
```json
{
  "status": "stopped"
}
```

## Configuration

The simulation is configured via `config/simulation_config.yaml`, which is automatically generated and updated. Example structure:

```yaml
world:
  width: 800
  height: 600
  map: hm3d_1.png
  obstacles:
    - type: circle
      position:
        x: 100
        y: 150
      radius: 10.0
      label: survivor_0

robots:
  - id: robot_0
    type: diff_drive
    start_position:
      x: 400
      y: 300
      theta: 0.0
    goals:
      - x: 100
        y: 150
      - x: 200
        y: 250
    max_speed: 50.0
    max_angular_speed: 2.0
    sensor_range: 100.0
    radius: 15.0
    color: '#00d9ff'

simulation:
  max_steps: 1000
  time_step: 0.1
  goal_threshold: 15.0
```

## Adding Custom Attributes

The `ConfigManager` class supports adding custom attributes to robots and obstacles:

```python
from config_manager import ConfigManager

config_manager = ConfigManager('config/simulation_config.yaml')

# Add custom attribute to robot
config_manager.add_robot_attribute('robot_0', 'battery_level', 100)

# Add custom attribute to obstacle
config_manager.add_obstacle_attribute(0, 'survivor_health', 'critical')
```

## Integrating IR-SIM

The current `simulation_runner.py` contains a simplified implementation. To integrate the actual ir-sim library:

1. Ensure ir-sim is installed:
```bash
pip install git+https://github.com/hanruihua/ir-sim.git
```

2. Replace the simulation logic in `simulation_runner.py` with ir-sim API calls:
```python
from ir_sim import World, Robot

class SimulationRunner:
    def __init__(self, config_path, map_path):
        # Initialize ir-sim World
        self.world = World(config_path)
        self.world.load_map(map_path)
    
    def run(self):
        for step in range(max_steps):
            # Step simulation
            done = self.world.step()
            
            # Render frame
            fig = self.world.render()
            yield {'frame': fig, 'status': 'running'}
            
            if done:
                yield {'status': 'completed'}
                break
```

## Troubleshooting

### CORS Errors
- Ensure `flask-cors` is installed: `pip install flask-cors`
- CORS is configured to allow all origins in development

### Map Not Loading
- Verify map files exist in `backend/maps/` directory
- Check file names match exactly (case-sensitive)
- Ensure files are valid PNG images

### Connection Refused
- Verify Flask server is running: `python app.py`
- Check server is listening on port 5000
- Ensure no firewall is blocking the connection

### Simulation Not Starting
- Check console output for errors
- Verify ir-sim is properly installed
- Check YAML configuration is valid

## Development

To run in development mode with auto-reload:

```bash
export FLASK_ENV=development  # On Windows: set FLASK_ENV=development
python app.py
```

## Production Deployment

For production, use a WSGI server like Gunicorn:

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## License

This backend is part of the Robot Simulation Control system.
