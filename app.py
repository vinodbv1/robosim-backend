from flask import Flask, request, jsonify, send_file, Response
from flask_cors import CORS
import os
import json
import time
import base64
from io import BytesIO
from config_manager import ConfigManager
from simulation_runner import SimulationRunner

app = Flask(__name__)
CORS(app)

MAPS_DIR = os.path.join(os.path.dirname(__file__), 'maps')
CONFIG_FILE = os.path.join(os.path.dirname(__file__), 'config', 'simulation_config.yaml')

config_manager = ConfigManager(CONFIG_FILE)
simulation_runner = None

@app.route('/api/get-map', methods=['GET'])
def get_map():
    """Retrieve a map image by name"""
    map_name = request.args.get('map_name')
    if not map_name:
        return jsonify({'error': 'Map name required'}), 400
    
    map_path = os.path.join(MAPS_DIR, map_name)
    if not os.path.exists(map_path):
        return jsonify({'error': f'Map not found: {map_name}'}), 404
    
    return send_file(map_path, mimetype='image/png')

@app.route('/api/start-simulation', methods=['POST'])
def start_simulation():
    """Start robot simulation with provided parameters"""
    global simulation_runner
    
    data = request.json
    map_name = data.get('map_name')
    robot_count = data.get('robot_count', 1)
    robot_position = data.get('robot_position')
    survivors = data.get('survivors', [])
    
    print(f"Starting simulation: {map_name}, {robot_count} robots, {len(survivors)} survivors")
    
    # Update configuration
    config_manager.update_config(
        map_name=map_name,
        robot_count=robot_count,
        robot_position=robot_position,
        survivor_positions=survivors
    )
    
    # Initialize simulation runner
    map_path = os.path.join(MAPS_DIR, map_name)
    simulation_runner = SimulationRunner(CONFIG_FILE, map_path)
    
    def generate_frames():
        """Generator for Server-Sent Events stream"""
        try:
            for frame_data in simulation_runner.run():
                if frame_data['status'] == 'completed':
                    yield f"data: {json.dumps({'status': 'completed'})}\n\n"
                    break
                elif frame_data['status'] == 'error':
                    yield f"data: {json.dumps({'error': frame_data.get('message', 'Unknown error')})}\n\n"
                    break
                else:
                    # frame_data['frame'] is already image bytes
                    image_base64 = base64.b64encode(frame_data['frame']).decode()
                    
                    yield f"data: {json.dumps({'frame': image_base64})}\n\n"
                    time.sleep(0.05)  # Control frame rate (~20 FPS)
        except Exception as e:
            print(f"Error in simulation: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            if simulation_runner:
                simulation_runner.cleanup()
    
    return Response(generate_frames(), mimetype='text/event-stream')

@app.route('/api/pause-simulation', methods=['POST'])
def pause_simulation():
    """Pause or resume the current simulation"""
    global simulation_runner
    data = request.json
    paused = data.get('paused', True)
    
    if simulation_runner:
        simulation_runner.set_paused(paused)
        return jsonify({'status': 'paused' if paused else 'resumed'})
    
    return jsonify({'error': 'No active simulation'}), 400

@app.route('/api/stop-simulation', methods=['POST'])
def stop_simulation():
    """Stop the current simulation"""
    global simulation_runner
    
    if simulation_runner:
        simulation_runner.stop()
        simulation_runner.cleanup()
        simulation_runner = None
        return jsonify({'status': 'stopped'})
    
    return jsonify({'error': 'No active simulation'}), 400

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'service': 'robot-simulation-backend'})

if __name__ == '__main__':
    # Ensure directories exist
    os.makedirs(MAPS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    
    print("=" * 50)
    print("Robot Simulation Backend Starting...")
    print(f"Maps directory: {MAPS_DIR}")
    print(f"Config file: {CONFIG_FILE}")
    print("Server running on http://localhost:5000")
    print("=" * 50)
    
    app.run(debug=True, port=5000, threaded=True)
