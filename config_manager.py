import yaml
import os

class ConfigManager:
    """Manages YAML configuration for robot simulation"""

    global canvas_height
    global canvas_width
    global map_height
    global map_width
    global px_to_m

    # Convert pixel coordinates to meters (assuming 800x600 pixel map = 8x6 meters)
    px_to_m = 100  # 100 pixels = 1 meter
    canvas_height = 1000
    canvas_width = 1000

    map_height = canvas_height / px_to_m # 6
    map_width= canvas_width / px_to_m # 8
    


    
    def __init__(self, config_path):
        self.config_path = config_path
        self.ensure_config_exists()
    
    def ensure_config_exists(self):
        """Create default config if it doesn't exist"""
        if not os.path.exists(self.config_path):
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            default_config = self._create_default_config()
            self.save_config(default_config)
            print(f"Created default config at {self.config_path}")
    
    def _create_default_config(self):
        """Create default configuration structure for ir-sim"""
        return {
            'world': {
                'height': map_height,  # world height in meters
                'width': map_width,   # world width in meters
                'step_time': 0.1,  # 10Hz calculate each step
                'sample_time': 0.1,  # 10Hz for render and data extraction
                'offset': [0, 0],  # offset of the world on x and y
                'collision_mode': 'stop'  # 'stop', 'unobstructed', 'unobstructed_obstacles'
            },
            'robot': [],
            'obstacle': []
        }
    
    def load_config(self):
        """Load configuration from YAML file"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config):
        """Save configuration to YAML file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    def update_config(self, map_name, robot_count, robot_position, survivor_positions):
        """
        Update configuration with simulation parameters in ir-sim format
        
        Args:
            map_name: Name of the map file
            robot_count: Number of robots
            robot_position: Dict with x, y coordinates for robot start (in pixels)
            survivor_positions: List of dicts with x, y coordinates for survivors (in pixels)
        """
        config = self.load_config()
        
        # Update world settings with save_ani to save frames
        config['world'] = {
            'height': map_height,  # 600 pixels = 6 meters
            'width': map_width,   # 800 pixels = 8 meters
            'step_time': 0.1,
            'sample_time': 0.1,
            'offset': [0, 0],
            'collision_mode': 'stop',
            'obstacle_map': map_name,
            'save_ani': True  # Enable animation frame saving
        }
        
        # Create robot configurations in ir-sim format
        robots = []
        for i in range(robot_count):
            # Convert pixel coordinates to meters (invert Y since canvas Y is top-down)
            start_x = float(robot_position['x']) / px_to_m
            start_y = (canvas_height - float(robot_position['y'])) / px_to_m  # Invert Y axis
            
            # For multiple survivors, use first as goal (ir-sim supports single goal per robot)
            # To visit all survivors, we'd need path planning or behavior modification
            goal_x = float(survivor_positions[0]['x']) / px_to_m if survivor_positions else start_x
            goal_y = (canvas_height - float(survivor_positions[0]['y'])) / px_to_m if survivor_positions else start_y  # Invert Y axis
            
            robot = {
                'kinematics': {'name': 'diff'},  # differential drive
                'shape': {'name': 'circle', 'radius': 0.15},  # 15 pixels = 0.15 meters
                'state': [start_x+i*0.1, start_y+i*0.1, 0],  # [x, y, theta]
                'goal': [goal_x, goal_y, 0],  # [x, y, theta]
                'behavior': {'name': 'dash'},  # move directly toward goal
                'color': self._get_robot_color(i)
            }
            robots.append(robot)
        
        config['robot'] = robots
        
        # Add survivors as obstacles in ir-sim format
        obstacles = []
        for i, pos in enumerate(survivor_positions):
            obstacle = {
                'shape': {'name': 'circle', 'radius': 0.1},  # 10 pixels = 0.1 meters
                'state': [float(pos['x']) / px_to_m, (canvas_height - float(pos['y'])) / px_to_m, 0]  # Invert Y axis
            }
            obstacles.append(obstacle)
        
        config['obstacle'] = obstacles
        
        # Save updated config
        self.save_config(config)
        
        print(f"Configuration updated: {robot_count} robots, {len(survivor_positions)} survivors")
        
        return config
    
    def _get_robot_color(self, index):
        """Get color for robot based on index"""
        colors = [
            '#00d9ff',  # cyan
            '#00ff88',  # green
            '#ff6b00',  # orange
            '#ff00ff',  # magenta
            '#ffff00',  # yellow
            '#00ffff',  # cyan
            '#ff0088',  # pink
            '#88ff00',  # lime
            '#0088ff',  # blue
            '#ff8800'   # orange
        ]
        return colors[index % len(colors)]
    
    def add_robot_attribute(self, robot_index, attribute_name, attribute_value):
        """
        Add or update a custom attribute for a specific robot
        
        Args:
            robot_index: Index of the robot
            attribute_name: Name of the attribute
            attribute_value: Value of the attribute
        """
        config = self.load_config()
        
        if robot_index < len(config['robot']):
            config['robot'][robot_index][attribute_name] = attribute_value
            self.save_config(config)
    
    def add_obstacle_attribute(self, obstacle_index, attribute_name, attribute_value):
        """
        Add or update a custom attribute for a specific obstacle
        
        Args:
            obstacle_index: Index of the obstacle
            attribute_name: Name of the attribute
            attribute_value: Value of the attribute
        """
        config = self.load_config()
        
        if obstacle_index < len(config['obstacle']):
            config['obstacle'][obstacle_index][attribute_name] = attribute_value
            self.save_config(config)
