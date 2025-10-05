import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import irsim
import os
import time
from io import BytesIO

class SimulationRunner:
    """
    Handles running the IR-SIM simulation using the ir-sim library
    """
    
    def __init__(self, config_path, map_path):
        self.config_path = config_path
        self.map_path = map_path
        self.paused = False
        self.stopped = False
        self.env = None
        self.frame_dir = os.path.join(os.path.dirname(__file__), 'animation_buffer')
        
        # Create frames directory
        os.makedirs(self.frame_dir, exist_ok=True)
        
        # Clean up old frames
        for file in os.listdir(self.frame_dir):
            if file.endswith('.png'):
                os.remove(os.path.join(self.frame_dir, file))
        
        # Load map image for background overlay
        if os.path.exists(map_path):
            self.map_image = Image.open(map_path)
        else:
            self.map_image = None
            print(f"Warning: Map not found at {map_path}")
        
        # Initialize ir-sim environment
        try:
            self.env = irsim.make(config_path, save_ani=True, display=False, full=True)
            print(f"Initialized ir-sim environment from {config_path}")
        except Exception as e:
            print(f"Error initializing ir-sim: {e}")
            raise
    
    def run(self):
        """
        Run ir-sim simulation and yield frames
        
        Yields:
            dict: Frame data with matplotlib figure or status
        """
        if not self.env:
            yield {'status': 'error', 'message': 'Environment not initialized'}
            return
        
        step = 0
        max_steps = 3000  # Maximum simulation steps
        
        while step < max_steps and not self.stopped:
            if self.paused:
                time.sleep(0.1)
                continue
            
            try:
                # Step the ir-sim environment (this saves frame automatically)
                self.env.step()

                self.env.render()
                
                # Read the saved frame from ir-sim
                frame_path = os.path.join(self.frame_dir, f'simulation_config_{step+1:04d}.png')
                
                # Wait a bit for file to be written
                time.sleep(0.02)
                
                # Read the frame if it exists
                if os.path.exists(frame_path):
                    with open(frame_path, 'rb') as f:
                        frame_bytes = f.read()
                    yield {'frame': frame_bytes, 'status': 'running'}
                    print("Simuilation running")
                else:
                    print(f"Warning: Frame not found at {frame_path}")
                
                # Check if simulation is done
                if self.env.done():
                    print("Simulation completed: All robots reached their goals")
                    yield {'status': 'completed'}
                    break
                
                step += 1
                time.sleep(0.05)  # Control frame rate
                
            except Exception as e:
                print(f"Error during simulation step: {e}")
                import traceback
                traceback.print_exc()
                yield {'status': 'error', 'message': str(e)}
                break
        
        if step >= max_steps:
            print("Simulation completed: Max steps reached")
            yield {'status': 'completed'}
    
    
    def set_paused(self, paused):
        """Pause or resume simulation"""
        self.paused = paused
        print(f"Simulation {'paused' if paused else 'resumed'}")
    
    def stop(self):
        """Stop simulation"""
        self.stopped = True
        print("Simulation stopped")
    
    def cleanup(self):
        """Clean up resources"""
        # if self.env:
        #     self.env.end()
        # plt.close('all')

    def cleanup_frames(self):
        print("Cleaning up frames............................")
        # Clean up frame files
        if os.path.exists(self.frame_dir):
            for file in os.listdir(self.frame_dir):
                if file.endswith('.png'):
                    try:
                        os.remove(os.path.join(self.frame_dir, file))
                    except:
                        pass
