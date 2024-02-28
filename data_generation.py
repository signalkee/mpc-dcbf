import numpy as np
import pandas as pd
from mpc_cbf import MPC
import config

# Ensure controller is set for MPC-DC operation
config.controller = "MPC-DC"

# Simulate for defined scenario
config.scenario = 1


def simulate_robot_movement(initial_position):
    """
    Simulate robot movement from an initial position using the MPC-DC controller
    and check for collision with the obstacle.
    
    Parameters:
    - initial_position: np.array, initial [x, y, theta] of the robot.
    
    Returns:
    - collision: bool, True if collision occurs, else False.
    """
    controller = MPC()
    controller.x0 = initial_position
    controller.set_init_state()
    
    collision = False
    for _ in range(controller.sim_time):
        u0 = controller.mpc.make_step(controller.x0)
        y_next = controller.simulator.make_step(u0)
        x0 = controller.estimator.make_step(y_next)
        
        if config.scenario == 1: # Single obstacle
            # Check for collision with the obstacle defined in scenario 1
            dist_to_obstacle = np.linalg.norm(np.transpose(x0[:2]) - np.array(config.obs[0][:2]))
            if dist_to_obstacle <= (config.obs[0][2]):
                collision = True
                break    
        elif config.scenario == 3: # Multiple obstacle
            break
        elif config.scenario == 4: # Circular trajectory with obstacle
            break
        elif config.scenario == 6: # Moving multiple obstacle with obstacle
            break
    
    
    return collision


def generate_data(samples=100):
    """
    Generate simulation data with random initial positions and check for collisions using MPC-DC.
    
    Parameters:
    - samples: int, number of samples to generate.
    """
    data = []
    for _ in range(samples):
        initial_position = np.random.uniform(low=[0, 0, 0], high=[1.2, 0.7, 2*np.pi], size=(3,))
        collision = simulate_robot_movement(initial_position)
        data.append(np.hstack((initial_position, collision)))
    
    df = pd.DataFrame(data, columns=['x_start', 'y_start', 'theta_start', 'collision'])
    df.to_csv(f'simulation_data_scenario{config.scenario}_sample{samples}.csv', index=False)



if __name__ == "__main__":
    generate_data(100)
