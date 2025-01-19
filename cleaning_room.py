import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.animation as animation

class FluidSimulator:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.dt = 0.1
        self.viscosity = 0.1
        self.diffusion_rate = 0.05
        self.incompressibility_iters = 10
        
        # Initialize grids
        self.dirt = np.zeros((grid_size, grid_size))
        self.velocity_x = np.zeros((grid_size, grid_size))
        self.velocity_y = np.zeros((grid_size, grid_size))
        self.pressure = np.zeros((grid_size, grid_size))
        
        # Previous velocity grids
        self.prev_velocity_x = np.zeros((grid_size, grid_size))
        self.prev_velocity_y = np.zeros((grid_size, grid_size))

        self.new_dirt = np.zeros((grid_size, grid_size))
        self.divergence = np.zeros((grid_size, grid_size))
    
        # Create meshgrid once during initialization
        self.i, self.j = np.meshgrid(np.arange(1, grid_size-1), 
                                    np.arange(1, grid_size-1), 
                                    indexing='ij')
    
    def diffuse_velocity(self):
        a = self.dt * self.viscosity * (self.grid_size - 2) * (self.grid_size - 2)
        for _ in range(self.incompressibility_iters):
            # Replace nested loops with vectorized operations
            self.velocity_x[1:-1, 1:-1] = (self.prev_velocity_x[1:-1, 1:-1] + a * (
                self.velocity_x[2:, 1:-1] + self.velocity_x[:-2, 1:-1] +
                self.velocity_x[1:-1, 2:] + self.velocity_x[1:-1, :-2]
            )) / (1 + 4*a)
            
            self.velocity_y[1:-1, 1:-1] = (self.prev_velocity_y[1:-1, 1:-1] + a * (
                self.velocity_y[2:, 1:-1] + self.velocity_y[:-2, 1:-1] +
                self.velocity_y[1:-1, 2:] + self.velocity_y[1:-1, :-2]
            )) / (1 + 4*a)
    
    def project(self):
        divergence = np.zeros((self.grid_size, self.grid_size))
        
        # Vectorize divergence calculation
        divergence[1:-1, 1:-1] = -0.5 * (
            self.velocity_x[2:, 1:-1] - self.velocity_x[:-2, 1:-1] +
            self.velocity_y[1:-1, 2:] - self.velocity_y[1:-1, :-2]
        ) / self.grid_size
        
        for _ in range(self.incompressibility_iters):
            # Vectorize pressure calculation
            self.pressure[1:-1, 1:-1] = (
                divergence[1:-1, 1:-1] +
                self.pressure[2:, 1:-1] + self.pressure[:-2, 1:-1] +
                self.pressure[1:-1, 2:] + self.pressure[1:-1, :-2]
            ) / 4
        
        # Vectorize velocity update
        self.velocity_x[1:-1, 1:-1] -= 0.5 * (
            self.pressure[2:, 1:-1] - self.pressure[:-2, 1:-1]
        ) * self.grid_size
        self.velocity_y[1:-1, 1:-1] -= 0.5 * (
            self.pressure[1:-1, 2:] - self.pressure[1:-1, :-2]
        ) * self.grid_size

    def advect_dirt(self):
        # Create meshgrid for vectorized operations
        i, j = np.meshgrid(np.arange(1, self.grid_size-1), 
                        np.arange(1, self.grid_size-1), 
                        indexing='ij')
        
        # Calculate positions
        pos_x = i - self.velocity_x[1:-1, 1:-1] * self.dt
        pos_y = j - self.velocity_y[1:-1, 1:-1] * self.dt
        
        # Clip positions
        pos_x = np.clip(pos_x, 0, self.grid_size-2)
        pos_y = np.clip(pos_y, 0, self.grid_size-2)
        
        # Floor to get indices
        x0 = pos_x.astype(int)
        y0 = pos_y.astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        
        # Calculate interpolation weights
        fx = pos_x - x0
        fy = pos_y - y0
        
        # Prepare weights for interpolation
        c00 = (1-fx)*(1-fy)
        c10 = fx*(1-fy)
        c01 = (1-fx)*fy
        c11 = fx*fy
        
        # Interpolate in one step
        new_dirt = np.zeros_like(self.dirt)
        new_dirt[1:-1, 1:-1] = (
            c00 * self.dirt[x0, y0] +
            c10 * self.dirt[x1, y0] +
            c01 * self.dirt[x0, y1] +
            c11 * self.dirt[x1, y1]
        )
        
        self.dirt = new_dirt

    def step(self):
        self.prev_velocity_x = self.velocity_x.copy()
        self.prev_velocity_y = self.velocity_y.copy()
        
        self.diffuse_velocity()
        self.project()
        self.advect_dirt()
        
        # Apply boundary conditions
        self.velocity_x[0, :] = self.velocity_x[-1, :] = 0
        self.velocity_y[:, 0] = self.velocity_y[:, -1] = 0

class CleaningRoomEnv(gym.Env):
    def __init__(self, grid_size=32, max_steps=400):
        super(CleaningRoomEnv, self).__init__()
        
        # Environment parameters
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.spray_strength = 2.0
        self.spray_radius = 3
        
        # Initialize fluid simulator
        self.fluid_sim = FluidSimulator(grid_size)
        
        # Action space: [move_x, move_y, spray_angle]
        # move_x, move_y: [-1, 1] for movement in each direction
        # spray_angle: [-π, π] for spray direction
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -np.pi]),
            high=np.array([1, 1, np.pi]),
            dtype=np.float32
        )
        
        # Observation space: [dirt_map, agent_pos_x, agent_pos_y, agent_rotation]
        self.observation_space = spaces.Dict({
            'dirt_map': spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32),
            'agent_state': spaces.Box(low=np.array([0, 0, -np.pi]), 
                                    high=np.array([grid_size-1, grid_size-1, np.pi]), 
                                    dtype=np.float32)
        })
        
        # Initialize positions
        self.drain_pos = np.array([grid_size-1, grid_size-1])
        self.reset()

        self.reward_history = []
    
        self.reward_history.clear()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset fluid simulator
        self.fluid_sim.dirt = np.random.rand(self.grid_size, self.grid_size) * 0.5
        self.fluid_sim.velocity_x.fill(0)
        self.fluid_sim.velocity_y.fill(0)
        
        # Reset agent state
        self.agent_pos = np.array([0.0, 0.0])
        self.agent_rotation = 0.0
        self.step_count = 0
        
        return self._get_obs(), {}

    def _get_obs(self):
        return {
            'dirt_map': self.fluid_sim.dirt.copy(),
            'agent_state': np.array([self.agent_pos[0], self.agent_pos[1], self.agent_rotation])
        }

    def _apply_spray(self, spray_angle):
        """Apply spray force with directional control"""
        x, y = self.agent_pos
        direction_x = np.cos(spray_angle)
        direction_y = np.sin(spray_angle)
        
        for r in range(self.spray_radius):
            for theta in np.linspace(-np.pi/6, np.pi/6, 5):
                spray_angle_full = spray_angle + theta
                dx = r * np.cos(spray_angle_full)
                dy = r * np.sin(spray_angle_full)
                
                pos_x = int(x + dx)
                pos_y = int(y + dy)
                
                if 0 <= pos_x < self.grid_size-1 and 0 <= pos_y < self.grid_size-1:
                    force = self.spray_strength * (1 - r/self.spray_radius)
                    self.fluid_sim.velocity_x[pos_x, pos_y] += direction_x * force
                    self.fluid_sim.velocity_y[pos_x, pos_y] += direction_y * force
                    self.fluid_sim.dirt[pos_x, pos_y] = max(0, 
                        self.fluid_sim.dirt[pos_x, pos_y] - force * 0.1)

    def _apply_drain_force(self):
        """Apply force field towards drain"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dx = self.drain_pos[0] - i
                dy = self.drain_pos[1] - j
                dist = np.hypot(dx, dy)
                if dist > 0:
                    drain_force = 0.1 / (dist + 1)
                    self.fluid_sim.velocity_x[i, j] += (dx / dist) * drain_force
                    self.fluid_sim.velocity_y[i, j] += (dy / dist) * drain_force
    
    def _spray_penalty(self, spray_angle):
        # Calculate the spray area
        spray_area = np.zeros((self.grid_size, self.grid_size))
        x, y = self.agent_pos
        direction_x = np.cos(spray_angle)
        direction_y = np.sin(spray_angle)
        
        for r in range(self.spray_radius):
            for theta in np.linspace(-np.pi/6, np.pi/6, 5):
                spray_angle_full = spray_angle + theta
                dx = r * np.cos(spray_angle_full)
                dy = r * np.sin(spray_angle_full)
                
                pos_x = int(x + dx)
                pos_y = int(y + dy)
                
                if 0 <= pos_x < self.grid_size-1 and 0 <= pos_y < self.grid_size-1:
                    spray_area[pos_x, pos_y] += 1
        
        # Calculate the penalty based on the spray area
        return -np.sum(spray_area) * 0.01

    def _proximity_reward(self):
        dirt_positions = np.indices((self.grid_size, self.grid_size))
        distance_to_agent = np.sqrt(
            (dirt_positions[0] - self.agent_pos[0])**2 +
            (dirt_positions[1] - self.agent_pos[1])**2
        )
        return np.sum(self.fluid_sim.dirt * (1 / (distance_to_agent + 1e-6))) * 0.1

    def _dynamic_reward_scaling(self, reward, current_dirt):
        if current_dirt < 0.1:
            return reward * 1.5  # Increase reward as dirt decreases
        return reward

    def step(self, action):
        # Store the initial dirt state
        initial_dirt = np.sum(self.fluid_sim.dirt)
        
        # Update agent position and rotation
        move_x, move_y, spray_angle = action
        
        # Move agent
        self.agent_pos[0] = np.clip(self.agent_pos[0] + move_x, 0, self.grid_size-1)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + move_y, 0, self.grid_size-1)
        self.agent_rotation = spray_angle
        
        # Apply spray and drain forces
        self._apply_spray(spray_angle)
        self._apply_drain_force()
        
        # Step fluid simulation
        self.fluid_sim.step()
        
        # Remove dirt at drain
        drain_x, drain_y = self.drain_pos
        self.fluid_sim.dirt[drain_x-1:drain_x+2, drain_y-1:drain_y+2] *= 0.8
        
        # Calculate current dirt level
        current_dirt = np.sum(self.fluid_sim.dirt)
        
        # Calculate dirt cleaned
        dirt_cleaned = initial_dirt - current_dirt
        cleaning_reward = dirt_cleaned * 1.0  # Reward for amount cleaned
        
        # Time penalty
        time_penalty = -0.1  # Small penalty for each step
        
        # Drain direction reward
        dirt_weighted_positions = np.indices((self.grid_size, self.grid_size))
        distance_to_drain = np.sqrt(
            (dirt_weighted_positions[0] - self.drain_pos[0])**2 +
            (dirt_weighted_positions[1] - self.drain_pos[1])**2
        )
        drain_direction_reward = -np.sum(self.fluid_sim.dirt * (1 / (distance_to_drain + 1e-6))) * 0.1
        
        # Spray penalty
        spray_penalty = self._spray_penalty(spray_angle)
        # Proximity reward
        proximity_reward = self._proximity_reward()
        # Combine rewards
        reward = cleaning_reward + time_penalty + drain_direction_reward + spray_penalty + proximity_reward
        reward = self._dynamic_reward_scaling(reward, current_dirt)
        # Update step count
        self.step_count += 1
        
        # Check if episode is done
        done = self.step_count >= self.max_steps or current_dirt < 0.1
        
        # Early completion bonus
        if done and current_dirt < 0.1:
            remaining_steps = self.max_steps - self.step_count
            reward += remaining_steps * 0.5  # Bonus for finishing early
        
        self.previous_total_dirt = current_dirt
        
        # Track rolling average of rewards
        self.reward_history.append(reward)
        
        if len(self.reward_history) > 20:  # Window size
            self.reward_history.pop(0)
            
        # Check for stagnation
        if len(self.reward_history) >= 20:
            recent_avg = np.mean(self.reward_history[-10:])
            older_avg = np.mean(self.reward_history[:-10])
            if abs(recent_avg - older_avg) < 0.005:  # Stagnation threshold
                done = True
                
        return self._get_obs(), reward, done, False, {
            'dirt_cleaned': dirt_cleaned,
            'time_penalty': time_penalty,
            'drain_direction_reward': drain_direction_reward,
            'spray_penalty': spray_penalty,
            'proximity_reward': proximity_reward,
            'step_count': self.step_count
        }

    def render(self):
        plt.clf()
        
        # Plot dirt concentration
        plt.imshow(self.fluid_sim.dirt, cmap='YlOrBr', vmin=0, vmax=0.5)
        
        # Plot velocity field (downsample for clarity)
        skip = 2
        x, y = np.meshgrid(np.arange(0, self.grid_size, skip), 
                          np.arange(0, self.grid_size, skip))
        plt.quiver(x, y, 
                  self.fluid_sim.velocity_x[::skip, ::skip],
                  self.fluid_sim.velocity_y[::skip, ::skip],
                  color='blue', alpha=0.3)
        
        # Plot agent and drain
        plt.scatter(self.drain_pos[1], self.drain_pos[0], 
                   c='blue', marker='s', s=100, label='Drain')
        plt.scatter(self.agent_pos[1], self.agent_pos[0], 
                   c='green', marker='o', s=100, label='Agent')
        
        # Plot spray direction
        spray_length = 3
        plt.arrow(self.agent_pos[1], self.agent_pos[0],
                 spray_length * np.cos(self.agent_rotation),
                 spray_length * np.sin(self.agent_rotation),
                 head_width=0.5, head_length=0.8, fc='green', ec='green')
        
        plt.title(f'Step {self.step_count}')
        plt.legend()
        plt.pause(0.1)
