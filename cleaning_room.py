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
    
    def diffuse_velocity(self):
        a = self.dt * self.viscosity * (self.grid_size - 2) * (self.grid_size - 2)
        for _ in range(self.incompressibility_iters):
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    self.velocity_x[i, j] = (self.prev_velocity_x[i, j] + a * (
                        self.velocity_x[i+1, j] + self.velocity_x[i-1, j] +
                        self.velocity_x[i, j+1] + self.velocity_x[i, j-1]
                    )) / (1 + 4*a)
                    
                    self.velocity_y[i, j] = (self.prev_velocity_y[i, j] + a * (
                        self.velocity_y[i+1, j] + self.velocity_y[i-1, j] +
                        self.velocity_y[i, j+1] + self.velocity_y[i, j-1]
                    )) / (1 + 4*a)
    
    def project(self):
        divergence = np.zeros((self.grid_size, self.grid_size))
        
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                divergence[i, j] = -0.5 * (
                    self.velocity_x[i+1, j] - self.velocity_x[i-1, j] +
                    self.velocity_y[i, j+1] - self.velocity_y[i, j-1]
                ) / self.grid_size
        
        for _ in range(self.incompressibility_iters):
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    self.pressure[i, j] = (
                        divergence[i, j] +
                        self.pressure[i+1, j] + self.pressure[i-1, j] +
                        self.pressure[i, j+1] + self.pressure[i, j-1]
                    ) / 4
        
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                self.velocity_x[i, j] -= 0.5 * (self.pressure[i+1, j] - self.pressure[i-1, j]) * self.grid_size
                self.velocity_y[i, j] -= 0.5 * (self.pressure[i, j+1] - self.pressure[i, j-1]) * self.grid_size

    def advect_dirt(self):
        new_dirt = np.zeros_like(self.dirt)
        
        for i in range(1, self.grid_size-1):
            for j in range(1, self.grid_size-1):
                pos_x = i - self.velocity_x[i, j] * self.dt
                pos_y = j - self.velocity_y[i, j] * self.dt
                
                pos_x = np.clip(pos_x, 0, self.grid_size-2)
                pos_y = np.clip(pos_y, 0, self.grid_size-2)
                
                x0 = int(pos_x)
                y0 = int(pos_y)
                x1 = x0 + 1
                y1 = y0 + 1
                
                fx = pos_x - x0
                fy = pos_y - y0
                
                new_dirt[i, j] = (
                    (1-fx)*(1-fy)*self.dirt[x0, y0] +
                    fx*(1-fy)*self.dirt[x1, y0] +
                    (1-fx)*fy*self.dirt[x0, y1] +
                    fx*fy*self.dirt[x1, y1]
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
    def __init__(self, grid_size=32, max_steps=100):
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
        
        # Calculate rewards components
        dirt_cleaned = initial_dirt - current_dirt
        cleaning_reward = dirt_cleaned * 10.0  # Reward for amount cleaned
        time_penalty = -0.1  # Small penalty for each step
        remaining_dirt_penalty = -current_dirt * 0.5  # Penalty for remaining dirt
        
        # Distance to drain bonus (encourage moving dirt towards drain)
        dirt_weighted_positions = np.indices((self.grid_size, self.grid_size))
        distance_to_drain = np.sqrt(
            (dirt_weighted_positions[0] - self.drain_pos[0])**2 +
            (dirt_weighted_positions[1] - self.drain_pos[1])**2
        )
        drain_direction_reward = -np.sum(self.fluid_sim.dirt * distance_to_drain) * 0.01
        
        # Combine rewards
        reward = cleaning_reward + time_penalty + remaining_dirt_penalty + drain_direction_reward
        
        # Update step count
        self.step_count += 1
        
        # Check if episode is done
        done = self.step_count >= self.max_steps or current_dirt < 0.1
        
        # Early completion bonus
        if done and current_dirt < 0.1:
            remaining_steps = self.max_steps - self.step_count
            reward += remaining_steps * 0.5  # Bonus for finishing early
        
        self.previous_total_dirt = current_dirt
        
        return self._get_obs(), reward, done, False, {
            'dirt_cleaned': dirt_cleaned,
            'time_penalty': time_penalty,
            'remaining_dirt': current_dirt,
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
