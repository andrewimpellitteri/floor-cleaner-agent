import numba
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import gymnasium as gym
from gymnasium import spaces


@numba.jit(nopython=True)
def numba_diffuse_velocity(velocity_x, velocity_y, prev_velocity_x, prev_velocity_y, dt, viscosity, grid_size, incompressibility_iters):
    diff_coeff = dt * viscosity * (grid_size - 2) * (grid_size - 2)
    for _ in range(incompressibility_iters):
        velocity_x[1:-1, 1:-1] = (prev_velocity_x[1:-1, 1:-1] + diff_coeff * (
            velocity_x[2:, 1:-1] + velocity_x[:-2, 1:-1] +
            velocity_x[1:-1, 2:] + velocity_x[1:-1, :-2]
        )) / (1 + 4 * diff_coeff)
        velocity_y[1:-1, 1:-1] = (prev_velocity_y[1:-1, 1:-1] + diff_coeff * (
            velocity_y[2:, 1:-1] + velocity_y[:-2, 1:-1] +
            velocity_y[1:-1, 2:] + velocity_y[1:-1, :-2]
        )) / (1 + 4 * diff_coeff)
    return velocity_x, velocity_y


@numba.jit(nopython=True)
def numba_project(velocity_x, velocity_y, pressure, grid_size, incompressibility_iters):
    divergence = np.zeros((grid_size, grid_size), dtype=np.float32)
    divergence[1:-1, 1:-1] = -0.5 * (
        velocity_x[2:, 1:-1] - velocity_x[:-2, 1:-1] +
        velocity_y[1:-1, 2:] - velocity_y[1:-1, :-2]
    ) / grid_size
    for _ in range(incompressibility_iters):
        pressure[1:-1, 1:-1] = (
            divergence[1:-1, 1:-1] +
            pressure[2:, 1:-1] + pressure[:-2, 1:-1] +
            pressure[1:-1, 2:] + pressure[1:-1, :-2]
        ) / 4
    velocity_x[1:-1, 1:-1] -= 0.5 * (
        pressure[2:, 1:-1] - pressure[:-2, 1:-1]
    ) * grid_size
    velocity_y[1:-1, 1:-1] -= 0.5 * (
        pressure[1:-1, 2:] - pressure[1:-1, :-2]
    ) * grid_size
    return velocity_x, velocity_y, pressure


@numba.jit(nopython=True)
def numba_advect_dirt(dirt, velocity_x, velocity_y, dt, grid_size, i_grid, j_grid): # Pass i_grid and j_grid
    predicted_dirt = np.zeros_like(dirt) # For predictor step
    corrected_dirt = np.zeros_like(dirt) # For corrector step
    new_dirt = np.zeros_like(dirt)

    # Predictor step (forward advection - same as before but storing in predicted_dirt)
    for row in range(grid_size):
        for col in range(grid_size):
            pos_x = i_grid[row, col] - velocity_x[row, col] * dt
            pos_y = j_grid[row, col] - velocity_y[row, col] * dt
            pos_x = max(0, min(pos_x, grid_size - 1))
            pos_y = max(0, min(pos_y, grid_size - 1))

            x0 = int(pos_x)
            y0 = int(pos_y)
            x1 = min(x0 + 1, grid_size - 1)
            y1 = min(y0 + 1, grid_size - 1)

            fx = pos_x - x0
            fy = pos_y - y0

            c00 = (1 - fx) * (1 - fy)
            c10 = fx * (1 - fy)
            c01 = (1 - fx) * fy
            c11 = fx * fy

            predicted_dirt[row, col] = (
                c00 * dirt[x0, y0] +
                c10 * dirt[x1, y0] +
                c01 * dirt[x0, y1] +
                c11 * dirt[x1, y1]
            )

    # Corrector step (backward advection using predicted dirt and velocities)
    for row in range(grid_size):
        for col in range(grid_size):
            pos_x_corr = i_grid[row, col] + velocity_x[row, col] * dt # Notice the + sign for backward advection
            pos_y_corr = j_grid[row, col] + velocity_y[row, col] * dt # Notice the + sign for backward advection
            pos_x_corr = max(0, min(pos_x_corr, grid_size - 1))
            pos_y_corr = max(0, min(pos_y_corr, grid_size - 1))

            x0_corr = int(pos_x_corr)
            y0_corr = int(pos_y_corr)
            x1_corr = min(x0_corr + 1, grid_size - 1)
            y1_corr = min(y0_corr + 1, grid_size - 1)

            fx_corr = pos_x_corr - x0_corr
            fy_corr = pos_y_corr - y0_corr

            c00_corr = (1 - fx_corr) * (1 - fy_corr)
            c10_corr = fx_corr * (1 - fy_corr)
            c01_corr = (1 - fx_corr) * fy_corr
            c11_corr = fx_corr * fy_corr

            corrected_dirt[row, col] = (
                c00_corr * predicted_dirt[x0_corr, y0_corr] + # Using predicted_dirt here
                c10_corr * predicted_dirt[x1_corr, y0_corr] + # Using predicted_dirt here
                c01_corr * predicted_dirt[x0_corr, y1_corr] + # Using predicted_dirt here
                c11_corr * predicted_dirt[x1_corr, y1_corr]  # Using predicted_dirt here
            )

    # Average predictor and corrector steps to get final dirt
    for row in range(grid_size):
        for col in range(grid_size):
            new_dirt[row, col] = 0.5 * (predicted_dirt[row, col] + corrected_dirt[row, col])

    return new_dirt


class FluidSimulator:
    def __init__(self, grid_size, viscosity=0.12, dt=0.05, incompressibility_iters=20):
        self.grid_size = grid_size
        self.dt = dt
        self.viscosity = viscosity
        self.diffusion_rate = 0.02
        self.incompressibility_iters = incompressibility_iters

        self.dirt = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.velocity_x = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.velocity_y = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.pressure = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.prev_velocity_x = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.prev_velocity_y = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.new_dirt = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.divergence = np.zeros((grid_size, grid_size), dtype=np.float32)
        self.i, self.j = np.meshgrid(np.arange(0, grid_size),
                                    np.arange(0, grid_size),
                                    indexing='ij')

    def diffuse_velocity(self):
        self.velocity_x, self.velocity_y = numba_diffuse_velocity(self.velocity_x, self.velocity_y, self.prev_velocity_x, self.prev_velocity_y, self.dt, self.viscosity, self.grid_size, self.incompressibility_iters)

    def project(self):
        self.velocity_x, self.velocity_y, self.pressure = numba_project(self.velocity_x, self.velocity_y, self.pressure, self.grid_size, self.incompressibility_iters)

    def advect_dirt(self):
        self.dirt = numba_advect_dirt(self.dirt, self.velocity_x, self.velocity_y, self.dt, self.grid_size, self.i, self.j)


    def step(self):
        self.prev_velocity_x = self.velocity_x.copy()
        self.prev_velocity_y = self.velocity_y.copy()
        self.diffuse_velocity()
        self.project()
        self.advect_dirt()
        self.velocity_x[0, :] = self.velocity_x[-1, :] = 0
        self.velocity_y[:, 0] = self.velocity_y[:, -1] = 0


class EnhancedCleaningRoomEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, grid_size=32, max_steps=1000, initial_dirt_level=0.95, drain_pos_ratio=(0.05, 0.95),
                 agent_start_pos_ratio=(0.85, 0), spray_strength=3, spray_radius=4, drain_force_strength=0,
                 dirt_decay_rate=0, stagnation_reward_threshold=0.002, reward_history_window=20, initial_dirt_pattern='rug_cleaning',
                 viscosity=10, dt=0.05, incompressibility_iters=20): # Include dt and incompressibility_iters in Env init
        super().__init__()

        # --- Environment Parameters ---
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.initial_dirt_level = initial_dirt_level
        self.current_step = 0
        self.initial_dirt_pattern = initial_dirt_pattern

        # --- Agent & Action Parameters ---
        self.spray_strength = spray_strength
        self.spray_radius = spray_radius
        self.action_space = spaces.Box(low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32) # [move_x, move_y, rotate]

        # --- Agent State ---
        self.agent_pos = None
        self.agent_rotation = 0.0

        # --- Observation Space ---
        self.observation_space = spaces.Dict({
            'dirt_map': spaces.Box(low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32),
            'agent_state': spaces.Box(low=np.array([0, 0, -np.pi]), high=np.array([grid_size-1, grid_size-1, np.pi]), dtype=np.float32),
            'relative_drain_position': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        })

        # --- Fluid Simulation ---
        self.fluid_sim = FluidSimulator(grid_size, viscosity=viscosity, dt=dt, incompressibility_iters=incompressibility_iters) # Pass dt and incompressibility_iters

        # --- Environment Elements ---
        self.drain_pos_ratio = drain_pos_ratio
        self.drain_pos = (int(drain_pos_ratio[0] * grid_size), int(drain_pos_ratio[1] * grid_size))

        self.agent_start_pos_ratio = agent_start_pos_ratio
        self.agent_pos = np.array([agent_start_pos_ratio[0] * self.grid_size, agent_start_pos_ratio[1] * self.grid_size], dtype=np.float32)


        self.drain_force_strength = drain_force_strength
        self.dirt_decay_rate = dirt_decay_rate

        # --- Reward System ---
        self.initial_total_dirt = 0
        self.previous_total_dirt = 0

        # --- Termination Conditions and Stagnation Detection ---
        self.stagnation_reward_threshold = stagnation_reward_threshold
        self.reward_history_window = reward_history_window
        self.reward_history = deque(maxlen=reward_history_window)

        # --- Rendering ---
        self.fig, self.ax = None, None

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # --- Reset Fluid Simulation ---
        if self.initial_dirt_pattern == 'rug_cleaning':
            self.fluid_sim.dirt = self._generate_rug_cleaning_dirt()
        else:
            self.fluid_sim.dirt = np.random.rand(self.grid_size, self.grid_size) * self.initial_dirt_level
        self.fluid_sim.velocity_x.fill(0)
        self.fluid_sim.velocity_y.fill(0)

        # --- Reset Agent State ---
        self.agent_pos = np.array([self.agent_start_pos_ratio[0] * self.grid_size, self.agent_start_pos_ratio[1] * self.grid_size], dtype=np.float32)
        self.agent_rotation = 0.0
        self.current_step = 0
        self.reward_history.clear()
        self.initial_total_dirt = np.sum(self.fluid_sim.dirt)
        self.previous_total_dirt = self.initial_total_dirt

        return self._get_obs(), {}

    def _generate_rug_cleaning_dirt(self):
        """Generates a dirt map simulating rug cleaning."""
        dirt_map = np.zeros((self.grid_size, self.grid_size))
        num_blobs = np.random.randint(3, 7)
        for _ in range(num_blobs):
            center_x = np.random.uniform(0, self.grid_size)
            center_y = np.random.uniform(0, self.grid_size)
            sigma = np.random.uniform(self.grid_size/10, self.grid_size/5)
            intensity = np.random.uniform(0.3, 1.0)

            x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
            gaussian_blob = np.exp(-((x - center_x)**2 + (y - center_y)**2) / (2 * sigma**2)) * intensity

            dirt_map += gaussian_blob

        dirt_map = np.clip(dirt_map, 0, 1) * self.initial_dirt_level
        dirt_map += np.random.rand(self.grid_size, self.grid_size) * self.initial_dirt_level * 0.1
        return dirt_map


    def _get_obs(self):
        """Returns the observation."""
        normalized_agent_pos = self.agent_pos / (self.grid_size - 1)
        normalized_drain_pos = np.array(self.drain_pos) / (self.grid_size - 1)
        relative_drain_position = normalized_drain_pos - normalized_agent_pos

        return {
            'dirt_map': self.fluid_sim.dirt.copy(),
            'agent_state': np.array([self.agent_pos[0], self.agent_pos[1], self.agent_rotation]),
            'relative_drain_position': relative_drain_position
        }


    def _apply_spray(self):
        """Applies spray force centered at agent's position, with rotation."""
        x, y = map(int, self.agent_pos)
        agent_direction_x = np.cos(self.agent_rotation)
        agent_direction_y = np.sin(self.agent_rotation)

        for r in range(self.spray_radius):
            for theta in np.linspace(-np.pi/4, np.pi/4, 7):
                spray_angle_local =  theta
                spray_angle_global = self.agent_rotation + spray_angle_local
                dx = r * np.cos(spray_angle_global)
                dy = r * np.sin(spray_angle_global)

                pos_x = int(x + dx)
                pos_y = int(y + dy)

                if 0 <= pos_x < self.grid_size and 0 <= pos_y < self.grid_size:
                    force = self.spray_strength * (1 - r / self.spray_radius)
                    self.fluid_sim.velocity_x[pos_x, pos_y] += agent_direction_x * force
                    self.fluid_sim.velocity_y[pos_x, pos_y] += agent_direction_y * force


    def _apply_drain_force(self):
        """Applies a force field towards the drain."""
        drain_x, drain_y = self.drain_pos
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dx = drain_x - i
                dy = drain_y - j
                dist = np.hypot(dx, dy)
                if dist > 0:
                    drain_force = self.drain_force_strength / (dist + 1)
                    self.fluid_sim.velocity_x[i, j] += (dx / dist) * drain_force
                    self.fluid_sim.velocity_y[i, j] += (dy / dist) * drain_force

    def _decay_dirt(self):
        """Gradually reduces dirt over time."""
        self.fluid_sim.dirt = np.maximum(0, self.fluid_sim.dirt - self.dirt_decay_rate)


    def step(self, action):
        """Executes one step in the environment with continuous spraying and rotation."""
        prev_dirt_amount = np.sum(self.fluid_sim.dirt)

        # --- Agent Action ---
        move_x, move_y, rotate_delta = action
        agent_step_size = 1

        x, y = self.agent_pos.astype(int)

        self.agent_pos[0] = np.clip(self.agent_pos[0] + move_x * agent_step_size, 0, self.grid_size - 1)
        self.agent_pos[1] = np.clip(self.agent_pos[1] + move_y * agent_step_size, 0, self.grid_size - 1)
        self.agent_rotation += rotate_delta * 0.1 * np.pi
        self.agent_rotation = np.clip(self.agent_rotation, -np.pi, np.pi)


        # --- Environment Dynamics ---
        self._apply_spray()
        self._apply_drain_force()
        self.fluid_sim.step() # Call the FluidSim step which now calls standalone Numba functions
        self._decay_dirt()

        # --- Reduced Dirt Removal at Drain ---
        drain_x, drain_y = self.drain_pos
        self.fluid_sim.dirt[drain_x-1:drain_x+2, drain_y-1:drain_y+2] *= 0.95

        current_dirt_amount = np.sum(self.fluid_sim.dirt)
        dirt_cleaned = prev_dirt_amount - current_dirt_amount


        # --- Reward Calculation ---
        cleaning_reward = dirt_cleaned * 10.0
        time_penalty = -0.02

        distance_to_drain = np.linalg.norm(self.agent_pos - self.drain_pos)
        proximity_reward = -0.01 * (distance_to_drain / self.grid_size)

        sparse_completion_reward = 0 if current_dirt_amount > 0.01 else 100

        reward = cleaning_reward + time_penalty + proximity_reward + sparse_completion_reward

        # --- Episode Termination Conditions ---
        self.current_step += 1
        truncated = False
        terminated = self.current_step >= self.max_steps or current_dirt_amount < 0.01


        # --- Stagnation Detection ---
        self.reward_history.append(reward)
        stagnated = False
        if len(self.reward_history) >= self.reward_history_window:
            recent_avg_reward = np.mean(list(self.reward_history)[-self.reward_history_window//2:])
            older_avg_reward = np.mean(list(self.reward_history)[:self.reward_history_window//2])
            if abs(recent_avg_reward - older_avg_reward) < self.stagnation_reward_threshold:
                terminated = True
                stagnated = True


        info = {
            'dirt_cleaned': dirt_cleaned,
            'time_penalty': time_penalty,
            'proximity_reward': proximity_reward,
            'sparse_completion_reward': sparse_completion_reward,
            'dirt_left': current_dirt_amount,
            'stagnated': stagnated,
            'step_count': self.current_step,
            'action': action
        }


        return self._get_obs(), reward, terminated, truncated, info


    def render(self, mode='human'):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()

        self.ax.clear()

        # --- Dirt Map ---
        dirt_plot = self.ax.imshow(self.fluid_sim.dirt, cmap='YlOrBr', vmin=0, vmax=self.initial_dirt_level, origin='lower')

        # --- Velocity Field ---
        skip = 3
        x_grid, y_grid = np.meshgrid(np.arange(0, self.grid_size, skip), np.arange(0, self.grid_size, skip))
        self.ax.quiver(y_grid, x_grid,
                       self.fluid_sim.velocity_y[::skip, ::skip],
                       -self.fluid_sim.velocity_x[::skip, ::skip],
                       color='blue', alpha=0.4, pivot='middle')

        # --- Drain and Agent ---
        drain_marker_size = 150
        agent_marker_size = 120

        self.ax.scatter(self.drain_pos[1], self.drain_pos[0], s=drain_marker_size, marker='s', color='blue', label='Drain')
        self.ax.scatter(self.agent_pos[1], self.agent_pos[0], s=agent_marker_size, marker='o', color='green', label='Agent')

        # --- Spray Direction Indicator ---
        agent_x_pixel = self.agent_pos[1]
        agent_y_pixel = self.agent_pos[0]

        spray_length_pixels = 4
        spray_angle_display = self.agent_rotation
        spray_dx_display = spray_length_pixels * np.sin(spray_angle_display)
        spray_dy_display = spray_length_pixels * np.cos(spray_angle_display)

        self.ax.arrow(agent_x_pixel, agent_y_pixel,
                      spray_dx_display, spray_dy_display,
                      head_width=0.5, head_length=0.8, fc='green', ec='green', alpha=0.7)


        # --- Titles and Labels ---
        self.ax.set_title(f'Cleaning Room - Step: {self.current_step}, Dirt: {np.sum(self.fluid_sim.dirt):.2f}, Rotation: {np.degrees(self.agent_rotation):.0f} deg')
        self.ax.legend(loc='upper right')

        # --- Grid and Axis ---
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(np.arange(0, self.grid_size, max(1, self.grid_size // 10)))
        self.ax.set_yticks(np.arange(0, self.grid_size, max(1, self.grid_size // 10)))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(False)

        if not hasattr(self, 'cbar'):
            self.cbar = self.fig.colorbar(dirt_plot, ax=self.ax, orientation='vertical', fraction=0.046, pad=0.04)
            self.cbar.set_label('Dirt Concentration')

        plt.draw()
        plt.pause(0.001)


        if mode == 'rgb_array':
            self.fig.canvas.draw()
            image_array = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_array = image_array.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image_array
        elif mode == 'human':
            return None


    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None



# if __name__ == '__main__':
#     env = EnhancedCleaningRoomEnv(grid_size=64, max_steps=500, drain_force_strength=0.01, dirt_decay_rate=0.0001, initial_dirt_pattern='rug_cleaning', viscosity=0.005, spray_strength=5.0, spray_radius=5, dt=0.05, incompressibility_iters=20) # Example with adjusted parameters, viscosity is now correctly passed
#     obs, _ = env.reset()
#     plt.ion()
#     env.render()
#     for _ in range(1000):
#         action = env.action_space.sample() # Random continuous actions
#         obs, reward, terminated, truncated, info = env.step(action)
#         env.render()
#         if terminated or truncated:
#             print(f"Episode finished after {info['step_count']} steps, total dirt: {info['current_dirt_amount']:.2f}, stagnated: {info['stagnated']}")
#             obs, _ = env.reset()
#             env.render()
#             if terminated and info['stagnated']:
#                 print("Environment stagnated, resetting with new dirt...")
#             else:
#                 print("Environment reset for a new episode...")
#         # time.sleep(0.01)
#     env.close()