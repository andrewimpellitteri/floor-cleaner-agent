import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import gymnasium
from gymnasium import spaces
from fluid_sim import FluidSimulator
from scipy.stats import norm

class EnhancedCleaningRoomEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        grid_size=32,
        max_steps=1000,
        initial_dirt_level=1,
        drain_pos_ratio=(0.05, 0.95),
        agent_start_pos_ratio=(0.85, 0),
        spray_strength=0,
        spray_radius=4,
        drain_force_strength=0,
        dirt_decay_rate=0,
        stagnation_reward_threshold=0.002,
        reward_history_window=20,
        initial_dirt_pattern="rug_cleaning",
        viscosity=20,
        dt=0.05,
        incompressibility_iters=20,
    ):  # Include dt and incompressibility_iters in Env init
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
        self.action_space = spaces.Box(
            low=np.array([-1, -1, -1]), high=np.array([1, 1, 1]), dtype=np.float32
        )  # [move_x, move_y, rotate]

        # --- Agent State ---
        self.agent_pos = None
        self.agent_rotation = 0.0

        # --- Observation Space ---
        self.observation_space = spaces.Dict(
            {
                "dirt_map": spaces.Box(
                    low=0, high=1, shape=(grid_size, grid_size), dtype=np.float32
                ),
                "agent_state": spaces.Box(
                    low=np.array([0, 0, -np.pi]),
                    high=np.array([grid_size - 1, grid_size - 1, np.pi]),
                    dtype=np.float32,
                ),
                "relative_drain_position": spaces.Box(
                    low=-1, high=1, shape=(2,), dtype=np.float32
                ),
            }
        )

        # --- Fluid Simulation ---
        self.fluid_sim = FluidSimulator(
            grid_size,
            viscosity=viscosity,
            dt=dt,
            incompressibility_iters=incompressibility_iters,
        )  # Pass dt and incompressibility_iters

        # --- Environment Elements ---
        self.drain_pos_ratio = drain_pos_ratio
        self.drain_pos = (
            int(drain_pos_ratio[0] * grid_size),
            int(drain_pos_ratio[1] * grid_size),
        )

        self.agent_start_pos_ratio = agent_start_pos_ratio
        self.agent_pos = np.array(
            [
                agent_start_pos_ratio[0] * self.grid_size,
                agent_start_pos_ratio[1] * self.grid_size,
            ],
            dtype=np.float32,
        )

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

        # --- Debugging Visualizations Initialization ---
        self.debug_fig, self.debug_axes = plt.subplots(1, 3, figsize=(15, 5))
        self.debug_spray_ax, self.debug_velocity_before_ax, self.debug_velocity_diff_ax = self.debug_axes
        
        self.velocity_before_colorbar = self.debug_fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=self.debug_velocity_before_ax, orientation='vertical', fraction=0.046, pad=0.04)
        self.velocity_diff_colorbar = self.debug_fig.colorbar(plt.cm.ScalarMappable(cmap='coolwarm'), ax=self.debug_velocity_diff_ax, orientation='vertical', fraction=0.046, pad=0.04)


        plt.ion()  # Ensure interactive mode is on for debug plots as well
        self.debug_fig.canvas.draw()  # Initial draw to set up canvas

        
        plt.ion() # Ensure interactive mode is on for debug plots as well
        self.debug_fig.canvas.draw() # Initial draw to set up canvas

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # --- Reset Fluid Simulation ---
        if self.initial_dirt_pattern == "rug_cleaning":
            self.fluid_sim.dirt = self._generate_rug_cleaning_dirt()
        else:
            self.fluid_sim.dirt = (
                np.random.rand(self.grid_size, self.grid_size) * self.initial_dirt_level
            )
        self.fluid_sim.velocity_x.fill(0)
        self.fluid_sim.velocity_y.fill(0)

        # --- Reset Agent State ---
        self.agent_pos = np.array(
            [
                self.agent_start_pos_ratio[0] * self.grid_size,
                self.agent_start_pos_ratio[1] * self.grid_size,
            ],
            dtype=np.float32,
        )
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
            sigma = np.random.uniform(self.grid_size / 10, self.grid_size / 5)
            intensity = np.random.uniform(0.3, 1.0)

            x, y = np.meshgrid(np.arange(self.grid_size), np.arange(self.grid_size))
            gaussian_blob = (
                np.exp(-((x - center_x) ** 2 + (y - center_y) ** 2) / (2 * sigma**2))
                * intensity
            )

            dirt_map += gaussian_blob

        dirt_map = np.clip(dirt_map, 0, 1) * self.initial_dirt_level
        dirt_map += (
            np.random.rand(self.grid_size, self.grid_size)
            * self.initial_dirt_level
            * 0.1
        )
        return dirt_map

    def _get_obs(self):
        """Returns the observation."""
        normalized_agent_pos = self.agent_pos / (self.grid_size - 1)
        normalized_drain_pos = np.array(self.drain_pos) / (self.grid_size - 1)
        relative_drain_position = normalized_drain_pos - normalized_agent_pos

        return {
            "dirt_map": self.fluid_sim.dirt.copy(),
            "agent_state": np.array(
                [self.agent_pos[0], self.agent_pos[1], self.agent_rotation]
            ),
            "relative_drain_position": relative_drain_position,
        }


    def _apply_spray(self):
        """Applies a directional spray force with proper Gaussian profile and directional momentum."""
        x, y = map(int, self.agent_pos)

        PLT_PAUSE = 0.0001

        # Direction vector from agent rotation
        direction_x = np.cos(self.agent_rotation)
        direction_y = np.sin(self.agent_rotation)

        # Spray parameters
        spray_length = self.spray_radius * 2
        spray_width = self.spray_radius

        # Create coordinate grids for vectorized calculations
        i_coords = np.arange(-spray_width, spray_width + 1)
        j_coords = np.arange(0, spray_length)
        j_grid, i_grid = np.meshgrid(j_coords, i_coords)

        # Calculate rotated positions
        rot_x = x + (direction_x * j_grid - direction_y * i_grid)
        rot_y = y + (direction_y * j_grid + direction_x * i_grid)

        # Create Gaussian profile for width (perpendicular to spray direction)
        width_profile = norm.pdf(i_coords, loc=0, scale=spray_width / 3)
        width_profile = width_profile / width_profile.max()

        # Create exponential decay for length (along spray direction)
        length_decay = np.exp(-3 * j_coords / spray_length)

        # Combine profiles into 2D spray pattern
        spray_pattern = np.outer(width_profile, length_decay)

        # --- Debugging Visualizations (temporary) ---
        if True:  # Change to False to disable debug visualizations
            self.debug_spray_ax.clear()
            self.debug_velocity_before_ax.clear()
            self.debug_velocity_diff_ax.clear()

            self.debug_spray_ax.imshow(spray_pattern, cmap='viridis', origin='lower')
            self.debug_spray_ax.set_title('Spray Pattern')

            velocity_x_before_spray = self.fluid_sim.velocity_x.copy()
            velocity_x_before_im = self.debug_velocity_before_ax.imshow(velocity_x_before_spray, cmap='coolwarm',
                                                                         origin='lower')  # Capture return from imshow
            self.debug_velocity_before_ax.set_title('Velocity X Before Spray')
            self.velocity_before_colorbar.update_normal(velocity_x_before_im) # Update colorbar


            velocity_x_diff = np.zeros_like(self.fluid_sim.velocity_x)
            velocity_x_diff_im = self.debug_velocity_diff_ax.imshow(velocity_x_diff, cmap='coolwarm', origin='lower')  # Capture return from imshow
            self.debug_velocity_diff_ax.set_title('Velocity X Change from Spray (Initialized)')
            self.velocity_diff_colorbar.update_normal(velocity_x_diff_im) # Update colorbar


            self.debug_fig.canvas.draw_idle()  # Update canvas
            plt.pause(PLT_PAUSE)  # Pause to see initial state

        # Apply spray pattern to valid grid positions
        valid_mask = (
                (rot_x >= 0)
                & (rot_x < self.grid_size)
                & (rot_y >= 0)
                & (rot_y < self.grid_size)
        )

        rot_x = rot_x.astype(int)
        rot_y = rot_y.astype(int)

        velocity_x_before_apply = self.fluid_sim.velocity_x.copy()
        velocity_y_before_apply = self.fluid_sim.velocity_y.copy()

        # Only apply forces to valid positions
        for i in range(spray_pattern.shape[0]):
            for j in range(spray_pattern.shape[1]):
                if valid_mask[i, j]:
                    force = self.spray_strength * spray_pattern[i, j]
                    # Swapped indices here: velocity_x[rot_x, rot_y] and velocity_y[rot_x, rot_y]
                    self.fluid_sim.velocity_x[rot_x[i, j], rot_y[i, j]] += (
                            direction_x * force
                    )
                    self.fluid_sim.velocity_y[rot_x[i, j], rot_y[i, j]] += (
                            direction_y * force
                    )

        # Add slight spreading effect at spray tip
        spread_radius = max(1, spray_width // 4)
        tip_positions = []
        for i in range(-spread_radius, spread_radius + 1):
            tip_x = int(x + direction_x * spray_length - direction_y * i)
            tip_y = int(y + direction_y * spray_length + direction_x * i)
            if 0 <= tip_x < self.grid_size and 0 <= tip_y < self.grid_size:
                tip_positions.append((tip_x, tip_y))

        # Apply small outward forces at spray tip
        spread_force = self.spray_strength * 0.1
        for tip_x, tip_y in tip_positions:
            dx = tip_x - x
            dy = tip_y - y
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > 0:
                # Swapped indices here for spread effect as well
                self.fluid_sim.velocity_x[tip_x, tip_y] += (dx / dist) * spread_force
                self.fluid_sim.velocity_y[tip_x, tip_y] += (dy / dist) * spread_force

        if True:  # Debug visualizations again, now AFTER spray application
            velocity_x_after_spray = self.fluid_sim.velocity_x
            velocity_x_diff = velocity_x_after_spray - velocity_x_before_apply

            self.debug_velocity_diff_ax.clear()  # Clear only the diff axis now for update
            velocity_x_diff_im = self.debug_velocity_diff_ax.imshow(velocity_x_diff, cmap='coolwarm',
                                                                      origin='lower')  # Capture return from imshow
            self.debug_velocity_diff_ax.set_title('Velocity X Change from Spray (Actual)')
            self.velocity_diff_colorbar.update_normal(velocity_x_diff_im) # Update colorbar

            self.debug_fig.canvas.draw_idle()
            plt.pause(PLT_PAUSE)


    def step(self, action):
        """Executes one step in the environment with continuous spraying and rotation."""
        prev_dirt_amount = np.sum(self.fluid_sim.dirt)

        # --- Agent Action ---
        if isinstance(action, np.ndarray):
            if len(action.shape) == 2:  # If action is (1, 3)
                action = action.flatten()  # Convert to (3,)

        # Now unpack the action
        if len(action) != 3:
            raise ValueError(f"Expected action with 3 values, got {len(action)} values")

        move_x, move_y, rotate_delta = action
        agent_step_size = 1

        x, y = self.agent_pos.astype(int)

        self.agent_pos[0] = np.clip(
            self.agent_pos[0] + move_x * agent_step_size, 0, self.grid_size - 1
        )
        self.agent_pos[1] = np.clip(
            self.agent_pos[1] + move_y * agent_step_size, 0, self.grid_size - 1
        )
        self.agent_rotation += rotate_delta * 0.1 * np.pi
        self.agent_rotation = np.clip(self.agent_rotation, -np.pi, np.pi)

        # --- Environment Dynamics ---
        self._apply_spray()
        self.fluid_sim.step()  # Call the FluidSim step which now calls standalone Numba functions
        # self._decay_dirt()

        # --- Reduced Dirt Removal at Drain ---
        drain_x, drain_y = self.drain_pos
        self.fluid_sim.dirt[drain_x - 1 : drain_x + 2, drain_y - 1 : drain_y + 2] *= (
            0.95
        )

        current_dirt_amount = np.sum(self.fluid_sim.dirt)
        dirt_cleaned = prev_dirt_amount - current_dirt_amount
        print(f"Step {self.current_step}: Dirt Amount AFTER FluidSim step: {current_dirt_amount:.6f}") # DEBUG PRINT


        # --- Reward Calculation ---
        cleaning_reward = dirt_cleaned * 10.0
        time_penalty = -0.02

        # distance_to_drain = np.linalg.norm(self.agent_pos - self.drain_pos)
        # proximity_reward = -0.01 * (distance_to_drain / self.grid_size)

        sparse_completion_reward = 0 if current_dirt_amount > 0.01 else 100

        reward = cleaning_reward + time_penalty + sparse_completion_reward

        # --- Episode Termination Conditions ---
        self.current_step += 1
        truncated = False
        terminated = self.current_step >= self.max_steps or current_dirt_amount < 0.01

        # --- Stagnation Detection ---
        self.reward_history.append(reward)
        stagnated = False
        if len(self.reward_history) >= self.reward_history_window:
            recent_avg_reward = np.mean(
                list(self.reward_history)[-self.reward_history_window // 2 :]
            )
            older_avg_reward = np.mean(
                list(self.reward_history)[: self.reward_history_window // 2]
            )
            if (
                abs(recent_avg_reward - older_avg_reward)
                < self.stagnation_reward_threshold
            ):
                terminated = True
                stagnated = True

        info = {
            "dirt_cleaned": dirt_cleaned,
            "time_penalty": time_penalty,
            # 'proximity_reward': proximity_reward,
            "sparse_completion_reward": sparse_completion_reward,
            "dirt_left": current_dirt_amount,
            "stagnated": stagnated,
            "step_count": self.current_step,
            "action": action,
        }

        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode="human"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots()
            plt.ion()

        self.ax.clear()

        # --- Dirt Map ---
        dirt_plot = self.ax.imshow(
            self.fluid_sim.dirt,
            cmap="YlOrBr",
            vmin=0,
            vmax=self.initial_dirt_level,
            origin="lower",
        )

        # --- Velocity Field ---
        skip = 3
        x_grid, y_grid = np.meshgrid(
            np.arange(0, self.grid_size, skip), np.arange(0, self.grid_size, skip)
        )
        self.ax.quiver(
            y_grid,
            x_grid,
            self.fluid_sim.velocity_y[::skip, ::skip],
            -self.fluid_sim.velocity_x[::skip, ::skip],
            color="blue",
            alpha=0.4,
            pivot="middle",
        )

        # --- Drain and Agent ---
        drain_marker_size = 150
        agent_marker_size = 120

        self.ax.scatter(
            self.drain_pos[1],
            self.drain_pos[0],
            s=drain_marker_size,
            marker="s",
            color="blue",
            label="Drain",
        )
        self.ax.scatter(
            self.agent_pos[1],
            self.agent_pos[0],
            s=agent_marker_size,
            marker="o",
            color="green",
            label="Agent",
        )

        # --- Spray Direction Indicator ---
        agent_x_pixel = self.agent_pos[1]
        agent_y_pixel = self.agent_pos[0]

        spray_length_pixels = 4
        spray_angle_display = self.agent_rotation
        spray_dx_display = spray_length_pixels * np.sin(spray_angle_display)
        spray_dy_display = spray_length_pixels * np.cos(spray_angle_display)

        self.ax.arrow(
            agent_x_pixel,
            agent_y_pixel,
            spray_dx_display,
            spray_dy_display,
            head_width=0.5,
            head_length=0.8,
            fc="green",
            ec="green",
            alpha=0.7,
        )

        # --- Titles and Labels ---
        self.ax.set_title(
            f"Cleaning Room - Step: {self.current_step}, Dirt: {np.sum(self.fluid_sim.dirt):.2f}, Rotation: {np.degrees(self.agent_rotation):.0f} deg"
        )
        self.ax.legend(loc="upper right")

        # --- Grid and Axis ---
        self.ax.set_xlim(-0.5, self.grid_size - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size - 0.5)
        self.ax.set_xticks(np.arange(0, self.grid_size, max(1, self.grid_size // 10)))
        self.ax.set_yticks(np.arange(0, self.grid_size, max(1, self.grid_size // 10)))
        self.ax.set_xticklabels([])
        self.ax.set_yticklabels([])
        self.ax.grid(False)

        if not hasattr(self, "cbar"):
            self.cbar = self.fig.colorbar(
                dirt_plot, ax=self.ax, orientation="vertical", fraction=0.046, pad=0.04
            )
            self.cbar.set_label("Dirt Concentration")

        plt.draw()
        plt.pause(0.001)

        if mode == "rgb_array":
            self.fig.canvas.draw()
            image_array = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            image_array = image_array.reshape(
                self.fig.canvas.get_width_height()[::-1] + (3,)
            )
            return image_array
        elif mode == "human":
            return None

    plt.show()

    def close(self):
        if self.fig:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


if __name__ == "__main__":
    env = EnhancedCleaningRoomEnv(
        grid_size=64,
        max_steps=500,
        drain_force_strength=0,
        dirt_decay_rate=0,
        initial_dirt_pattern="rug_cleaning",
        viscosity=10,
        spray_strength=300,
        spray_radius=5,
        dt=0.05,
        incompressibility_iters=20,
    )  # Example with adjusted parameters, viscosity is now correctly passed
    obs, _ = env.reset()
    plt.ion()
    env.render()
    for _ in range(1000):
        action = env.action_space.sample()  # Random continuous actions
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        if terminated or truncated:
            print(
                f"Episode finished after {info['step_count']} steps, total dirt: {info['current_dirt_amount']:.2f}, stagnated: {info['stagnated']}"
            )
            obs, _ = env.reset()
            env.render()
            if terminated and info["stagnated"]:
                print("Environment stagnated, resetting with new dirt...")
            else:
                print("Environment reset for a new episode...")
        # time.sleep(0.01)
    env.close()
