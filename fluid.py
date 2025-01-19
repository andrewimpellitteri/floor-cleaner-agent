import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt

class FluidSimulator:
    def __init__(self, grid_size, drain_pos=None):
        self.grid_size = grid_size
        self.dt = 0.1
        self.viscosity = 0.4
        self.diffusion_rate = 0.05
        self.incompressibility_iters = 50
        
        # Initialize grids
        self.dirt = np.random.rand(grid_size, grid_size) * 0.5
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
        
        if drain_pos is None:
            self.drain_pos = np.array([grid_size-1, grid_size-1])
        else:
            self.drain_pos = drain_pos
    
    def diffuse_velocity(self):
        a = self.dt * self.viscosity * (self.grid_size - 2) * (self.grid_size - 2)
        for _ in range(self.incompressibility_iters):
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
        divergence[1:-1, 1:-1] = -0.5 * (
            self.velocity_x[2:, 1:-1] - self.velocity_x[:-2, 1:-1] +
            self.velocity_y[1:-1, 2:] - self.velocity_y[1:-1, :-2]
        ) / self.grid_size
        
        for _ in range(self.incompressibility_iters):
            self.pressure[1:-1, 1:-1] = (
                divergence[1:-1, 1:-1] +
                self.pressure[2:, 1:-1] + self.pressure[:-2, 1:-1] +
                self.pressure[1:-1, 2:] + self.pressure[1:-1, :-2]
            ) / 4
        
        self.velocity_x[1:-1, 1:-1] -= 0.5 * (
            self.pressure[2:, 1:-1] - self.pressure[:-2, 1:-1]
        ) * self.grid_size
        self.velocity_y[1:-1, 1:-1] -= 0.5 * (
            self.pressure[1:-1, 2:] - self.pressure[1:-1, :-2]
        ) * self.grid_size

    def advect_dirt(self):
        i, j = np.meshgrid(np.arange(1, self.grid_size-1), 
                           np.arange(1, self.grid_size-1), 
                           indexing='ij')
        
        pos_x = i - self.velocity_x[1:-1, 1:-1] * self.dt
        pos_y = j - self.velocity_y[1:-1, 1:-1] * self.dt
        
        pos_x = np.clip(pos_x, 0, self.grid_size-2)
        pos_y = np.clip(pos_y, 0, self.grid_size-2)
        
        x0 = pos_x.astype(int)
        y0 = pos_y.astype(int)
        x1 = x0 + 1
        y1 = y0 + 1
        
        fx = pos_x - x0
        fy = pos_y - y0
        
        c00 = (1 - fx) * (1 - fy)
        c10 = fx * (1 - fy)
        c01 = (1 - fx) * fy
        c11 = fx * fy
        
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
        
        self.velocity_x[0, :] = self.velocity_x[-1, :] = 0
        self.velocity_y[:, 0] = self.velocity_y[:, -1] = 0
        
        drain_x, drain_y = self.drain_pos
        self.dirt[drain_x-1:drain_x+2, drain_y-1:drain_y+2] *= 0.8
    
    def _apply_spray(self, agent_pos, spray_direction, spray_strength=10.0, spray_radius=5):
        x, y = agent_pos
        direction_x = np.cos(spray_direction)
        direction_y = np.sin(spray_direction)
        
        for r in range(spray_radius):
            for theta in np.linspace(-np.pi/6, np.pi/6, 5):
                spray_angle_full = spray_direction + theta
                dx = r * np.cos(spray_angle_full)
                dy = r * np.sin(spray_angle_full)
                
                pos_x = int(round(x + dx))
                pos_y = int(round(y + dy))
                
                if 0 <= pos_x < self.grid_size and 0 <= pos_y < self.grid_size:
                    force = spray_strength * (1 - r / spray_radius)
                    self.velocity_x[pos_x, pos_y] += direction_x * force
                    self.velocity_y[pos_x, pos_y] += direction_y * force
                    self.dirt[pos_x, pos_y] = max(0, self.dirt[pos_x, pos_y] - force * 0.2)
        print(f"Spray applied with strength {spray_strength} and radius {spray_radius}")
    def _apply_drain_force(self):
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dx = self.drain_pos[0] - i
                dy = self.drain_pos[1] - j
                dist = np.hypot(dx, dy)
                if dist > 0:
                    drain_force = 0.1 / (dist + 1)
                    self.velocity_x[i, j] += (dx / dist) * drain_force
                    self.velocity_y[i, j] += (dy / dist) * drain_force

# Initialize the fluid simulator
grid_size = 32
fluid_sim = FluidSimulator(grid_size)

# Initialize agent position and rotation
agent_pos = np.array([0.0, 0.0])
agent_rotation = 0.0  # in radians

# Set up the figure and axis
plt.ion()
fig, ax = plt.subplots()
dirt_plot = ax.imshow(fluid_sim.dirt, cmap='YlOrBr', vmin=0, vmax=0.5)
velocity_x = fluid_sim.velocity_x
velocity_y = fluid_sim.velocity_y
velocity_quiver = ax.quiver(velocity_x, velocity_y, scale=100)
agent_scatter = ax.scatter(agent_pos[1], agent_pos[0], c='green', marker='o', s=100)
spray_arrow = ax.arrow(agent_pos[1], agent_pos[0], 
                       3 * np.cos(agent_rotation),
                       3 * np.sin(agent_rotation),
                       head_width=0.5, head_length=0.8, fc='green', ec='green')

# Add text annotation for total dirt
total_dirt_text = ax.text(0.02, 0.98, 'Total Dirt: 0.0', transform=ax.transAxes, 
                          fontsize=12, va='top')


pressed_keys = set()
running = True

def on_key_press(event):
    global pressed_keys, running
    pressed_keys.add(event.key)
    if event.key == 'q':
        running = False

def on_key_release(event):
    global pressed_keys
    pressed_keys.discard(event.key)

fig.canvas.mpl_connect('key_press_event', on_key_press)
fig.canvas.mpl_connect('key_release_event', on_key_release)

# Define movement and rotation parameters
movement_step = 0.5
rotation_step = np.deg2rad(15)

# Main loop
while running:
    # Update agent's state based on key presses
    if 'up' in pressed_keys:
        agent_pos[0] += movement_step * np.cos(agent_rotation)
        agent_pos[1] += movement_step * np.sin(agent_rotation)
    if 'down' in pressed_keys:
        agent_pos[0] -= movement_step * np.cos(agent_rotation)
        agent_pos[1] -= movement_step * np.sin(agent_rotation)
    if 'left' in pressed_keys:
        agent_rotation -= rotation_step
    if 'right' in pressed_keys:
        agent_rotation += rotation_step
    agent_pos[0] = np.clip(agent_pos[0], 0, grid_size-1)
    agent_pos[1] = np.clip(agent_pos[1], 0, grid_size-1)
    
    # Apply spray continuously while 's' is held down
    if 's' in pressed_keys:
        fluid_sim._apply_spray(agent_pos, agent_rotation)
    
    # Apply drain forces
    fluid_sim._apply_drain_force()
    
    # Step the fluid simulation
    fluid_sim.step()
    
    # Update plots
    dirt_plot.set_data(fluid_sim.dirt)
    velocity_x = fluid_sim.velocity_x
    velocity_y = fluid_sim.velocity_y
    velocity_quiver.set_UVC(velocity_x, velocity_y)
    agent_scatter.set_offsets([agent_pos[1], agent_pos[0]])
    spray_arrow.remove()
    spray_arrow = ax.arrow(agent_pos[1], agent_pos[0], 
                           3 * np.cos(agent_rotation),
                           3 * np.sin(agent_rotation),
                           head_width=0.5, head_length=0.8, fc='green', ec='green')
    # Calculate and update total dirt
    total_dirt = np.sum(fluid_sim.dirt)
    total_dirt_text.set_text(f'Total Dirt: {total_dirt:.2f}')
    
    fig.canvas.draw()
    
    # Pause for frame rate control
    plt.pause(0.1)