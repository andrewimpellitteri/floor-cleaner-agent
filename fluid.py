import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class FluidSimulation:
    def __init__(self, grid_size=50):
        self.grid_size = grid_size
        self.dt = 0.1  # Time step
        self.viscosity = 0.1  # Fluid viscosity
        self.diffusion_rate = 0.05
        self.incompressibility_iters = 20
        
        # Initialize grids
        self.dirt = np.random.rand(grid_size, grid_size) * 0.5
        self.velocity_x = np.zeros((grid_size, grid_size))
        self.velocity_y = np.zeros((grid_size, grid_size))
        self.pressure = np.zeros((grid_size, grid_size))
        
        # Initialize previous step grids for velocity update
        self.prev_velocity_x = np.zeros((grid_size, grid_size))
        self.prev_velocity_y = np.zeros((grid_size, grid_size))
        
        # Drain and agent positions
        self.drain_pos = (grid_size - 1, grid_size - 1)
        self.agent_pos = (0, 0)
        self.spray_angle = 0  # Angle in radians
        self.spray_force = 2.0
        self.spray_radius = 5

    def diffuse_velocity(self):
        """Diffuse velocity using Gauss-Seidel relaxation"""
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
        """Ensure mass conservation by projecting velocity onto divergence-free field"""
        divergence = np.zeros((self.grid_size, self.grid_size))
        
        # Calculate divergence
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                divergence[i, j] = -0.5 * (
                    self.velocity_x[i+1, j] - self.velocity_x[i-1, j] +
                    self.velocity_y[i, j+1] - self.velocity_y[i, j-1]
                ) / self.grid_size
        
        # Solve pressure Poisson equation
        for _ in range(self.incompressibility_iters):
            for i in range(1, self.grid_size - 1):
                for j in range(1, self.grid_size - 1):
                    self.pressure[i, j] = (
                        divergence[i, j] +
                        self.pressure[i+1, j] + self.pressure[i-1, j] +
                        self.pressure[i, j+1] + self.pressure[i, j-1]
                    ) / 4
        
        # Update velocities
        for i in range(1, self.grid_size - 1):
            for j in range(1, self.grid_size - 1):
                self.velocity_x[i, j] -= 0.5 * (self.pressure[i+1, j] - self.pressure[i-1, j]) * self.grid_size
                self.velocity_y[i, j] -= 0.5 * (self.pressure[i, j+1] - self.pressure[i, j-1]) * self.grid_size

    def apply_spray(self):
        """Apply spray force with directional control"""
        x, y = self.agent_pos
        direction_x = np.cos(self.spray_angle)
        direction_y = np.sin(self.spray_angle)
        
        # Create spray pattern
        for r in range(self.spray_radius):
            for theta in np.linspace(-np.pi/6, np.pi/6, 5):  # Spray cone
                spray_angle = self.spray_angle + theta
                dx = r * np.cos(spray_angle)
                dy = r * np.sin(spray_angle)
                
                # Calculate spray position
                pos_x = int(x + dx)
                pos_y = int(y + dy)
                
                if 0 <= pos_x < self.grid_size-1 and 0 <= pos_y < self.grid_size-1:
                    # Add velocity at spray position
                    force = self.spray_force * (1 - r/self.spray_radius)  # Force decreases with distance
                    self.velocity_x[pos_x, pos_y] += direction_x * force
                    self.velocity_y[pos_x, pos_y] += direction_y * force
                    
                    # Remove dirt proportional to spray force
                    self.dirt[pos_x, pos_y] = max(0, self.dirt[pos_x, pos_y] - force * 0.1)

    def apply_drain_force(self):
        """Apply force field towards drain"""
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                dx = self.drain_pos[0] - i
                dy = self.drain_pos[1] - j
                dist = np.hypot(dx, dy)
                if dist > 0:
                    drain_force = 0.1 / (dist + 1)  # Force decreases with distance
                    self.velocity_x[i, j] += (dx / dist) * drain_force
                    self.velocity_y[i, j] += (dy / dist) * drain_force

    def advect_dirt(self):
        """Advect dirt based on velocity field using semi-Lagrangian advection"""
        new_dirt = np.zeros_like(self.dirt)
        
        for i in range(1, self.grid_size-1):
            for j in range(1, self.grid_size-1):
                # Trace particle back in time
                pos_x = i - self.velocity_x[i, j] * self.dt
                pos_y = j - self.velocity_y[i, j] * self.dt
                
                # Ensure positions are within bounds
                pos_x = np.clip(pos_x, 0, self.grid_size-2)
                pos_y = np.clip(pos_y, 0, self.grid_size-2)
                
                # Bilinear interpolation
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
        """Perform one simulation step"""
        # Save previous velocities
        self.prev_velocity_x = self.velocity_x.copy()
        self.prev_velocity_y = self.velocity_y.copy()
        
        # Apply forces
        self.apply_spray()
        self.apply_drain_force()
        
        # Velocity steps
        self.diffuse_velocity()
        self.project()
        
        # Advect dirt
        self.advect_dirt()
        
        # Apply boundary conditions
        self.velocity_x[0, :] = self.velocity_x[-1, :] = 0
        self.velocity_y[:, 0] = self.velocity_y[:, -1] = 0
        
        # Remove dirt at drain
        drain_x, drain_y = self.drain_pos
        self.dirt[drain_x-1:drain_x+2, drain_y-1:drain_y+2] *= 0.8  # Drain effect

def create_animation(grid_size=50, max_steps=200):
    sim = FluidSimulation(grid_size)
    fig, ax = plt.subplots(figsize=(8, 8))
    
    def update(frame):
        sim.step()
        
        # Update visualization
        ax.clear()
        im = ax.imshow(sim.dirt, cmap='YlOrBr', vmin=0, vmax=0.5)
        
        # Draw velocity field (downsample for clarity)
        skip = 3
        x, y = np.meshgrid(np.arange(0, grid_size, skip), np.arange(0, grid_size, skip))
        ax.quiver(x, y, 
                 sim.velocity_x[::skip, ::skip],
                 sim.velocity_y[::skip, ::skip],
                 color='blue', alpha=0.3)
        
        # Draw drain and agent
        ax.scatter(sim.drain_pos[1], sim.drain_pos[0], c='blue', marker='s', s=100, label='Drain')
        ax.scatter(sim.agent_pos[1], sim.agent_pos[0], c='green', marker='o', s=100, label='Agent')
        
        # Draw spray direction
        spray_length = 3
        ax.arrow(sim.agent_pos[1], sim.agent_pos[0],
                spray_length * np.cos(sim.spray_angle),
                spray_length * np.sin(sim.spray_angle),
                head_width=0.5, head_length=0.8, fc='green', ec='green')
        
        ax.set_title(f'Step {frame}')
        ax.legend()
        return [im]
    
    ani = animation.FuncAnimation(fig, update, frames=max_steps, interval=50, blit=True)
    plt.show()
    return ani

if __name__ == "__main__":
    create_animation()