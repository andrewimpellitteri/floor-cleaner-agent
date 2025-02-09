
import numba
import numpy as np

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

