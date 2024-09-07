import matplotlib.pyplot as plt
import numpy as np

# Initialize a 16x16 grid
grid_size = 15
grid = np.zeros((grid_size, grid_size))

# Region 1
for x in range(1, 10):
    for y in range(1, 10):
        grid[x-1, y-1] = 1  # Mark Region 1 with value 1

# Region 2
for x in range(7, 16):
    for y in range(1, 10):
        grid[x-1, y-1] = 2  # Mark Region 4 with value 2


# Region 3
for x in range(7, 16):
    for y in range(7, 16):
        grid[x-1, y-1] = 3  # Mark Region 3 with value 3


# Region 4
for x in range(1, 10):
    for y in range(7, 16):
        grid[x-1, y-1] = 4  # Mark Region 2 with value 4



# Plotting the grid
plt.figure(figsize=(8, 8))
plt.imshow(grid, cmap='tab10', origin='lower')
plt.colorbar(ticks=[1, 2, 3, 4], label='Region Number')
plt.title("Grid with Highlighted Regions")
plt.xlabel("Y-axis (Columns)")
plt.ylabel("X-axis (Rows)")
plt.grid(False)

# Annotate each region on the grid
for i in range(grid_size):
    for j in range(grid_size):
        if grid[i, j] != 0:
            plt.text(j, i, f'{int(grid[i, j])}', ha='center', va='center', color='black')

plt.show()
