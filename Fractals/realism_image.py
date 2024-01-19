import numpy as np
import matplotlib.pyplot as plt
from noise import pnoise2

# Landscape size
width, height = 800, 800

# Generate a 2D heightmap with Perlin noise
landscape = np.zeros((width, height))

# Perlin noise parameters
scale = 100.0  # Higher scale will make the terrain smoother
octaves = 6    # More octaves will make the landscape more detailed
persistence = 0.5
lacunarity = 2.0

# Generate heightmap
for x in range(width):
    for y in range(height):
        landscape[x][y] = pnoise2(x / scale, 
                                  y / scale, 
                                  octaves=octaves, 
                                  persistence=persistence, 
                                  lacunarity=lacunarity)

# Normalize the landscape to [0, 1]
landscape = (landscape - np.min(landscape)) / (np.max(landscape) - np.min(landscape))

# Plotting
plt.imshow(landscape, cmap='terrain')
plt.colorbar(label='Height')
plt.title("Fractal Landscape")
plt.show()
