import numpy as np
import matplotlib.pyplot as plt

# Set the size of the plot
width, height = 800, 800

# Define the range in the complex plane to visualize
x_min, x_max = -2, 1
y_min, y_max = -1.5, 1.5

# Create a grid of c-values (complex numbers)
x, y = np.linspace(x_min, x_max, width), np.linspace(y_min, y_max, height)
C = np.array(np.meshgrid(x, y)).T.reshape(-1, 2)
C = C[:, 0] + 1j * C[:, 1]

# Mandelbrot set parameters
max_iter = 100
escape_radius = 10

# Initialize an array to store the results
output = np.zeros(C.shape, dtype=int)

# Compute the Mandelbrot set
for i, c in enumerate(C):
    z = 0
    for n in range(max_iter):
        z = z + c
        if abs(z) > escape_radius:
            output[i] = n
            break

# Reshape the results to a 2D array for plotting
output = output.reshape((width, height))

# Plotting
plt.imshow(output.T, extent=[x_min, x_max, y_min, y_max], cmap='hot')
plt.colorbar()
plt.title("Mandelbrot Set")
plt.xlabel("Re")
plt.ylabel("Im")
plt.show()
