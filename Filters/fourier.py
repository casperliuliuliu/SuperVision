import numpy as np
import matplotlib.pyplot as plt
import cv2
def main():
    # Example 2D array (can be an image)
    data = np.random.rand(256, 256)  # Generating a random 2D array

    # Compute the 2D FFT
    fft_result = np.fft.fft2(data)

    # Compute the 2D inverse FFT
    ifft_result = np.fft.ifft2(fft_result)

    # Plotting the results
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(data, cmap='gray')
    plt.title('Original Data')

    plt.subplot(1, 3, 2)
    plt.imshow(np.log(np.abs(fft_result)), cmap='gray')  # Log scale for better visibility
    plt.title('FFT of Data')

    plt.subplot(1, 3, 3)
    plt.imshow(np.abs(ifft_result), cmap='gray')  # Absolute value for plotting
    plt.title('Recovered Data after IFFT')

    plt.show()

    print(ifft_result == data)
def turn_gray_scale(rgb_img):
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    return gray_img

def fft2d(img):
    fft_result = np.fft.fft2(img)

    return fft_result

def ifft2d(fft_result):
    ifft_result = np.fft.ifft2(fft_result)
    return ifft_result

def fft_part(fft_result, option="m"):
    if option == "m":
        return np.abs(fft_result)
    elif option == "p":
        return np.angle(fft_result)
    elif option == "r":
        return fft_result.real
    elif option == "i":
        return fft_result.imag

def normalize_log(fft_result):
    
    # Apply logarithmic scaling (adding a small value to avoid log(0))
    log_magnitude = np.log(fft_result + 1e-10)
    
    # Normalize to [0, 255]
    norm_magnitude = 255 * (log_magnitude - np.min(log_magnitude)) / (np.max(log_magnitude) - np.min(log_magnitude))
    
    return norm_magnitude.astype(np.uint8)

def shift(fft_result):
    min_value = np.min(fft_result)
    if min_value < 0:
        return fft_result - np.min(fft_result) + 1e-10
    return fft_result

def normalize_linear(fft_result):
     # Normalize to [0, 255]
    norm_magnitude = 255 * (fft_result - np.min(fft_result)) / (np.max(fft_result) - np.min(fft_result))
    
    return norm_magnitude.astype(np.uint8)

def plot_2d_arrays(arrays):
    # Number of plots
    n = len(arrays)

    # Calculate grid size (simple square layout)
    grid_size = int(np.ceil(np.sqrt(n)))

    # Create subplots
    fig, axes = plt.subplots(int(np.ceil(n/grid_size)), grid_size, figsize=(10, 10))

    # Flatten axes array for easy iteration
    axes_flat = axes.flatten()

    # Plot each array
    for i, (array, title) in enumerate(arrays):
        if i < n:
            ax = axes_flat[i]
            im = ax.imshow(array, cmap='gray')
            ax.set_title(title)
            ax.axis('off')
            fig.colorbar(im, ax=ax)
        else:
            axes_flat[i].axis('off')  # Turn off unused subplots

    # Adjust layout
    plt.tight_layout()
    plt.show()