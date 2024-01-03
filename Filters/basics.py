import cv2

def apply_flip(rgb_img, HORIZON_FLAG=1):
    flipped_img = cv2.flip(rgb_img, HORIZON_FLAG)
    return flipped_img

def apply_grayscale_conversion(rgb_img):
    grayscale_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    grayscale_img_rgb = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
    return grayscale_img_rgb

def apply_in_range(rgb_img, lower_bound=190, upper_bound=255):
    grayscale_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    grayscale_img = cv2.inRange(grayscale_img, lower_bound, upper_bound)
    grayscale_img_rgb = cv2.cvtColor(grayscale_img, cv2.COLOR_GRAY2RGB)
    return grayscale_img_rgb

def apply_thresholding_rgb(rgb_img):
    rgb_filtered = cv2.inRange(rgb_img, (190, 190, 190), (255, 255, 255))
    return rgb_filtered

def apply_single_channel(rgb_img, channel="r"):
    temp_img = rgb_img.copy()

    if channel == "b":
        temp_img[:, :, 0] = 0  # Set red channel to 0
        temp_img[:, :, 1] = 0  # Set green channel to 0
    elif channel == "g":
        temp_img[:, :, 0] = 0  # Set red channel to 0
        temp_img[:, :, 2] = 0  # Set blue channel to 0
    elif channel == "r":
        temp_img[:, :, 2] = 0  # Set blue channel to 0
        temp_img[:, :, 1] = 0  # Set green channel to 0
    return temp_img

def apply_invert_colors(rgb_img):
    inverted_img = cv2.bitwise_not(rgb_img)
    return inverted_img