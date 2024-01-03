import cv2
import numpy as np

def apply_canny_filter(rgb_img, var_a=30, var_b=100):
    canny_edges = cv2.Canny(rgb_img, var_a, var_b)

    canny_edges_rgb = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2RGB)
    return canny_edges_rgb

def apply_adaptiveThreshold_filter(rgb_img):
    img_gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)

    edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    return edges_rgb

def apply_emboss_filter(rgb_img, kernel_option=0, intensity=1):
    embossing_kernel_array = np.array([
        [
        [0, -1, -1],
        [1,  0, -1],
        [1,  1,  0]
        ],

        [
        [1,  1,  0],
        [1,  0, -1],
        [0, -1, -1]
        ],

        [
        [ 0,  1,  1],
        [-1,  0,  1],
        [-1, -1,  0]
        ],

        [
        [-1, -1,  0],
        [-1,  0,  1],
        [ 0,  1,  1]
        ],

        ])
    embossing_kernel = embossing_kernel_array[kernel_option] * intensity
    embossed_img = cv2.filter2D(rgb_img, cv2.CV_8U, embossing_kernel)
    return embossed_img

def apply_sharpen_effect(img, kernel_option=0, intensity=1):

    sharpened_kernel_array = np.array([
        [
        [-1, -1, -1],
        [-1,  8, -1],
        [-1, -1, -1]
        ],

        [
        [ 0, -1,  0],
        [-1,  4, -1],
        [ 0, -1,  0]
        ],

        [
        [ 0,  0,  0],
        [-1,  2, -1],
        [ 0,  0,  0]
        ],

        [
        [ 0, -1,  0],
        [ 0,  2,  0],
        [ 0, -1,  0]
        ],
        ])
    
    sharpend_kernel = sharpened_kernel_array[kernel_option] * intensity
    sharpened_img = cv2.filter2D(img, cv2.CV_8U, sharpend_kernel)
    return sharpened_img