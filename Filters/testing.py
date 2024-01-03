import sys
parts = sys.path[0].split('/') 
desired_path = '/'.join(parts[:5])
sys.path.append(desired_path)
from my_config import get_config

from edge_detection import *
from image_processing import *
from basics import *
from fourier import *

def test_filters(img):
    img1 = img.copy()
    img2 = apply_sharpen_effect(img,0)
    img3 = apply_sharpen_effect(img,1)
    img4 = apply_single_channel(img, "r")
    img4 = apply_thresholding_rgb(img4)
    img5 = apply_grayscale_conversion(img)
    img5 = apply_in_range(img5)
    img6 = apply_in_range(img)
    print((img5 == img6).all())
    print((img6 == img6).all())


    # show_images(3, img1, img2, img3,img4, img5, img6)
    arrays = [
        (img1, "sha"),
        (img2, "pha"),
        (img3, "real"),

        (img4, "lmag"),
        (img5, "lpha"),
        (img6, "lreal"),
    ]
    plot_2d_arrays(arrays)

def test_fourier(img):
    gray_img = np.array([
        [ 1000, 2000],
        [ 3000, 4000]
        ])
    gray_img = turn_gray_scale(img)
    
    fft_result = fft2d(gray_img)
    print(fft_result)
    mag = fft_part(fft_result, "m")
    pha = fft_part(fft_result, "p")
    real = fft_part(fft_result, "r")
    imag = fft_part(fft_result, "i")
    shifted_mag = shift(mag)
    shifted_pha = shift(pha)
    shifted_real = shift(real)
    shifted_imag = shift(imag)

    normalized_mag = normalize_log(mag)
    normalized_pha = normalize_log(shifted_pha)
    normalized_real = normalize_log(shifted_real)
    normalized_imag = normalize_log(shifted_imag)

    linear_mag = normalize_linear(mag)
    linear_pha = normalize_linear(shifted_pha)
    linear_real = normalize_linear(shifted_real)
    linear_imag = normalize_linear(shifted_imag)

    print(linear_mag)
    # show_single_image(normalizeqd_mag)
    
    arrays = [
        (normalized_mag, "mag"),
        (normalized_pha, "pha"),
        (normalized_real, "real"),
        # (normalized_imag, "imag"),

        (linear_mag, "lmag"),
        (linear_pha, "lpha"),
        (linear_real, "lreal"),
        # (linear_imag, "limag"),
    ]
    plot_2d_arrays(arrays)
    
if __name__ == "__main__":
    config = get_config()
    filename = config['Test_image2']
    height, width = None, 256
    img = read_image(filename, height, width)
    test_filters(img)