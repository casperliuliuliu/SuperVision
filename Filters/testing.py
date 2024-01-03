
    
import sys
parts = sys.path[0].split('/') 
desired_path = '/'.join(parts[:5])
sys.path.append(desired_path)
from my_config import get_config

from edge_detection import *
from image_processing import *
from basics import *
    
if __name__ == "__main__":
    # array = np.array([
    #     [ 0,  0,  0],
    #     [-1,  2, -1],
    #     [ 0,  0,  0]
    #     ],)

    config = get_config()

    filename = config['Test_image2']

    height, width = None, 256
    img = read_image(filename, height, width)

    img1 = img.copy()
    img2 = apply_sharpen_effect(img,0)
    img3 = apply_sharpen_effect(img,1)
    img4 = apply_single_channel(img, "r")
    img4 = apply_thresholding_rgb(img4)
    img5 = apply_grayscale_conversion(img)
    img5 = apply_thresholding(img5)
    img6 = apply_thresholding(img)
    print((img5 == img6).all())
    print((img6 == img6).all())
    # img2 = img4 + img5
    # img = apply_emboss_filter(img)

    show_image(3, img1, img2, img3,img4, img5, img6)