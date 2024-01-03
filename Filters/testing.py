
    
import sys
parts = sys.path[0].split('/') 
desired_path = '/'.join(parts[:5])
sys.path.append(desired_path)
from my_config import get_config

from edge_detection import *
from image_processing import *
from basics import *
    
if __name__ == "__main__":
    config = get_config()

    filename = config['Test_image2']

    height, width = None, 256
    img = read_image(filename, height, width)

    img0 = img.copy()
    img1 = apply_flip(img,0)
    # img1 = img.copy()
    img2 = apply_flip(img,1)
    img3 = apply_invert_colors(img)
    img4 = apply_canny_filter(img)
    img5 = apply_adaptiveThreshold_filter(img)
    # img = apply_emboss_filter(img)

    show_image(3, img0, img1, img2, img3,img4, img5)