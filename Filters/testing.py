
    
import sys
parts = sys.path[0].split('/') 
desired_path = '/'.join(parts[:5])
sys.path.append(desired_path)
from my_config import get_config

from edge_detection import *
from image_processing import *

    
if __name__ == "__main__":
    config = get_config()

    filename = config['Test_image2']

    height, width = None, 256
    img = read_image(filename, height, width)

    img0 = img.copy()
    img1 = adaptiveThreshold_filter(img)
    img2 = canny_filter(img)
    img3 = emboss_filter(img)
    img4 = mosaic_filter(img)
    # img = emboss_filter(img)

    show_image(3, img0, img1, img2, img3,img4)