import cv2
import numpy as np


def apply_mosaic_filter(rgb_img, block_size=10):
    h, w, _ = rgb_img.shape

    num_blocks_h = h // block_size
    num_blocks_w = w // block_size

    mosaic_img = np.zeros_like(rgb_img)

    for i in range(num_blocks_h):
        for j in range(num_blocks_w):
            start_h = i * block_size
            end_h = (i + 1) * block_size
            start_w = j * block_size
            end_w = (j + 1) * block_size

            avg_color = np.mean(rgb_img[start_h:end_h, start_w:end_w], axis=(0, 1))

            mosaic_img[start_h:end_h, start_w:end_w] = avg_color

    return mosaic_img

def show_image(max_images_per_row,*imgs):
    valid_imgs = [cv2.cvtColor(img, cv2.COLOR_RGB2BGR) for img in imgs if img is not None]
    if not valid_imgs:
        raise ValueError("No images provided")
    
    rows = []
    for i in range(0, len(valid_imgs), max_images_per_row):
        row_imgs = valid_imgs[i:i + max_images_per_row]
         # For the last row, if there are fewer images than max_images_per_row, pad the row
        while len(row_imgs) < max_images_per_row:
            row_imgs.append(np.zeros_like(row_imgs[0]))
        rows.append(np.hstack(row_imgs))

    combined_img = np.vstack(rows)

    while True:
        cv2.imshow('Compare Images', combined_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()

def read_image(filename, resized_height=None, resized_width=None):
    bgr_img = cv2.imread(filename)
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    if rgb_img is None:
        raise ValueError("Image not found or unable to read")
    
    h, w, _ = rgb_img.shape

    if resized_width is None and resized_height is not None:
        resized_width = int(w * resized_height / h)
    elif resized_width is not None and resized_height is None:
        resized_height = int(h * resized_width / w)
    elif resized_width is None and resized_height is None:
        resized_height, resized_width = h, w

    rgb_img = cv2.resize(rgb_img, (resized_width, resized_height))
    return rgb_img
   