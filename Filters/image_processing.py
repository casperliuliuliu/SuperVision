import cv2

def canny_filter(rbg_img, var_a=30, var_b=100):
    cv2.cvtColor(rbg_img, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(rbg_img, var_a, var_b)

    sketch = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return sketch

def emboss_filter():
    pass

def mosaic_filter():
    pass

def adaptiveThreshold_filter():
    pass

def show_image(img):
    while True:
        cv2.imshow('Recording', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
    cv2.destroyAllWindows()

def read_image(filename, resized_height=256):
    img =  cv2.imread(filename)
    h, w, _ = img.shape
    resize_w = int(w * resized_height / h) 
    img = cv2.resize(img, (resize_w,resized_height))

    return img

if __name__ == "__main__":
    filename = "StableDiffusion/DATA/Chill_norman/norman2.jpg"
    img = read_image(filename, 512)

    img = canny_filter(img, 100, 120)

    show_image(img)

   