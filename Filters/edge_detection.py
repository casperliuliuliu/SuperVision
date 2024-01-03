import cv2

def canny_filter(bgr_img, var_a=30, var_b=100):
    cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    canny_edges = cv2.Canny(bgr_img, var_a, var_b)

    canny_edges_bgr = cv2.cvtColor(canny_edges, cv2.COLOR_GRAY2BGR)
    return canny_edges_bgr

def adaptiveThreshold_filter(bgr_img):

    img_gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

    edges = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 2)
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    return edges_bgr