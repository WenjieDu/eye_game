from .api import *
import cv2


def get_eyeball_direction(image_path):
    cv_img_array = cv2.imread(image_path)
    return api.get_eyeball_direction(cv_img_array)
