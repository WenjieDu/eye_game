from .api import *
import cv2


def get_gaze_direction(image_path):
    cv_img_array = cv2.imread(image_path)
    return api.get_gaze_direction(cv_img_array)
