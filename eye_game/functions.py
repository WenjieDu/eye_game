import math

import cv2
import face_recognition as fr
import numpy as np
from PIL import Image


def PIL2CV(img_array):
    """ Convert PIL image array to OpenCV image array
    :param img_array: PIL image object
    :return: OpenCV image object
    """
    return cv2.cvtColor(np.asarray(img_array), cv2.COLOR_RGB2BGR)


def CV2PIL(img_array):
    """ Convert OpenCV image array to PIL image array
    :param img_array: OpenCV image object
    :return: PIL image object
    """
    return Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))


def CV2FR(img_array):
    """ Convert OpenCV image array to lib face_recognition image array
    :param img_array: OpenCV image object
    :return: face_recognition image object
    """
    return cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)


def PIL2FR(img_array):
    """ Convert PIL image array to lib face_recognition image array
    :param img_array: PIL image object
    :return: face_recognition image object
    """
    return np.array(img_array)


def FR2PIL(img_array):
    """ Convert lib face_recognition image array to PIL image array
    :param img_array: face_recognition image object
    :return: PIL image object
    """
    return Image.fromarray(np.uint8(img_array))


def get_face_position(fr_img_array):
    """ Get the face location in the image
    :param fr_img_array: face_recognition image array
    :return: A list of tuples of found face locations in css order (top, right, bottom, left), like [(171, 409, 439, 141)]
    """
    face_location = fr.api.face_locations(fr_img_array)
    return face_location


def get_eye_positions(fr_img_array, face_location):
    """
    :param fr_img_array: An image (as a numpy array)
    :param face_location: face location
    :return: dict include eyes locations
    """
    face_landmarks = fr.api.face_landmarks(fr_img_array, face_locations=face_location)
    if face_landmarks:
        left_eye = face_landmarks[0]["left_eye"]
        right_eye = face_landmarks[0]["right_eye"]
        eye_locations = {"left_eye": left_eye, "right_eye": right_eye}
        return eye_locations
    else:
        return None


def image_preprocessing(cv_img_array):
    """ Image preprocessing. Adjust image size.
    :param cv_img_array: image array returned by cv2.imread
    :return: a dict containing OpenCV image array and the eye positions
    """
    if cv_img_array.shape[1] > 500:
        cv_img_array = cv2.resize(cv_img_array, (0, 0), fx=0.4, fy=0.4)
    fr_img_array = CV2FR(cv_img_array)
    face_location = get_face_position(fr_img_array)
    if face_location:
        cv_img_array = cv_img_array[face_location[0][0]:face_location[0][2], face_location[0][3]:face_location[0][1]]
        eye_locations = get_eye_positions(fr_img_array, face_location)

        for i in range(6):
            height = eye_locations["left_eye"][i][0] - face_location[0][3]
            width = eye_locations["left_eye"][i][1] - face_location[0][0]
            eye_locations["left_eye"][i] = (height, width)

        for i in range(6):
            height = eye_locations["right_eye"][i][0] - face_location[0][3]
            width = eye_locations["right_eye"][i][1] - face_location[0][0]
            eye_locations["right_eye"][i] = (height, width)

        result = {"cv_img_array": cv_img_array, "eye_locations": eye_locations}
        return result
    else:
        return None


def get_pupil_position(width, height, center):
    """ Parse the pupil position
    :param width: the width of the eye socket
    :param height: the height of the eye socket
    :param center: the center position of the eye
    :return: the pupil position
    """
    width_trisection = int(width / 3)
    height_trisection = int(height / 3)

    if (center["x"] < width_trisection) & (center["y"] < height_trisection):
        return 0
    elif (center["x"] > width_trisection * 2) & (center["y"] < height_trisection):
        return 2
    elif center["y"] < height_trisection:
        return 1
    elif (center["x"] < width_trisection) & (center["y"] > height_trisection * 2):
        return 6
    elif (center["x"] > width_trisection * 2) & (center["y"] > height_trisection * 2):
        return 8
    elif center["y"] > height_trisection * 2:
        return 7
    elif center["x"] < width_trisection:
        return 3
    elif center["x"] > width_trisection * 2:
        return 5
    else:
        return 4


def rect_eye(eye_landmarks):
    """ Get the rectangular box of the eye from the eye position produced by function face_recognition.face_landmarks
    :param eye_landmarks: the eye position, like [(193, 251), (205, 242), (223, 244), (235, 258), (220, 261), (203, 259)]
    :return: A dict containing the coordinate of the upper left corner (x1,y1) and the coordinate of the lower right corner (x2,y2)
    """
    width = eye_landmarks[3][0] - eye_landmarks[0][0]
    height = int(
        (eye_landmarks[4][1] + eye_landmarks[5][1] - eye_landmarks[1][1] - eye_landmarks[2][1]) / 2
    )
    x = eye_landmarks[0][0]
    y = int(
        (eye_landmarks[1][1] + eye_landmarks[2][1]) / 2
    )
    return {"x1": x, "y1": y, "x2": x + width, "y2": y + height}


def gaze_direction(cv_img_array, eye_locations):
    """
    :param cv_img_array: OpenCV image array
    :param eye_locations: coordinates of eyes
    """
    cv_img_array = np.uint8(np.clip(1.1 * cv_img_array + 30, 0, 255))
    left_coordinate = rect_eye(eye_locations["left_eye"])
    right_coordinate = rect_eye(eye_locations["right_eye"])

    left_eyeball = get_eyeball_position(cv_img_array, left_coordinate)
    left_percent = left_eyeball["percent"]
    left_pupil_position = left_eyeball["pupil_position"]

    right_eyeball = get_eyeball_position(cv_img_array, right_coordinate)
    right_percent = right_eyeball["percent"]
    right_pupil_position = right_eyeball["pupil_position"]

    return [left_pupil_position, left_percent, right_pupil_position, right_percent]


def get_eyeball_position(cv_image_array, eye_coordinate):
    """ Get the position of eyeball in the rectangular box of the eye
    :param cv_image_array: OpenCV image array from cv2.imread
    :param eye_coordinate: the coordinate of the rectangular box of the eye, from function rect_eye
    :return: percent: the percentage of pixels used to determine the pupil position of the eye among the total pixels
             eyeball_center: the coordinate of the eyeball center
             pupil_position: the position of pupil
    """
    eyeball_roi = cv_image_array[eye_coordinate["y1"]:eye_coordinate["y2"], eye_coordinate["x1"]:eye_coordinate["x2"]]
    eyeball_roi = cv2.cvtColor(eyeball_roi, cv2.COLOR_BGR2GRAY)
    gray_val_total = 0
    gray_val_min = 255

    for i in range(eyeball_roi.shape[0]):  # height
        for j in range(eyeball_roi.shape[1]):  # width
            gray_val_total += eyeball_roi[i][j]
            if gray_val_min > eyeball_roi[i][j]:
                gray_val_min = eyeball_roi[i][j]

    gray_val_avg = int(gray_val_total / (eyeball_roi.shape[0] * eyeball_roi.shape[1]))

    eyeball_center_x = 0
    eyeball_center_y = 0
    counter = 0

    if ((gray_val_avg * 2) / 3) > gray_val_min:
        for i in range(eyeball_roi.shape[0]):  # height
            for j in range(eyeball_roi.shape[1]):  # width
                if eyeball_roi[i][j] <= ((gray_val_avg * 2) / 3):
                    eyeball_center_y += i
                    eyeball_center_x += j
                    counter += 1

    else:
        for i in range(eyeball_roi.shape[0]):  # height
            for j in range(eyeball_roi.shape[1]):  # width
                if eyeball_roi[i][j] <= gray_val_min:
                    eyeball_center_y += i
                    eyeball_center_x += j
                    counter += 1

    percent = counter / (eyeball_roi.shape[0] * eyeball_roi.shape[1])

    eyeball_center_x = math.ceil(eyeball_center_x / counter)
    eyeball_center_y = math.ceil(eyeball_center_y / counter)
    eyeball_center = {"x": eyeball_center_x, "y": eyeball_center_y}
    pupil_position = get_pupil_position(eyeball_roi.shape[1], eyeball_roi.shape[0], eyeball_center)
    return {"percent": percent, "eyeball_center": eyeball_center, "pupil_position": pupil_position}


def determine_direction(left_result, left_percent, right_result, right_percent):
    """ Determine the gaze direction of eyes
    :param left_result: the pupil position of the left eye
    :param left_percent: the percentage of pixels used to determine the pupil position of the left eye among the total pixels
    :param right_result: the pupil position of the right eye
    :param right_percent: the percentage of pixels used to determine the pupil position of the right eye among the total pixels
    :return: determined gaze direction, single result
    """
    if ((left_result == 4) & (right_result != 4)) | ((right_result == 4) & (left_result != 4)):
        return right_result if left_result == 4 else left_result
    elif ((left_result == 1) & ((right_result == 0) | (right_result == 2))) | (
            (right_result == 1) & ((left_result == 0) | (left_result == 2))):
        return right_result if left_result == 1 else left_result
    elif ((left_result == 7) & ((right_result == 6) | (right_result == 8))) | (
            (right_result == 7) & ((left_result == 6) | (left_result == 8))):
        return right_result if left_result == 7 else left_result
    elif ((left_result == 3) & ((right_result == 0) | (right_result == 6))) | (
            (right_result == 3) & ((left_result == 0) | (left_result == 6))):
        return right_result if left_result == 3 else left_result
    elif ((left_result == 5) & ((right_result == 2) | (right_result == 8))) | (
            (right_result == 5) & ((left_result == 2) | (left_result == 8))):
        return right_result if left_result == 7 else left_result
    else:
        return right_result if left_percent < right_percent else left_result


def interpret_result_direction(result):
    """ Interpret the number result into the readable direction
    :param result: the direction result in the numeric code
    :return: the readable direction, like 'right'
    """
    if result == 0:
        return "upper left"
    elif result == 1:
        return "up"
    elif result == 2:
        return "upper right"
    elif result == 3:
        return "left"
    elif result == 4:
        return "center"
    elif result == 5:
        return "right"
    elif result == 6:
        return "lower left"
    elif result == 7:
        return "down"
    else:
        return "lower right"
