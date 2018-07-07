from .functions import *


def PIL2CV(img_array):
    return PIL2CV(img_array)


def CV2PIL(img_array):
    return CV2PIL(img_array)


def CV2FR(img_array):
    return CV2FR(img_array)


def PIL2FR(img_array):
    return PIL2FR(img_array)


def FR2PIL(img_array):
    return FR2PIL(img_array)


def get_eyeball_direction(cv_img_array):
    try:
        cv_img_array = resize_image(cv_img_array)
        if cv_img_array is not None:
            result = eyeball_direction(cv_img_array)
            if result is not None:
                direction_result = judge_direction(result[0], result[1], result[2], result[3])
                return result_direction(direction_result)
            else:
                return "no face detected"
        else:
            return "no face detected"
    except AttributeError as e:
        print(e)
        return "image read error, please check your image path"
