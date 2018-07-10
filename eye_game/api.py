from . import functions


def PIL2CV(img_array):
    return functions.PIL2CV(img_array)


def CV2PIL(img_array):
    return functions.CV2PIL(img_array)


def CV2FR(img_array):
    return functions.CV2FR(img_array)


def PIL2FR(img_array):
    return functions.PIL2FR(img_array)


def FR2PIL(img_array):
    return functions.FR2PIL(img_array)


def get_eyeball_direction(cv_img_array):
    try:
        cv_img_array = functions.resize_image(cv_img_array)
        if cv_img_array is not None:
            result = functions.eyeball_direction(cv_img_array)
            if result is not None:
                direction_result = functions.judge_direction(result[0], result[1], result[2], result[3])
                return functions.result_direction(direction_result)
            else:
                return "no face detected"
        else:
            return "no face detected"
    except AttributeError as e:
        print(e)
        return "image read error, please check your image path"
