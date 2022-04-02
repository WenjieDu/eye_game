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


def get_gaze_direction(cv_img_array):
    try:
        result = functions.image_preprocessing(cv_img_array)
        assert result is not None, "No face detected."
        result = functions.gaze_direction(result["cv_img_array"], result["eye_locations"])
        direction_result = functions.determine_direction(result[0], result[1], result[2], result[3])
        return functions.interpret_result_direction(direction_result)

    except AttributeError as e:
        print(e)
        return "image read error, please check your image path"
