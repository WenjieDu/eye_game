from .functions import eyeball_direction, resize_image, judge_direction, result_direction


def get_eyeball_direction(image_path):
    try:
        cv_img_array = resize_image(image_path)
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
