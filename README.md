# eye_game

## Introduction
This module is used to judge the eyeball direction: you input an image containing a human face, although the result may be not accurate, it will return the eyeball direction. 

## Usage
Steps:👇
1. `pip install eye_game`
2. `import eye_game`
3. `eye_game.get_eyeball_direction(image_path)`

### Other APIs
If you have converted a image to opencv numpy array, you can use `eye_game.api.get_eyeball_direction(cv_image_array)` to get the eyeball direction.

## PyPI
[project on PyPI](https://pypi.org/project/eye-game/)