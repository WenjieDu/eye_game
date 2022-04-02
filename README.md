<p align="center">
    <a id="EyeGame" href="#EyeGame">
        <img src="https://raw.githubusercontent.com/WenjieDu/eye_game/master/EyeGameLogo.svg?sanitize=true" alt="EyeGame Title" title="EyeGame Title" width="200"/>
    </a>
</p>
<p align="center">
    <b>A Python Module for Parsing Gaze Direction</b>
</p>
<p align="center">
    <a href="https://pypi.org/project/eye-game">
        <img src="https://img.shields.io/pypi/v/eye-game?color=green" />
    </a>
    <a href="https://pepy.tech/project/eye-game">
        <img src="https://static.pepy.tech/personalized-badge/eye-game?period=total&units=international_system&left_color=grey&right_color=blue&left_text=downloads/total" />
    </a>
    <a href="https://pepy.tech/project/eye-game">
        <img src="https://pepy.tech/badge/eye-game/month" />
    </a>
    <a href="https://pepy.tech/project/eye-game">
        <img src="https://pepy.tech/badge/eye-game/week" />
    </a>
</p>

## ❖ Introduction
This module is used to parse human gaze direction. Given an image containing a human face, the module will return its gaze direction.

## ❖ Usage
You need to install it first by running command `pip install eye_game`, then import and use it!

```python
import eye_game

eye_game.get_gaze_direction(image_path)
```

## ❖ Dependencies
```yml
{
    "face_recognition",
    "opencv-python",
    "pillow",
    "numpy"
}
```