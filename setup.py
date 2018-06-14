from setuptools import setup, find_packages

NAME = "eye_game"
DESCRIPTION = ""
AUTHOR = "Vanjay Do"
AUTHOR_EMAIL = "vanjaydo@gmail.com"
URL = "https://github.com/VanjayDo/eye_game"
VERSION = "0.0.1"

setup(
    name=NAME,
    version=VERSION,
    keywords=("eye", "eyeball", "eye tracking", "eyeball direction"),
    description="input: an image containing a human face; output: the eyeball direction",
    long_description="judge eyeball direction according to an image containing a human face",
    license="MIT Licence",

    url=URL,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,

    packages=find_packages(),
    include_package_data=True,
    platforms=["any"],
    install_requires=["face_recognition", "numpy", "opencv-python", "pillow"],
)
