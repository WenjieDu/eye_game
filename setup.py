from setuptools import setup, find_packages

with open('./README.md', encoding='utf-8') as f:
    README = f.read()

setup(
    name="eye_game",
    version="0.1",
    keywords=[
        "eye", "eyeball", "eye tracking",
        "eyeball direction", 'eyeball position', 'gaze direction'
    ],
    description="A python module for determining gaze direction",
    long_description=README,
    long_description_content_type='text/markdown',
    license="MIT Licence",
    url="https://github.com/WenjieDu/eye_game",
    author="Wenjie Du",
    author_email="wenjay.du@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    platforms=["any"],
    install_requires=[
        "face_recognition",
        "opencv-python",
        "pillow",
        "numpy",
    ],
)
