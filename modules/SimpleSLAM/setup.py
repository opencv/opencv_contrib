# setup.py
from setuptools import setup, find_packages

setup(
    name        = "opencv_simple_slam",
    version     = "0.1.0",
    description = "A simple feature-based SLAM using OpenCV and LightGlue",
    packages    = find_packages(),  # this will pick up your `slam/` package
    install_requires=[
        "opencv-python",
        "numpy",
        "lz4",
        "tqdm",
        "torch",          # if you intend to support LightGlue
        "lightglue @ git+https://github.com/cvg/LightGlue.git",
    ],
)

