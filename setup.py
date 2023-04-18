from setuptools import setup
import sys

setup(
    name='nav_gym',
    py_modules=['nav_gym'],
    version= '1.0',
    install_requires=[
        'matplotlib',
        'numpy',
        'scipy',
        'pyyaml',
        'pynput',
        'imageio',
        'pathlib'
    ],
    description="2D simulator for mobile robot learning",
    author="Zhaofeng Tian",
)