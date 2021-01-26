from setuptools import find_packages
from setuptools import setup

setup(
    packages=find_packages(exclude=['tests*', 'examples*']),
)
