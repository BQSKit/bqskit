from setuptools import find_packages
from setuptools import setup

setup(
    project_urls={
        'Bug Tracker': 'https://github.com/BQSKit/bqskit/issues',
        'Source Code': 'https://github.com/BQSKit/bqskit',
    },
    packages=find_packages(exclude=['tests*', 'examples*']),
)
