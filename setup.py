"""The BQSKit Quantum Compiler Framework Setup File."""
from __future__ import annotations

import os

from setuptools import find_packages
from setuptools import setup

root_dir_path = os.path.abspath(os.path.dirname(__file__))
pkg_dir_path = os.path.join(root_dir_path, 'bqskit')
readme_path = os.path.join(root_dir_path, 'README.md')
version_path = os.path.join(pkg_dir_path, 'version.py')

# Load Version Number
with open(version_path) as version_file:
    exec(version_file.read())

# Load Readme
with open(readme_path) as readme_file:
    long_description = readme_file.read()

setup(
    name='bqskit',
    version=__version__,  # type: ignore # noqa # Defined in version.py loaded above
    description='Berkeley Quantum Synthesis Toolkit',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/BQSKit/bqskit',
    author='LBNL - BQSKit developers',
    author_email='edyounis@lbl.gov',
    license='BSD 3-Clause License',
    license_files=['LICENSE'],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: MacOS',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Compilers',
        'Typing :: Typed',
    ],
    keywords=[
        'berkeley',
        'quantum',
        'synthesis',
        'toolkit',
        'partitioning',
    ],
    project_urls={
        'Bug Tracker': 'https://github.com/BQSKit/bqskit/issues',
        'Source Code': 'https://github.com/BQSKit/bqskit',
        'Documentation': 'https://bqskit.readthedocs.io/en/latest',
    },
    packages=find_packages(exclude=['examples*', 'test*']),
    install_requires=[
        'bqskitrs>=0.4.1',
        'lark-parser',
        'numpy>=1.22.0',
        'scipy>=1.8.0',
        'typing-extensions>=4.0.0',
        'dill>=0.3.8',
    ],
    python_requires='>=3.8, <4',
    entry_points={
        'console_scripts': [
            'bqskit-server = bqskit.runtime.detached:start_server',
            'bqskit-manager = bqskit.runtime.manager:start_manager',
            'bqskit-worker = bqskit.runtime.worker:start_worker_rank',
        ],
    },
    extras_require={
        'dev': [
            'hypothesis[zoneinfo]',
            'mypy',
            'pre-commit',
            'psutil',
            'pytest',
            'tox',
        ],
    },
)
