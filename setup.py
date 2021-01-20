from os import path
from setuptools import setup, find_packages


# Use READEME for long description.
root_dir = path.abspath( path.dirname( __file__ ) )
readme_path = path.join( root_dir, "README.md" )

with open( readme_path, encoding = "utf-8" ) as f:
    long_description = f.read()

# use requirements.txt for requirements.
req_path = path.join( root_dir, "requirements.txt" )

with open( req_path ) as f:
    requirements = f.read().split( '\n' )
    requirements.remove( '' )

setup( name = "bqskit",
       version = "0.1",
       description = "Berkeley Quantum Synthesis Toolkit",
       long_description = long_description,
       long_description_content_type = "text/markdown",
       url = "https://github.com/BQSKit/bqskit",
       author = "Ed Younis",
       author_email = "edyounis@lbl.gov",
       classifiers = [
           "Development Status :: 2 - Pre-Alpha",
           "Environment :: Console",
           "Intended Audience :: Developers",
           "Intended Audience :: Education",
           "Intended Audience :: Science/Research",
           "Operating System :: OS Independent",
           "Programming Language :: Python :: 3 :: Only",
           "Programming Language :: Python :: 3.5",
           "Programming Language :: Python :: 3.6",
           "Programming Language :: Python :: 3.7",
           "Programming Language :: Python :: 3.8",
           "Programming Language :: Python :: 3.9",
           "Topic :: Scientific/Engineering",
           "Topic :: Scientific/Engineering :: Mathematics",
           "Topic :: Scientific/Engineering :: Physics",
           "Topic :: Scientific/Engineering :: Quantum Computing",
           "Topic :: Software Development :: Compilers"
       ],
       keywords = "berkeley quantum synthesis toolkit",
       project_urls = {
           "Bug Tracker": "https://github.com/BQSKit/bqskit/issues",
           "Source Code": "https://github.com/BQSKit/bqskit"
       },
       packages = find_packages( exclude = [ "tests*", "examples*" ] ),
       install_requires = requirements,
       python_requires = ">=3.5, <4",
)

