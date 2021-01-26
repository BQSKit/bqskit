from os import path
from setuptools import setup, find_packages


# Use READEME for long description.
root_dir = path.abspath( path.dirname( __file__ ) )
readme_path = path.join( root_dir, "README.md" )

with open( readme_path, encoding = "utf-8" ) as f:
    long_description = f.read()

setup( long_description = long_description,
       project_urls = {
           "Bug Tracker": "https://github.com/BQSKit/bqskit/issues",
           "Source Code": "https://github.com/BQSKit/bqskit"
       },
       packages = find_packages( exclude = [ "tests*", "examples*" ] ),
)

