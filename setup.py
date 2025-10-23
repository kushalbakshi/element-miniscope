#!/usr/bin/env python
from os import path
from setuptools import find_packages, setup
import urllib.request

pkg_name = "element_miniscope"
here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), "r") as f:
    long_description = f.read()

with open(path.join(here, pkg_name, "version.py")) as f:
    exec(f.read())

setup(
    keywords="neuroscience miniscope science datajoint",
    packages=["element_miniscope", "element_miniscope.plotting"],
    scripts=[],
    install_requires=[
        "datajoint>=0.14.4",
        "ipykernel>=6.0.1",
        "ipywidgets",
        "plotly",
        "opencv-python",
        "element-interface @ git+https://github.com/kushalbakshi/element-interface.git",
    ],
    extras_require={
        "elements": [
            "element-animal @ git+https://github.com/datajoint/element-animal.git",
            "element-event @ git+https://github.com/datajoint/element-event.git",
            "element-lab @ git+https://github.com/datajoint/element-lab.git",
            "element-session @ git+https://github.com/datajoint/element-session.git",
        ],
        "tests": ["pytest", "pytest-cov", "shutils"],
    },
)
