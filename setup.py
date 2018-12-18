#!/usr/bin/env python
import os
from setuptools import setup

cd = os.path.dirname(__file__)

# read version info
with open(os.path.join(cd, 'torchtable', '__version__.py')) as f:
    exec(f.read())

long_description = open(os.path.join(cd, 'README.rst'), "rt", encoding="utf-8").read()

setup(
    name="torchtable",
    version=VERSION,
    author="Keita Kurita",
    author_email="keita.kurita@gmail.com",
    description="Tools for processing tabular datasets for PyTorch",
    long_description=long_description,
    license="MIT",
    python_requires = ">=3.6",
    keywords = "PyTorch, deep learning, machine learning",
    setup_requires=["pytest", ],
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "torch>=1.0.0",
    ],
)
