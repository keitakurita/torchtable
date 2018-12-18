#!/usr/bin/env python
import os
import io
import re
from setuptools import setup, find_packages

cd = os.path.dirname(__file__)

VERSION = "0.1.0"
long_description = open(os.path.join(cd, 'README.rst'), "rt", encoding="utf-8").read()

setup(
    name="torchtable",
    version=VERSION,
    author="Keita Kurita",
    author_email="keita.kurita@gmail.com",
    description="Tools for processing tabular datasets for PyTorch",
    long_description=long_description,
    license="MIT",
)
