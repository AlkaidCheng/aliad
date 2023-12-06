import os
import re
import sys
import glob
import builtins
from contextlib import contextmanager

import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext as _build_ext
from setuptools.command.install import install as _install

with open("README.md", "r") as fh:
    long_description = fh.read()

PACKAGENAME = "aliad"
VERSIONFILE = f"{PACKAGENAME}/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))    

setup(
    name=PACKAGENAME, # Replace with your own username
    version=verstr,
    author="Alkaid Cheng",
    author_email="chi.lung.cheng@cern.ch",
    description="A library for anomaly detection.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    package_data={PACKAGENAME: []},
    exclude_package_data={PACKAGENAME: []},
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
          'numpy',
          'pandas',
          'matplotlib',
          'click',
          'quickstats'
      ],
    scripts=[f'bin/{PACKAGENAME}'],
    python_requires='>=3.8',
)
