#!/usr/bin/env python
"""
setup.py: REAL Pipetting project module repository
"""
import os
import sys
import time
from setuptools import setup, find_packages

__author__ = "William Ballengee, Huat Chiang, Ahmed Eshaq, Samantha Tetef"
__copyright__ = "LICENSE.txt"
__version__ = "0.1.0"

setup(
    name='realpy',
    version=__version__,
    url='https://github.com/REAL-Pipetting/REinforced-Automaton-Learning-REAL-Pipetting',
    author=__author__,
    classifiers=[
        'Development Status :: 1 - Planning',
        'Environment :: Console',
        'Operating System :: OS Independant',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering'
    ],
    license=__copyright__,
    description='Reinforcement learning agent for planning pipetting experiments executed by the OT2 robots.',
    keywords=[
        'GPBUCB',
        'GPUCB',
        'genetic_algorithm',
        'reinforcement_learning',
    ],
    packages=find_packages(exclude="tests"),
    python_requires='>=3.6',
    install_requires=[
        'numpy>=1.19.1'
        'smt>=0.9.0'
    ]
)
