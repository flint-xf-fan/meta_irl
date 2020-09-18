# setup.py
from setuptools import setup,find_packages

setup(
    name='inverse_rl',
    packages=[package for package in find_packages()
                if package.startswith('inverse_rl')],
    version='0.1.0',
)

