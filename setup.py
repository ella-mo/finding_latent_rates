"""Setup file for data_functions package."""
from setuptools import setup, find_packages

setup(
    name="data_functions",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "h5py",
        "scipy",
    ],
)

