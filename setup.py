"""Setup script for OpenSG package."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="opensg",
    version="0.1.0",
    author="OpenSG Development Team",
    author_email="opensg@example.com",
    description="Open Source Structural Analysis for Wind Turbine Blades",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/opensg/opensg",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "dolfinx>=0.6.0",
        "mpi4py>=3.0.0",
        "petsc4py>=3.18.0",
        "pyyaml>=5.4.0",
        "basix>=0.6.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "mypy>=0.800.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "numpydoc>=1.1.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
) 