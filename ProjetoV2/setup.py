from setuptools import setup, find_packages

setup(
    name="tcc-project",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "sionna-rt",
        "mitsuba",
        "drjit",
        "tensorflow==2.15.0",
        "numpy>=1.26",
        "pandas>=2.2",
        "matplotlib>=3.9",
        "scikit-learn>=1.5",
    ],
    python_requires=">=3.10,<3.12",
)
