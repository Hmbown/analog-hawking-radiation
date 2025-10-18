from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="analog-hawking-radiation",
    version="0.2.0",
    author="Hunter Bown",
    author_email="hunter@shannonlabs.dev",
    description="A computational framework for simulating analog Hawking radiation in laser-plasma systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/hmbown/analog-hawking-radiation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "nbconvert",
            "ipykernel",
            "jupyter",
        ],
    },
)
