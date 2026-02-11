from setuptools import setup, find_packages

setup(
    name="Suika",
    version="0.1.0",
    description="Suika module",
    packages=find_packages(include=["Suika", "Suika.*"]),
    install_requires=[],  # manage dependencies via requirements.txt if preferred
    python_requires=">=3.10",
)
