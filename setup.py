from setuptools import setup, find_packages

setup(
    name="predator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "tqdm",
        "torch",
        "transformers",
        "datasets",
        "texthero",
        "sklearn",
    ],
)
