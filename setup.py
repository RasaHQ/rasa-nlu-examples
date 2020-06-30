from setuptools import setup, find_packages

base_packages = ["rasa==1.10.1", "fasttext==0.9.2", "bpemb==0.3.0"]

dev_packages = [
    "flake8>=3.6.0",
    "pytest==4.0.2",
    "pytest-xdist==1.32.0",
    "mkdocs==1.1",
    "mkdocs-material==4.6.3",
    "mkdocstrings==0.8.0",
]


setup(
    name="rasa_nlu_examples",
    version="0.1.0",
    packages=find_packages(exclude=["notebooks"]),
    install_requires=base_packages,
    extras_require={"dev": dev_packages},
)
