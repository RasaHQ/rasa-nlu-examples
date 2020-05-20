from setuptools import setup, find_packages

base_packages = ["rasa==1.10.1", "fasttext==0.9.2", ]

dev_packages = ["flake8>=3.6.0", "pytest==4.0.2", ]


setup(
    name="rasa_nlu_examples",
    version="0.0.1",
    packages=find_packages(exclude=['notebooks']),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages
    }
)
