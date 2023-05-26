from setuptools import setup, find_packages


base_packages = [
    "rasa~=3.5.8",
    "bpemb>=0.3.4",
    "gensim~=4.3.1",
    "tensorflow-macos~=2.11.0",
    "pandas~=2.0.1",
]


dev_packages = [
    "flake8>=6.0.0",
    "black=23.3.0",
    "pre-commit>=3.3.2",
    "pytype>=2023.5.24",
    "pytest>=7.3.1",
    "pytest-xdist==3.3.1",
    "mkdocs==1.4.3",
    "mkdocs-material==9.1.14",
    "mkdocstrings==0.21.2",
    "pymdown-extensions>=10.0.1",
    "flake8-print==5.0.0",
    "rich==13.3.5",
]

thai_deps = [
    "pythainlp>=2.3.2",
]

fasttext_deps = [
    "fasttext~=0.9.2",
]

flashtext_deps = [
    "flashtext==2.7",
]

dateparser_deps = [
    "dateparser>=1.0.0",
]

setup(
    name="rasa_nlu_examples",
    version="0.3.0",
    packages=find_packages(exclude=["notebooks", "data"]),
    install_requires=base_packages,
    extras_require={
        "dev": dev_packages,
        "dev-windows": dev_packages + thai_deps,
        "all": dev_packages
        + thai_deps
        + fasttext_deps
        + flashtext_deps
        + dateparser_deps,
        "thai": base_packages + thai_deps,
        "fasttext": base_packages + fasttext_deps,
        "flashtext": base_packages + flashtext_deps,
        "dateparser": base_packages + dateparser_deps,
    },
)
