import asyncio
from rich.traceback import install
from rich.console import Console
from rasa.model_training import train_nlu
from rasa.model_testing import test_nlu

console = Console()
install()


if __name__ == "__main__":
    english_yml_files = [
        # Sparse Featurizers
        "hashing-config.yml",
        "tfidf-config.yml",
        # Dense Featurizers
        "bytepair-config.yml",
        "gensim-config.yml",
        "fasttext-config.yml",
        # Entity Extractors
        "dateparser-config.yml",
        "flashtext-config.yml",
        # Classifiers
        "sparse-naive-bayes-intent-classifier-config.yml",
        "logistic-regression-intent-classifier-config.yml",
    ]
    for fp in english_yml_files:
        console.log(f"Training: {fp}")
        mod = train_nlu(
            nlu_data="tests/data/nlu/en/nlu.yml",
            config=f"tests/configs/{fp}",
            output="models",
        )
        console.log(f"Attempting run: {mod}")
        asyncio.run(
            test_nlu(
                model=mod, nlu_data="tests/data/nlu/en/nlu.yml", additional_arguments={}
            )
        )
        console.log(f"Done! Moving on!")
