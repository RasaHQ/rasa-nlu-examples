"""
The asyncio library (which we need here) does not play nice with pytest. That's
a bummer, because we really need to have smoke tests. That's why we've made this
script. If it runs to completion, we know that the configuration files defined in
tests/configs are working as we expect!
"""

import asyncio
from rich.traceback import install
from rich.console import Console
from rasa.model_training import train_nlu
from rasa.model_testing import test_nlu

console = Console()
install()


if __name__ == "__main__":
    english_yml_files = [
        # Tokenizers
        "spacy-tok.yml",
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
        console.log(f"Moving on!")
    console.log(f"Done.")
