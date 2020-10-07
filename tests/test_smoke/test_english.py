import pytest

from rasa.train import train_nlu

# Take heed! Pytest fails if you use a function that starts with "test"
from rasa.test import test_nlu as run_nlu


english_yml_files = [
    "stanza-tokenizer-config.yml",
    "fasttext-config.yml",
    "printer-config.yml",
    "bytepair-config.yml",
    "gensim-config.yml",
]


@pytest.mark.fasttext
@pytest.mark.parametrize("fp", english_yml_files)
def test_run_train_test_command_english(fp):
    mod = train_nlu(
        nlu_data="tests/data/nlu/en/nlu.md",
        config=f"tests/configs/{fp}",
        output="models",
    )
    run_nlu(model=f"models/{mod}", nlu_data="tests/data/nlu/en/nlu.md")
