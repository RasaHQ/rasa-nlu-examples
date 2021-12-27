import pytest
import asyncio

from rasa.model_training import train_nlu

# Take heed! Pytest fails if you use a function that starts with "test"
from rasa.model_testing import test_nlu as run_nlu


english_yml_files = [
    "fasttext-config.yml",
    "bytepair-config.yml",
    "gensim-config.yml",
    "dateparser-config.yml",
    "semantic_map-config.yml",
    "lang-detect-ft-config.yml",
    "sparse-naive-bayes-intent-classifier-config.yml",
    "sparse-logistic-regression-intent-classifier-config.yml",
    "flashtext-config.yml",
    "sparse-bytepair-config.yml",
]


@pytest.mark.fasttext
@pytest.mark.parametrize("fp", english_yml_files)
@pytest.mark.asyncio
def test_run_train_test_command_english(fp):
    """
    This smoke test is like running;

    rasa train nlu -u tests/data/nlu/en/nlu.yml --config tests/configs/sparse-bytepair-config.yml --out models
    """
    mod = train_nlu(
        nlu_data="tests/data/nlu/en-yml/nlu.yml",
        config=f"tests/configs/{fp}",
        output="models",
    )
    asyncio.run(
        run_nlu(
            model=mod, nlu_data="tests/data/nlu/en/nlu.yml", additional_arguments={}
        )
    )
