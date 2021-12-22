import pytest

from rasa.model_training import train_nlu

# Take heed! Pytest fails if you use a function that starts with "test"
from rasa.model_testing import test_nlu as run_nlu

non_english_files = [("thai-tokenizer-config.yml", "tests/data/nlu/th/nlu-th.yml")]


@pytest.mark.parametrize("fp,nlu", non_english_files)
def test_run_train_test_command_non_english(fp, nlu):
    mod = train_nlu(nlu_data=nlu, config=f"tests/configs/{fp}", output="models")
    run_nlu(model=f"models/{mod}", nlu_data=nlu)
