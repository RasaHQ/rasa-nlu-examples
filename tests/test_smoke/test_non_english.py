import pytest

from rasa.train import train_nlu
from rasa.test import test_nlu


non_english_files = [("thai-tokenizer-config.yml", "tests/data/nlu/th/nlu-th.md")]


@pytest.mark.parametrize("fp,nlu", non_english_files)
def test_run_train_test_command_non_english(fp, nlu):
    mod = train_nlu(nlu_data=nlu, config=f"tests/configs/{fp}", output="models")
    test_nlu(model=f"models/{mod}", nlu_data=nlu)
