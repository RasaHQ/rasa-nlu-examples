import subprocess
import pytest


yml_files = [
    "stanza-tokenizer-config.yml",
    "fasttext-config.yml",
    "printer-config.yml",
    "bytepair-config.yml",
    "gensim-config.yml",
    "thai_tokenizer-config.yml",
]


@pytest.mark.parametrize("fp", yml_files)
def test_run_test_command(fp):
    cmd = [
        "rasa",
        "test",
        "nlu",
        "-u",
        "tests/data/nlu.md",
        "--config",
        f"tests/configs/{fp}",
        "--cross-validation",
        "--runs",
        "1",
        "--folds",
        "2",
    ]
    status = subprocess.run(cmd)
    assert status.returncode == 0


@pytest.mark.parametrize("fp", yml_files)
def test_run_train_command(fp):
    cmd = [
        "rasa",
        "train",
        "nlu",
        "-u",
        "tests/data/nlu.md",
        "--config",
        f"tests/configs/{fp}",
    ]
    status = subprocess.run(cmd)
    assert status.returncode == 0
