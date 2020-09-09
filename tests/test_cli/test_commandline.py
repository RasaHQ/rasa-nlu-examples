import subprocess
import pytest


english_yml_files = [
    "stanza-tokenizer-config.yml",
    "fasttext-config.yml",
    "printer-config.yml",
    "bytepair-config.yml",
    "gensim-config.yml",
]

non_english_files = [("thai-tokenizer-config.yml", "tests/data/nlu/th/nlu-th.md")]


@pytest.mark.parametrize("fp", english_yml_files)
def test_run_test_command_english(fp):
    cmd = [
        "rasa",
        "test",
        "nlu",
        "-u",
        "tests/data/nlu/en/nlu.md",
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


@pytest.mark.parametrize("fp", english_yml_files)
def test_run_train_command_english(fp):
    cmd = [
        "rasa",
        "train",
        "nlu",
        "-u",
        "tests/data/nlu/en/nlu.md",
        "--config",
        f"tests/configs/{fp}",
    ]
    status = subprocess.run(cmd)
    assert status.returncode == 0


@pytest.mark.parametrize("yml,nlu", non_english_files)
def test_run_test_command_non_english(yml, nlu):
    cmd = [
        "rasa",
        "test",
        "nlu",
        "-u",
        nlu,
        "--config",
        f"tests/configs/{yml}",
        "--cross-validation",
        "--runs",
        "1",
        "--folds",
        "2",
    ]
    status = subprocess.run(cmd)
    assert status.returncode == 0


@pytest.mark.parametrize("yml,nlu", non_english_files)
def test_run_train_command_non_english(yml, nlu):
    cmd = [
        "rasa",
        "train",
        "nlu",
        "-u",
        nlu,
        "--config",
        f"tests/configs/{yml}",
    ]
    status = subprocess.run(cmd)
    assert status.returncode == 0
