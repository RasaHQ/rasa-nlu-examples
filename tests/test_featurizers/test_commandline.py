import subprocess

import pytest

filepaths = ["tests/configs/fasttext-config.yml"]


@pytest.mark.parametrize("fp", filepaths)
def test_run_train_command(fp):
    cmd = [
        "rasa",
        "test",
        "nlu",
        "-u",
        "tests/data/nlu.md",
        "--config",
        fp,
        "--cross-validation",
        "--runs",
        "1",
        "--folds",
        "2",
    ]
    status = subprocess.run(cmd)
    assert status.returncode == 0
