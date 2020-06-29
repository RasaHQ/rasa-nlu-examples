import subprocess

import pytest


@pytest.mark.parametrize("fp", ["fasttext-config.yml", "printer-config.yml"])
def test_run_train_command(fp):
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
