from rich.console import Console
import fasttext
import pathlib

console = Console()

if __name__ == "__main__":
    # This script adds a small fasttext model that is used in testing.
    # Otherwise we might need to download 6Gb files just for unit tests.
    if not pathlib.Path(
        "tests", "data", "fasttext", "custom_fasttext_model.bin"
    ).exists():
        model = fasttext.train_unsupervised(
            "tests/data/fasttext/fasttext-dummy-data.txt", model="cbow", dim=10
        )
        model.save_model("tests/data/fasttext/custom_fasttext_model.bin")
        console.log("local model saved for fasttext")
    else:
        console.log("found local fasttext model")
