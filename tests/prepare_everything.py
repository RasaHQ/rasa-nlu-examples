import fasttext
import pathlib
import stanza

if __name__ == "__main__":
    # This script adds a small fasttext model that is used in testing.
    # Otherwise we might need to download 6Gb files just for unit tests.
    if not pathlib.Path("tests", "data", "custom_fasttext_model.bin").exists():
        model = fasttext.train_unsupervised(
            "tests/data/fasttext-dummy-data.txt", model="cbow", dim=10
        )
        model.save_model("tests/data/custom_fasttext_model.bin")
        print("local model saved for fasttext")
    else:
        print("found local fasttext model")

    # This part of the script will download a stanza model.
    # If there's a more lightweight way of doing this I'd love to hear it.
    stanza.download("en", dir="tests/data/stanza")
