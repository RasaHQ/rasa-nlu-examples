import stanza

if __name__ == "__main__":
    # This part of the script will download a stanza model.
    # If there's a more lightweight way of doing this I'd love to hear it.
    stanza.download("en", model_dir="tests/data/stanza")
