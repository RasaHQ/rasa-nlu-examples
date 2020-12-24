import pytest

from rasa_nlu_examples.scikit import RasaClassifier

from rasa.train import train_nlu


def test_base_predict():
    mod = train_nlu(
        nlu_data="tests/data/nlu/en/nlu.md",
        config=f"tests/configs/printer-config.yml",
        output="models",
    )
    clf = RasaClassifier(model_path=f"models/{mod}")
    clf.class_names_ = [
        "greet",
        "goodbye",
        "out_of_scope",
        "bot_challenge",
        "talk_code",
    ]
    preds = clf.predict(["hello world", "hello there"])
    for p in preds:
        assert p == "greet"
    assert len(preds) == 2

    pred_proba = clf.predict_proba(["hello world", "hello there"])
    assert pred_proba.shape[0] == 2
