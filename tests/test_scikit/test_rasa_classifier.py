from rasa_nlu_examples.scikit import RasaClassifier

from rasa.train import train_nlu


def test_base_predict():
    mod = train_nlu(
        nlu_data="tests/data/nlu/en/nlu.md",
        config="tests/configs/printer-config.yml",
        output="models",
    )
    clf = RasaClassifier(model_path=f"{mod}")
    preds = clf.predict(["hello world", "hello there"])
    assert len(preds) == 2
