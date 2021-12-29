# from rasa_nlu_examples.scikit import RasaClassifier

# from rasa.model_training import train_nlu


# def test_base_predict():
#     mod = train_nlu(
#         nlu_data="tests/data/nlu/en/nlu.yml",
#         config="tests/configs/bytepair-config.yml",
#         output="models",
#     )
#     clf = RasaClassifier(model_path=f"{mod}")
#     preds = clf.predict(["hello world", "hello there"])
#     assert len(preds) == 2
