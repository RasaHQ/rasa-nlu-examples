import pathlib

from rasa.train import train_nlu
from rasa.cli.utils import get_validated_path
from rasa.model import get_model, get_model_subdirectories
from rasa.core.interpreter import RasaNLUInterpreter
from rasa.shared.nlu.training_data.loading import load_data


def load_interpreter(model_path):
    model_path = pathlib.Path(model_path)
    model_path_val = get_validated_path(str(model_path), "model")
    model_path = get_model(model_path_val)
    _, nlu_model = get_model_subdirectories(model_path)
    return RasaNLUInterpreter(nlu_model)


def test_predict():
    NLU_DATA_PATH = "tests/data/nlu/en/nlu.md"
    model_path = train_nlu(
        nlu_data=NLU_DATA_PATH,
        config="tests/configs/sparse-sklearn-intent-classifier-config.yml",
        output="models",
    )

    interpreter = load_interpreter(model_path)

    # Get features from the pipeline and prepare data in the format sklearn
    # expects.
    training_data = load_data(NLU_DATA_PATH)
    for example in training_data.intent_examples:
        interpreter.featurize_message(example)
    model = interpreter.interpreter.pipeline[-1]
    X, y = model.prepare_data(training_data)

    # Fit the equivalent sklearn classifier.
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB(alpha=0.1, binarize=0.0, fit_prior=True)
    clf.fit(X, y)

    # Check that predictions agree.
    assert (clf.predict_proba(X) == model.predict_prob(X)).all()
    assert (clf.predict(X) == model.predict(X)[0][:, 0]).all()
