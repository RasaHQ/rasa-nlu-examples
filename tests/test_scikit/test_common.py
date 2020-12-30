from rasa_nlu_examples.scikit import nlu_path_to_dataframe


def test_yaml_nlu_equal_path():
    df1 = nlu_path_to_dataframe("tests/data/nlu/en/nlu.md")
    df2 = nlu_path_to_dataframe("tests/data/nlu/en-yml/nlu_converted.yml")
    assert df1.shape == df2.shape
