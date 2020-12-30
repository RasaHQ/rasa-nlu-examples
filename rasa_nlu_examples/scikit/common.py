import json
import pathlib
import pandas as pd


def nlu_path_to_dataframe(path):
    """
    Converts a single nlu file with intents into a dataframe.

    Usage:

    ```python
    from rasa_nlu_examples.scikit import nlu_path_to_dataframe

    df = nlu_path_to_dataframe("path/to/nlu/nlu.yml")
    ```
    """
    from rasa.nlu.convert import convert_training_data

    data = []
    p = pathlib.Path(path)
    name = p.parts[-1]
    name = name[: name.find(".")]
    convert_training_data(str(p), f"{name}.json", output_format="json", language="en")
    blob = json.loads(pathlib.Path(f"{name}.json").read_text())
    for d in blob["rasa_nlu_data"]["common_examples"]:
        data.append({"text": d["text"], "label": d["intent"]})
    pathlib.Path(f"{name}.json").unlink()
    return pd.DataFrame(data)
