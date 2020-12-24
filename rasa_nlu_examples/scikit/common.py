import json
import pathlib
import pandas as pd


def nlu_folder_to_dataframe(path):
    """
    Converts the data in an NLU folder into a pandas Dataframe

    Arguments:
        path: refers to the `nlu` folder
    """
    from rasa.nlu.convert import convert_training_data

    data = []
    for p in pathlib.Path(path).glob("*.md"):
        name = p.parts[-1]
        name = name[: name.find(".")]
        convert_training_data(
            str(p), f"{name}.json", output_format="json", language="en"
        )
        blob = json.loads(pathlib.Path(f"{name}.json").read_text())
        for d in blob["rasa_nlu_data"]["common_examples"]:
            data.append({"text": d["text"], "label": d["intent"]})
        pathlib.Path(f"{name}.json").unlink()
        return pd.DataFrame(data)
