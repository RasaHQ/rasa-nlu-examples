# import pandas as pd
# from rasa_nlu_examples.scikit import nlu_path_to_dataframe, dataframe_to_nlu_file


# def test_yaml_both_ways(tmp_path):
#     df = pd.DataFrame(
#         [
#             {"text": "i really really like this", "intent": "positive"},
#             {"text": "i enjoy this", "intent": "positive"},
#             {"text": "this is not my thing", "intent": "negative"},
#         ]
#     )
#     path = f"{tmp_path}/temp_nlu.yml"
#     dataframe_to_nlu_file(df, write_path=path)
#     df_read = nlu_path_to_dataframe(path)
#     assert len(df_read) == 3
