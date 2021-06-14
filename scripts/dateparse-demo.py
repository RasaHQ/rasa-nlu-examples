from rasa.shared.nlu.training_data.message import Message
from rasa_nlu_examples.extractors.dateparser_extractor import DateparserEntityExtractor
from rich import print

for txt in [
    "hello tomorrow, goodbye yesterday",
    "ik wil een pizza bestellen voor morgen",
]:
    msg = Message.build(txt)
    extractor = DateparserEntityExtractor({})
    extractor.process(msg)
    print(msg.as_dict_nlu())

print("lets prefer dates in the future")
for txt in ["i want a pizza thursday", "ik wil donderdag een pizza"]:
    msg = Message.build(txt)
    extractor = DateparserEntityExtractor({"prefer_dates_from": "future"})
    extractor.process(msg)
    print(msg.as_dict_nlu())

print("lets prefer dates in the past")
for txt in ["i want to buy a pizza thursday", "ik wil donderdag een pizza"]:
    msg = Message.build(txt)
    extractor = DateparserEntityExtractor({"prefer_dates_from": "past"})
    extractor.process(msg)
    print(msg.as_dict_nlu())
