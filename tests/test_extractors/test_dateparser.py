from typing import Text

import pytest
import datetime as dt
from rasa.shared.nlu.constants import TEXT, ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

from rasa_nlu_examples.extractors import DateparserEntityExtractor

yesterday_str = str(dt.datetime.now() - dt.timedelta(days=1))[:10]
tomorrow_str = str(dt.datetime.now() + dt.timedelta(days=1))[:10]
day_after_tomorrow_str = str(dt.datetime.now() + dt.timedelta(days=2))[:10]


@pytest.mark.parametrize(
    "text, expected, lang",
    [
        ("I want to fly tomorrow", tomorrow_str, None),
        ("I want to fly tomorrow", tomorrow_str, "en"),
        ("ik wil morgen weg vliegen", tomorrow_str, "nl"),
        ("quiero volar mañana", tomorrow_str, "es"),
        ("I wanna fly yesterday", yesterday_str, "en"),
        ("ik wil overmorgen weg vliegen", day_after_tomorrow_str, "nl"),
        ("ich gehe gern ubermorgen", day_after_tomorrow_str, "de"),
    ],
)
def test_process_tomorrow(text: Text, expected: Text, lang: Text):
    """
    This is a basic dateparser test. There is a known issue for "day after tomorrow"
    https://github.com/scrapinghub/dateparser/issues/933
    """
    message = Message(data={TEXT: text})
    entity_extractor = DateparserEntityExtractor(
        {"languages": [lang] if lang else None}
    )
    entity_extractor.process(message)

    print(message.get(ENTITIES))

    parsed_date = message.get(ENTITIES)[0]["parsed_date"][:10]
    assert parsed_date == expected


def test_do_not_overwrite_any_entities():
    message = Message(data={TEXT: "Max lives in Berlin."})
    message.set(ENTITIES, [{"entity": "person", "value": "Max", "start": 0, "end": 3}])

    DateparserEntityExtractor({}).process(message)
    entities = message.get(ENTITIES)
    assert entities == [
        {"entity": "person", "value": "Max", "start": 0, "end": 3},
    ]
