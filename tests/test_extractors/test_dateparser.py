import pathlib
import datetime as dt
from typing import Text, Optional

import pytest
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.shared.nlu.constants import TEXT, ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa_nlu_examples.extractors.dateparser_extractor import DateparserEntityExtractor

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
def test_process_tomorrow(text: Text, expected: Text, lang: Optional[Text], tmpdir):
    """
    This is a basic dateparser test. There is a known issue for "day after tomorrow"
    https://github.com/scrapinghub/dateparser/issues/933
    """
    message = Message(data={TEXT: text})
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    node_resource = Resource("sparse_feat")
    context = ExecutionContext(node_storage, node_resource)
    entity_extractor = DateparserEntityExtractor(
        config={"languages": [lang] if lang else None},
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
    )
    entity_extractor.process([message])

    parsed_date = message.get(ENTITIES)[0]["parsed_date"][:10]
    assert parsed_date == expected


def test_do_not_overwrite_any_entities(tmpdir):
    message = Message(data={TEXT: "Max lives in Berlin."})
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    node_resource = Resource("sparse_feat")
    context = ExecutionContext(node_storage, node_resource)

    entity_extractor = DateparserEntityExtractor(
        config={},
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
    )
    message.set(ENTITIES, [{"entity": "person", "value": "Max", "start": 0, "end": 3}])

    entity_extractor.process([message])
    entities = message.get(ENTITIES)
    assert entities == [
        {"entity": "person", "value": "Max", "start": 0, "end": 3},
    ]


def test_relative_base(tmpdir):
    message = Message(data={TEXT: "I want a pizza tomorrow"})
    node_storage = LocalModelStorage(pathlib.Path(tmpdir))
    node_resource = Resource("sparse_feat")
    context = ExecutionContext(node_storage, node_resource)

    entity_extractor = DateparserEntityExtractor(
        config={"relative_base": "2010-01-01"},
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
    )

    entity_extractor.process([message])
    entities = message.get(ENTITIES)
    assert entities[0]["parsed_date"][:10] == "2010-01-02"
