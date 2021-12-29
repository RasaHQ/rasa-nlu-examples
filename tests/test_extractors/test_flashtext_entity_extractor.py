import pathlib
from typing import Text, List, Dict, Any

import pytest
from rasa.shared.nlu.constants import TEXT, INTENT, ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.engine.graph import ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.local_model_storage import LocalModelStorage

from rasa_nlu_examples.extractors import FlashTextEntityExtractor


@pytest.fixture()
def flashtext_entity_extractor(tmp_path) -> FlashTextEntityExtractor:
    node_storage = LocalModelStorage(pathlib.Path(tmp_path))
    node_resource = Resource("flashtext")
    context = ExecutionContext(node_storage, node_resource)
    return FlashTextEntityExtractor(
        config={"entity_name": "city", "path": "tests/data/flashtext/cities.txt"},
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
    )


@pytest.mark.parametrize(
    "text, expected_entities",
    [
        (
            "Berlin and London are cities.",
            [
                {
                    "entity": "city",
                    "value": "Berlin",
                    "start": 0,
                    "end": 6,
                    "confidence": 1.0,
                    "extractor": "FlashTextEntityExtractor",
                },
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "confidence": 1.0,
                    "extractor": "FlashTextEntityExtractor",
                },
            ],
        ),
        (
            "Sophie is visiting Thomas in Berlin.",
            [
                {
                    "entity": "city",
                    "value": "Berlin",
                    "start": 29,
                    "end": 35,
                    "confidence": 1.0,
                    "extractor": "FlashTextEntityExtractor",
                },
            ],
        ),
        (
            "Rasa is great.",
            [],
        ),
    ],
)
def test_process(
    text: Text, expected_entities: List[Dict[Text, Any]], flashtext_entity_extractor
):
    message = Message(data={TEXT: text})

    training_data = TrainingData()
    training_data.training_examples = [
        Message(
            data={
                TEXT: "I live in Berlin",
                INTENT: "inform",
                ENTITIES: [{"entity": "city", "value": "Berlin"}],
            }
        ),
    ]

    flashtext_entity_extractor.train(training_data)
    flashtext_entity_extractor.process([message])

    entities = message.get(ENTITIES)
    assert entities == expected_entities


@pytest.mark.parametrize(
    "text, case_sensitive, expected_entities",
    [
        (
            "berlin and London are cities.",
            True,
            [
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "confidence": 1.0,
                    "extractor": "FlashTextEntityExtractor",
                }
            ],
        ),
        (
            "berlin and London are cities.",
            False,
            [
                {
                    "entity": "city",
                    "value": "Berlin",
                    "start": 0,
                    "end": 6,
                    "confidence": 1.0,
                    "extractor": "FlashTextEntityExtractor",
                },
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "confidence": 1.0,
                    "extractor": "FlashTextEntityExtractor",
                },
            ],
        ),
    ],
)
def test_lowercase(
    text: Text, case_sensitive: bool, expected_entities: List[Dict[Text, Any]], tmp_path
):
    message = Message(data={TEXT: text})
    training_data = TrainingData()
    training_data.training_examples = [
        Message(
            data={
                TEXT: "I live in Berlin",
                INTENT: "inform",
                ENTITIES: [{"entity": "city", "value": "Berlin"}],
            }
        ),
    ]

    node_storage = LocalModelStorage(pathlib.Path(tmp_path))
    node_resource = Resource("flashtext")
    context = ExecutionContext(node_storage, node_resource)
    entity_extractor = FlashTextEntityExtractor(
        {
            "case_sensitive": case_sensitive,
            "entity_name": "city",
            "path": "tests/data/flashtext/cities.txt",
        },
        name=context.node_name,
        resource=node_resource,
        model_storage=node_storage,
    )
    entity_extractor.train(training_data)
    entity_extractor.process([message])

    entities = message.get(ENTITIES)
    assert entities == expected_entities


def test_do_not_overwrite_any_entities(flashtext_entity_extractor):
    message = Message(data={TEXT: "Max lives in Berlin.", INTENT: "infrom"})
    message.set(ENTITIES, [{"entity": "person", "value": "Max", "start": 0, "end": 3}])

    training_data = TrainingData()
    training_data.training_examples = [
        Message(
            data={
                TEXT: "Hi Max!",
                INTENT: "greet",
                ENTITIES: [{"entity": "person", "value": "Max"}],
            }
        ),
        Message(
            data={
                TEXT: "I live in Berlin",
                INTENT: "inform",
                ENTITIES: [{"entity": "city", "value": "Berlin"}],
            }
        ),
    ]
    training_data.lookup_tables = [
        {"name": "city", "elements": ["London", "Berlin", "Amsterdam"]}
    ]

    flashtext_entity_extractor.train(training_data)
    flashtext_entity_extractor.process([message])
    entities = message.get(ENTITIES)
    assert entities == [
        {"entity": "person", "value": "Max", "start": 0, "end": 3},
        {
            "entity": "city",
            "value": "Berlin",
            "start": 13,
            "end": 19,
            "confidence": 1.0,
            "extractor": "FlashTextEntityExtractor",
        },
    ]
