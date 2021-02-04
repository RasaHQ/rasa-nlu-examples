from typing import Text, List, Dict, Any

import pytest
from rasa.shared.nlu.constants import TEXT, INTENT, ENTITIES
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.training_data.training_data import TrainingData

from rasa_nlu_examples.extractors import FlashTextEntityExtractor


@pytest.mark.parametrize(
    "text, lookup, expected_entities",
    [
        (
            "Berlin and London are cities.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                }
            ],
            [
                {
                    "entity": "city",
                    "value": "Berlin",
                    "start": 0,
                    "end": 6,
                    "extractor": "FlashTextEntityExtractor",
                },
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "extractor": "FlashTextEntityExtractor",
                },
            ],
        ),
        (
            "Sophie is visiting Thomas in Berlin.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                },
                {"name": "person", "elements": ["Max", "John", "Sophie", "Lisa"]},
            ],
            [
                {
                    "entity": "person",
                    "value": "Sophie",
                    "start": 0,
                    "end": 6,
                    "extractor": "FlashTextEntityExtractor",
                },
                {
                    "entity": "city",
                    "value": "Berlin",
                    "start": 29,
                    "end": 35,
                    "extractor": "FlashTextEntityExtractor",
                },
            ],
        ),
        (
            "Rasa is great.",
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                },
                {"name": "person", "elements": ["Max", "John", "Sophie", "Lisa"]},
            ],
            [],
        ),
    ],
)
def test_process(
    text: Text,
    lookup: List[Dict[Text, List[Text]]],
    expected_entities: List[Dict[Text, Any]],
):
    message = Message(data={TEXT: text})

    training_data = TrainingData()
    training_data.lookup_tables = lookup
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

    entity_extractor = FlashTextEntityExtractor()
    entity_extractor.train(training_data)
    entity_extractor.process(message)

    entities = message.get(ENTITIES)
    assert entities == expected_entities


@pytest.mark.parametrize(
    "text, case_sensitive, lookup, expected_entities",
    [
        (
            "berlin and London are cities.",
            True,
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "London"],
                }
            ],
            [
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "extractor": "FlashTextEntityExtractor",
                }
            ],
        ),
        (
            "berlin and London are cities.",
            False,
            [
                {
                    "name": "city",
                    "elements": ["Berlin", "Amsterdam", "New York", "london"],
                }
            ],
            [
                {
                    "entity": "city",
                    "value": "berlin",
                    "start": 0,
                    "end": 6,
                    "extractor": "FlashTextEntityExtractor",
                },
                {
                    "entity": "city",
                    "value": "London",
                    "start": 11,
                    "end": 17,
                    "extractor": "FlashTextEntityExtractor",
                },
            ],
        ),
    ],
)
def test_lowercase(
    text: Text,
    case_sensitive: bool,
    lookup: List[Dict[Text, List[Text]]],
    expected_entities: List[Dict[Text, Any]],
):
    message = Message(data={TEXT: text})
    training_data = TrainingData()
    training_data.lookup_tables = lookup
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

    entity_extractor = FlashTextEntityExtractor({"case_sensitive": case_sensitive})
    entity_extractor.train(training_data)
    entity_extractor.process(message)

    entities = message.get(ENTITIES)
    assert entities == expected_entities


def test_do_not_overwrite_any_entities():
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

    entity_extractor = FlashTextEntityExtractor()
    entity_extractor.train(training_data)
    entity_extractor.process(message)
    entities = message.get(ENTITIES)
    assert entities == [
        {"entity": "person", "value": "Max", "start": 0, "end": 3},
        {
            "entity": "city",
            "value": "Berlin",
            "start": 13,
            "end": 19,
            "extractor": "FlashTextEntityExtractor",
        },
    ]
