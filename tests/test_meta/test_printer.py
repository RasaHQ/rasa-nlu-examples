from rasa_nlu_examples.meta.printer import print_message
from rasa.shared.nlu.training_data.message import Message


def test_can_print_empty_message():
    print_message(Message())
