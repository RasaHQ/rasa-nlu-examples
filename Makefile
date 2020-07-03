install:
	python -m pip install -e .
	pre-commit install
	python tests/prepare_everything.py

test:
	python tests/prepare_everything.py
	pytest -n 2

style:
	black --check rasa_nlu_examples tests
	flake8 rasa_nlu_examples tests

types:
	pytype --keep-going rasa_nlu_examples -j 16

check: style types test
