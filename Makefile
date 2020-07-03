install:
	python -m pip install -e .
	pre-commit install
	python tests/prepare_everything.py

test:
	python tests/prepare_everything.py
	pytest

style:
	black --check --diff --target-version py37 rasa_nlu_examples
	flake8 rasa_nlu_examples tests

types:
	pytype --keep-going rasa_nlu_examples --python-version 3.7 -j 16

check: style types test
