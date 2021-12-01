install:
	python -m pip install -e .
	pre-commit install
	python tests/scripts/prepare_fasttext.py
	python tests/scripts/prepare_stanza.py

test:
	pytest

black:
	black --check --diff --target-version py37 rasa_nlu_examples tests

flake:
	flake8 rasa_nlu_examples tests

style: black flake

check: style test

clean:
	rm models/*.tar.gz
