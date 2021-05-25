install:
	python -m pip install -e .
	pre-commit install
	python tests/scripts/prepare_fasttext.py
	python tests/scripts/prepare_stanza.py

test:
	pytest

style:
	black --check --diff --target-version py37 rasa_nlu_examples
	flake8 rasa_nlu_examples tests

check: style test

clean:
	rm models/*.tar.gz
