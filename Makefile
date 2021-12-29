install:
	python -m pip install -e .
	pre-commit install
	python tests/scripts/prepare_fasttext.py
	python -m pip install deadlink

test:
	pytest

black:
	black --check --diff --target-version py37 rasa_nlu_examples tests

flake:
	flake8 rasa_nlu_examples tests

style: black flake

links:
	deadlink check readme.md docs

smoketest:
	python scripts/smoketests.py

check: style test links smoketest clean

clean:
	rm -rf */**/__pycache__ models/*.tar.gz __pycache__ .ipynb_checkpoints .pytest_cache
