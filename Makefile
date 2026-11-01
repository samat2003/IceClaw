# IceClaw routine commands. Run inside the project virtual environment (.venv).
PYTHON ?= python
DEMO_BUNDLE ?= inv-20261031T201513Z-p6-public

.PHONY: help install test lint format-check check ui verify-demo replay-demo

help:  ## list targets
	@grep -E '^[a-z-]+:.*##' $(MAKEFILE_LIST) | sed 's/:.*##/ -/'

install:  ## editable install with test/lint tools
	$(PYTHON) -m pip install -e ".[dev]"

test:  ## offline test suite (no API key, no paid calls)
	$(PYTHON) -m pytest -q

lint:  ## ruff lint
	ruff check .

format-check:  ## ruff formatting check
	ruff format --check .

check: test lint format-check  ## all offline quality gates

ui:  ## start the Streamlit workbench on http://localhost:8501
	$(PYTHON) -m streamlit run app.py

verify-demo:  ## verify the recorded public bundle offline (copy it into runs/ first)
	$(PYTHON) -m iceclaw.cli verify --run $(DEMO_BUNDLE)

replay-demo:  ## replay the recorded investigation from saved events (0 model calls)
	$(PYTHON) -m iceclaw.cli replay --run $(DEMO_BUNDLE)
