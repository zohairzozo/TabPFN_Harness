.PHONY: test lint smoke full fast

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

smoke:
	uv run python -c "import pfngate; print('pfngate import ok')"

# The whole suite from one command: ruff, pytest, the frozen gate, every variant on every split,
# the summary and results/run_manifest.json. See scripts/run_all.py.
full:
	uv run python scripts/run_all.py

# A quick check: 5 scenarios of each kind, 1 seed each, written to results/fast/.
fast:
	uv run python scripts/run_all.py --fast
