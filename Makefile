.PHONY: test lint smoke full fast report demo

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

# The go/no-go report: results/report/index.html, one self-contained page. It reads files in
# results/ only (run `make full` first) and fails if a number on the page is in no result file.
report:
	uv run python -m pfngate.report

# A short walkthrough of two held-out runs, offline (--llm-mode replay), saved to
# results/demo_transcript.txt. Needs the caches that `make full` leaves in results/cache.
demo:
	uv run python scripts/demo.py
