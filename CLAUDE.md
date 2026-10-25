# pfngate: release-gate harness for early-warning models

Purpose: demo for an interview. A swappable PFN backend (TabPFN now, TipPFN later), an agent that picks context, a deterministic gate written in code, and a go/no-go report.

## Hard rules

- The gate is plain Python. An LLM agent may recommend, but it can never override or edit a gate decision.
- Split data by system family, never by window. Gate thresholds are tuned only on in-prior validation data and frozen in configs/gate.yaml BEFORE any out-of-prior or real-data evaluation.
- Every number in the README or report must be read from a file in results/ written by code. Never type a result by hand.
- Never claim to reproduce TipPFN. TipPFNBackend is a stub that raises NotImplementedError and documents the expected input and output.
- Never commit model weights, tokens, .env files or raw datasets. Newer TabPFN weights are gated and non-commercial. Do not redistribute them.
- Before calling any third-party API (tabpfn, tabpfn-client, pydantic-ai, anthropic), check the installed version with pip show and read its own docs or help(). Do not rely on memory for signatures or import paths.
- Set seeds everywhere. LLM calls use temperature 0.
- Keep modules small and typed. Run ruff and pytest before every commit.
- Write docs and comments in plain, direct English.
