# Claim Verifier

## Setup

```bash
UV_CACHE_DIR=.uv-cache uv sync --dev
```

Set `OPENAI_API_KEY` in your environment or in a local `.env` file. Optional runtime
defaults:

- `OPENAI_VISION_MODEL` defaults to `gpt-5.5`
- `OPENAI_AUDITOR_MODEL` defaults to `gpt-5.4-mini`
- `OPENAI_CONCURRENCY` defaults to `2`
- `OPENAI_TIMEOUT_SECONDS` defaults to `90`

## Final Predictions

```bash
UV_CACHE_DIR=.uv-cache uv run python code/main.py
```

This writes the authoritative submission artifact to root `output.csv`.

Useful smoke test:

```bash
UV_CACHE_DIR=.uv-cache uv run python code/main.py --limit 2 --no-audit
```

## Evaluation

Score an existing predictions file:

```bash
UV_CACHE_DIR=.uv-cache uv run python code/evaluation/main.py --predictions output.csv
```

Generate and score sample predictions:

```bash
UV_CACHE_DIR=.uv-cache uv run python code/evaluation/main.py --generate
```

Generated traces, reports, and mismatch CSVs are written under `code/runs/<timestamp>/`.
