# Publishing Checklist

Run this before pushing to a public GitHub repository.

## Repository Hygiene

```bash
git status --short
git ls-files | sort
USER_PREFIX='/Users'
PATTERN="(sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{20,}|${USER_PREFIX}/[^[:space:]]+)"
git grep -nE "$PATTERN" -- ':!docs/PUBLISHING_CHECKLIST.md'
```

Expected:

- no API keys
- no `config.json`
- no `.env`
- no model weights
- no SQLite databases
- no raw log directory
- no prompt-cache binaries

## Validation

```bash
.venv/bin/python scripts/validate_environment.py
.venv/bin/python -m qpu_mcp_lab.cli init-db
.venv/bin/python -m qpu_mcp_lab.cli best --limit 5
```

## Docs

- `README.md` describes the project and quick start.
- `docs/REPRODUCIBILITY.md` contains exact validation protocol.
- `docs/RESULTS.md` separates strict-quality and speed-only records.
- `SECURITY.md` explains secret handling and QPU guardrails.
- `CITATION.cff` is present for paper/repo citation.
- `LICENSE` is present.

## Release Notes

For a first public tag, include:

- hardware and OS
- model quant and source
- llama.cpp / ik_llama.cpp commit
- strict-quality record config
- quality gate outputs
- known failure modes
- statement that model weights are not redistributed
