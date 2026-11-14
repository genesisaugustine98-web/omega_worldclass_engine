# Start here — omega_worldclass_engine

1. Run `agent-pickup --brief`.
2. Read `STATUS.md` for current state and the next safe action.
3. Read the entry points below before editing.
4. If memory and code disagree, verify in the tree and fix `STATUS.md`.

## What this is

Colab-first, reproducible research software for estimating probabilities of six 30-minute FX market phenomena. Educational/academic use only, not financial advice.

## Entry points

- `omega/pipeline.py` — pipeline entry
- `omega/` — acquisition, features, labels, backtest, validation
- `scripts/run_demo.py` — demo run
- `tests/` — pytest suite

## Verify / run

```bash
pip install -e '.[dev]'
pytest
```

## Source of truth

`docs/agent/STATUS.md` is the only current-state document. Verify it against the tree.

Keep this file short. It is a router, not a transcript.
