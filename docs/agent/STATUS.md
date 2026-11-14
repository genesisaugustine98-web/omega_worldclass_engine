# STATUS — current session handoff

Last updated: 2026-11-14
Commit: 23a300f
Branch: main
Working tree: clean at scaffold time (verify with `git status --short`)
Updated by: opencode — agent memory scaffold bootstrap
Review rule: read this first, then verify it against the tree. Overwrite it before you finish.

## One-line state

The agent memory scaffold was added on top of `23a300f`. No product code
changed in this bootstrap; any claim below is a pointer, not a fresh test result.

## What is true now

- Repository purpose: Colab-first, reproducible research software for estimating probabilities of six 30-minute FX market phenomena. Educational/academic use only, not financial advice.
- Entry points: `omega/pipeline.py` — pipeline entry; `omega/` — acquisition, features, labels, backtest, validation; `scripts/run_demo.py` — demo run; `tests/` — pytest suite.
- Current state authority: this file.

## In-flight work

None. The working tree was clean when the scaffold was committed.

## Verification recorded

- Scaffold is documentation only; no product tests were run during bootstrap.
- `pip install -e '.[dev]'` — not run (install).
- `pytest` — not run (unit tests).

## Open risks / blockers

- This is educational research software, not financial advice; do not present its outputs as trading recommendations.

## Next safe action

1. Run the verify command(s) in `START_HERE.md` and confirm they pass on this commit.
2. Then make the scoped change and record the real result here.
