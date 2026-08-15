# OMEGA-WORLDCLASS-ENGINE

Colab-first, reproducible research software for estimating probabilities of six 30-minute FX market phenomena. **Educational/academic use only; not financial advice.**

## Scientific contract

- Inputs must be observable at prediction time.
- Raw files are immutable and content-hashed.
- Failed validation stops the pipeline.
- Model explanation is associational, not proof of trader intent or causality.
- Evaluation is temporal and includes calibration, costs, and abstention.
- Feature generation rejects unsorted or duplicate timestamps and never backfills warm-up rows.
- Forward-horizon responses use non-overlapping entries and round-trip cost assumptions.
- Every learned model is compared with a training-prevalence probability baseline.

## Reliability contract

The engine treats failure detection as a first-class deliverable:

- **Categorized errors.** Every failure is a `ConfigError`, `DataError`,
  `IntegrityError`, `ProviderError`, `OperationalError`, or `ResourceError`
  (`omega/errors.py`) so a run can distinguish "fix your config" from
  "transient, retry" from "storage invariant broken, stop".
- **Transient is retried.** Provider network failures get bounded retry with
  exponential backoff and jitter before surfacing (`omega/providers/oanda.py`).
- **Stale locks are reclaimed.** A crashed run can never deadlock the ledger;
  abandoned locks older than the stale budget are recovered
  (`omega/state.py`).
- **Progress is checkpointed.** Every fold/label/model stage writes its metrics
  and predictions atomically; restarting resumes completed stages instead of
  recomputing them (`omega/pipeline.py`).
- **Config is validated at load time.** Wrong keys fail fast with a message
  naming the exact path and allowed values (`omega/config.py`).
- **Cross-partition integrity is audited.** A bar can never silently exist in
  two monthly partitions; imports and acquisition verify global timestamp
  uniqueness before a dataset is accepted (`omega/partitions.py`).
- **Backtest fills are honest.** The declared `one_bar_latency` is honored; a
  decision at bar `t` executes at bar `t+1` (`omega/backtest.py`).

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py
pytest -q
```

The demo creates synthetic data only, validates it, engineers features, labels
phenomena, trains temporal baselines, and writes artifacts under `artifacts/`.
Rerunning the demo resumes completed pipeline stages instead of refitting them.
The full test suite (research contracts + reliability red-team + pressure
regressions + free-provider adapters) is `pytest -q` (76 tests). The adversarial
campaign that found and pinned these regressions is replayable:

```bash
python scripts/pressure_test.py   # 68 scenarios, expects 0 WEAK/FAIL
```

## Colab

Open notebooks in `colab/` in numerical order. Mount Drive, clone/copy this repository, install requirements, and set `OMEGA_DATA_ROOT=/content/drive/MyDrive/OMEGA_DATA_v2`. Real vendor downloads are deliberately not automatic until source access, terms, coverage, and credentials are confirmed.

Run `colab/00_BOOTSTRAP_COLAB.ipynb` first. Kaggle fallback uses `colab/00_BOOTSTRAP_KAGGLE.ipynb`; both perform dependency and writable-storage checks and do not bypass provider terms.

## Bounded history acquisition

Dry-run one exact monthly partition first:

```bash
python scripts/acquire_history.py --start 2024-01-01T00:00:00Z --end 2024-02-01T00:00:00Z
```

Live execution is deliberately double-gated. Review the provider terms, set `data_source.explicit_terms_accepted: true`, provide `OMEGA_OANDA_TOKEN` through the platform secret store, then add `--execute --accept-provider-terms`. The default hard cap is one partition. This adapter does not establish long-history availability; inspect the resulting coverage and provenance manifest before scaling.

OANDA is not assumed to be available in every jurisdiction. Do not bypass regional restrictions. A provider-neutral local importer is available for legally obtained broker/vendor exports:

```bash
python scripts/import_history.py history.csv --source broker_export_v1 --instrument EUR_USD \
  --timestamp-column Date --open-column Open --high-column High --low-column Low \
  --close-column Close --spread-column Spread --timezone UTC --require-spread
```

The importer supports CSV/TXT and Parquet/PQ, requires an explicit timezone for naive timestamps, maps source columns explicitly, validates every calendar month before committing, stores the original file by SHA-256, and refuses to overwrite a conflicting monthly partition. Use a new `--source` identifier for a genuinely different dataset version.

## Free data providers and automatic self-refresh

Two no-cost historical adapters ship alongside OANDA. They are terms-gated like
every other source: review the provider terms, set
`data_source.explicit_terms_accepted: true`, and provide the free API key through
the platform secret store (never commit it).

- **Twelve Data** (`config/cloud_twelvedata.yaml`, `OMEGA_TWELVEDATA_API_KEY`):
  native 30-minute FX candles on its free tier for a limited major-pair set.
  This is the recommended free path for M30 research. The adapter paces
  requests (default 8s) to stay inside the free-tier credit budget.
- **Polygon.io** (`config/cloud_polygon.yaml`, `OMEGA_POLYGON_API_KEY`): free
  tier historically exposes daily FX aggregates only. The adapter resamples
  them to M30, so the underlying signal is daily, not intraday. It fails closed
  by default (`require_intraday: true`) so a researcher must consciously opt
  into the daily-resampled mode with `require_intraday: false`.

Both free providers return midpoint prices only (`price: M`), so a panel built
from them must be trained with `require_spread: false` (already the default in
`config.yaml`).

### Refresh missing data automatically

```bash
python scripts/acquire_history.py --config config/cloud_twelvedata.yaml \
  --start 2024-01-01T00:00:00Z --end 2026-09-01T00:00:00Z \
  --refresh --max-partitions 12 --accept-provider-terms
```

`--refresh` fetches only monthly partitions that are missing **or incomplete**. A
manifest existing is not proof of completeness: free tiers can truncate responses
(see Twelve Data's point cap below), and a stored month whose bars don't span the
full calendar month is automatically re-fetched. Months already stored complete
are never re-fetched, so repeated runs are cheap and idempotent. The current
calendar month is exempt from the completeness rule (it is still being produced).
The result reports `fetched_partitions`, `skipped_partitions`, and
`already_present`, plus a cross-partition integrity audit. Bounds the run with
`--max-partitions`.

**Twelve Data coverage note.** The free tier returns a rolling window capped at
~5000 points and fills the closed FX market (Saturday and Sunday before 20:00
UTC) with flat zero-volume bars. The adapter therefore requests each month with
explicit `start_date`/`end_date` (so history back to 2022 is reachable) and
drops the zero-volume closed-market bars so every stored bar is a real trade on
the FX calendar.

### Refresh-then-train in one command

```bash
python scripts/train.py --config config/cloud_twelvedata.yaml \
  --accept-provider-terms --max-partitions 12
```

`scripts/train.py` merges the cloud config's `data_source` with the full research
schema from `config.yaml`, calls `refresh_dataset()` to pull any missing months,
concatenates the stored partitions into a validated, deduplicated training panel
(`load_dataset()`), and runs the standard pipeline. Use `--dry-run` to preview
without executing, `--start`/`--end` to override the config window, and
`--research-config` to point at a different research schema. For scheduled
refresh-then-train (e.g. cron or a Colab cell), run `scripts/train.py` whenever
training is due; only new months are downloaded each time.

Secrets are loaded from the platform secret store by `load_platform_secrets`
(`omega/secrets.py`); new provider keys are `OMEGA_TWELVEDATA_API_KEY` and
`OMEGA_POLYGON_API_KEY`.

## Add a new SOTA paper in 10 minutes

1. Add an exact citation and supported proposition to `docs/references.md`.
2. Implement `BasePhenomenonModel` in `models/`.
3. Declare information availability and compute budget.
4. Add it to `config.yaml` disabled by default.
5. Compare against prevalence, logistic, and histogram-gradient-boosting baselines on identical temporal splits.
6. Add a red-team test and model card. Keep only if it improves untouched periods without breaking calibration or runtime limits.

## Data source policy

Adapters accept user-supplied CSV/Parquet. Dukascopy/HistData/news scraping is not silently enabled because historical coverage and licensing differ by pair and date. FRED latest-vintage data is not point-in-time safe; use ALFRED vintages for causal/event research.

## Project status

V1 prioritizes predictive benchmarking with causal hygiene. TFT, TabPFN, LLM news, causal graphs, graph neural networks, RL, and quantum-walk modules are opt-in research extensions, not default claims.

**CONVERSATION_HOOK:** Next upgrade candidates: point-in-time ALFRED adapter, quote-level spread adapter, TFT benchmark, and adaptive conformal calibration. The reliability pass (error taxonomy, retry/backoff, stale-lock recovery, resumable pipeline, cross-partition integrity audit, latency-honest backtest) is complete; see `docs/ANALYSIS.md` for the full audit.
