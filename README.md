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

## Quick start

```bash
python -m pip install -r requirements.txt
python scripts/run_demo.py
pytest -q
```

The demo creates synthetic data only, validates it, engineers features, labels phenomena, trains temporal baselines, and writes artifacts under `artifacts/`.

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

**CONVERSATION_HOOK:** Next upgrade candidates: point-in-time ALFRED adapter, quote-level spread adapter, TFT benchmark, and adaptive conformal calibration.
