# OMEGA-WORLDCLASS-ENGINE — End-to-End Analysis and Reliability Revamp

This document is the deliverable of a full end-to-end audit of the repository.
It records what the system is for, what it already does well, where it can fail,
how the concept was flipped and pressure-tested from first principles, and the
engineering changes applied to make it a more capable, crash-resistant system.

Academic/educational research only. Not financial advice.

---

## 1. What the system is (goals)

The engine estimates probabilities of six 30-minute FX path phenomena —
trend ignition, liquidity sweep, volatility expansion, compression, failed
breakout, mean reversion — from information observable at prediction time:

```
P(phenomenon_{t:t+H} | information observable by t)
```

Its real product is **epistemic integrity**, not a trading signal. Every design
choice defends a small number of invariants:

1. **Point-in-time correctness** — no future leakage; features use only data
   available at each row's timestamp (`omega/features.py:9`).
2. **Immutable provenance** — raw data is content-addressed by SHA-256, stored
   under `raw/<source>/<symbol>/<digest>` and never silently overwritten
   (`omega/storage.py:40`).
3. **Fail-closed validation** — seven validation layers stop the pipeline before
   any training if data is unsorted, duplicated, invalid, or weekend-corrupted
   (`omega/validation.py:24`).
4. **Honest evaluation** — temporal walk-forward splits with embargo, calibration
   metrics (Brier, log-loss, ECE), prevalence baselines, and abstention
   (`omega/evaluation.py`, `omega/backtest.py`).
5. **Bounded, gated acquisition** — live vendor downloads are double-gated,
   capped to one partition by default, and never silently substituted
   (`omega/acquisition.py:76`).
6. **Reproducibility** — fixed seeds, deterministic fixtures, resumable
   acquisition via `StageLedger` (`omega/state.py`).

The V1 is deliberately narrow: 1-2 pairs, 25-50 features, logistic + histogram
gradient boosting, six config-driven labels. Optional modules (TFT, TabPFN, LLM,
quantum, RL, GNN) are explicitly OFF (`config.yaml`).

## 2. What the system already does well (strengths)

- **Scientific discipline is unusually strong.** The epistemic rule in
  `00_RESEARCH_AND_PLAN.md` is operationalized in code: quarantine rather than
  silently repair, version every transform, never convert SHAP into causal
  claims, compare every model against a training-prevalence baseline.
- **Immutable, content-addressed storage is genuinely correct.** `FileStore`
  blocks path traversal, `put_immutable` refuses same-key different-content
  writes, manifests record hashes, row counts, and timestamp ranges
  (`omega/storage.py`, `omega/data.py:13`).
- **Validation is layered and stops the run.** Seven checks: schema, timestamp
  integrity, calendar/gaps, OHLC/outliers, spread quality, point-in-time
  alignment, row uniqueness (`omega/validation.py:24`).
- **Acquisition is resumable and gated.** `StageLedger` persists per-partition
  state; the same run is idempotent (verified by tests); live execution requires
  both `--accept-provider-terms` and `data_source.explicit_terms_accepted`.
- **Tests encode the scientific contract.** 33 tests pass, including red-team
  checks for leakage (features must not change before a modification point),
  embargo gaps, content-addressed write conflicts, and causal-language guards in
  attribution.
- **Clean module separation** with single-responsibility files and a small,
  readable surface area (~3k lines).

## 3. Pressure test: where it can fail (weaknesses)

The audit red-teamed every module against "runs without failures or crashes."
Real gaps found, ordered by severity:

### A. Failures that would crash or corrupt a real run

| # | Weakness | Location | Consequence |
|---|---|---|---|
| 1 | **Network calls have no retry/backoff.** A transient 503 or timeout aborts the whole acquisition partition. | `omega/providers/oanda.py:49` | One flake kills the run |
| 2 | **Stale ledger locks never expire.** `os.open(..., O_EXCL)` leaves an orphaned `.lock` if the process dies mid-write; every later run times out forever. | `omega/state.py:71` | Permanent deadlock, no recovery |
| 3 | **No checkpoint/resume in the training pipeline.** `run_pipeline` recomputes every fold from scratch on crash; there is no per-fold persistence. | `omega/pipeline.py:31` | Long research runs lost |
| 4 | **`one_bar_latency` config is ignored.** `config.yaml:44` advertises one-bar latency but `hypothetical_state_response` fills at the decision bar. | `omega/backtest.py:13` | Cost-free, unrealistic fill timing |
| 5 | **Cross-partition duplicates are invisible.** Each monthly partition validates uniqueness *within itself only*; an overlapping/duplicated bar across a month boundary passes and later corrupts the merged training panel. | `omega/local_import.py:157`, `omega/partitions.py:49` | Silent duplicate rows in training |
| 6 | **Config is unvalidated dictionaries.** Missing/typo'd keys crash deep inside the pipeline with raw `KeyError`; the two expanders in `config.py` and `cloud_config.py` are duplicated code with no shared schema. | `omega/config.py:19`, `omega/cloud_config.py:23` | Opaque config failures, drift |
| 7 | **No structured logging.** `get_logger` exists but nothing logs progress or errors; a failed long run gives no trail. | `omega/utils.py:22` | Un-debuggable runs |
| 8 | **Exceptions are generic.** Everything raises `ValueError`/`RuntimeError`; callers cannot distinguish a bad-data failure from a network failure from a config error. | everywhere | No systematic recovery |

### B. Correctness and science hygiene issues

| # | Weakness | Location | Consequence |
|---|---|---|---|
| 9 | **Silent model skips.** A label/fold with <2 classes in train or calibration is silently skipped — metrics can vanish without any log. | `omega/pipeline.py:36` | Confusing, empty result sets |
| 10 | **Empty results are not guarded.** If no fold fits the data, `metrics.to_csv` writes an empty frame and the demo prints warnings instead of a clear message. | `omega/pipeline.py:46` | Opaque failure |
| 11 | **Spread requirement is inferred, not declared.** `require_spread` is derived from `"B" in price and "A" in price`; a price mode change silently changes validation strictness. | `omega/acquisition.py:100` | Silent behavior change |
| 12 | **Atomic writes lack `fsync`.** A power/VM loss can lose a manifest despite "atomic" rename. | `omega/utils.py:16` | Rare data loss |
| 13 | **`check_storage` / `bootstrap` free-space requirement is hard-coded at 1 GB defaults** with no relation to actual dataset size. | `omega/bootstrap.py:40` | Wrong failure thresholds |
| 14 | **Platform detection guesses by `/content` existence** — any Linux box with that dir misdetects as Colab. | `omega/runtime.py:26` | Wrong data roots |

### C. Minor / hygiene

- `_expand` env regex only supports `${VAR:-default}`, not plain `${VAR}`.
- `seeds` vary per fold but `seed_everything` mutates global state only; `PYTHONHASHSEED` set after interpreter start has no effect.
- `run_id` for acquisition is derived from provider/instrument/start/end; two different configs for the same window collide on the ledger path.
- `TelegramNotifier.send` propagates network errors into the caller (acceptable, but should be opt-in and non-fatal).
- Capacity report counts files under a path while another process may write; acceptable for reporting.

## 4. Flipping the concept

The conventional framing is: **predict the phenomenon, then trade it.** Pressure
testing that framing exposed the real risk concentration: the *data path* is
where false discoveries are born (leakage, revision, venue drift, duplicate
bars), and the *operational path* is where runs die (network flakes, stale
locks, crashes mid-fold).

**The flip:** make the system's primary deliverable **failure detection and
integrity preservation**, and treat the probability estimate as a well-guarded
side effect. Concretely:

1. **Reliability is the product.** The engine must degrade observably and
   recover deterministically: categorize errors, retry transient ones, persist
   progress, and refuse to continue when integrity is ambiguous.
2. **Abstention is the first decision.** The response policy already abstains
   below a threshold; the flipped design extends this to *data*: abstain from
   training a panel whose cross-partition integrity cannot be proven.
3. **Crashes are features to be tested.** Every persistence and lock path is
   tested by injecting failures (kill mid-write, stale lock, flaky network,
   duplicate months), so a future crash is a rehearsed scenario, not an event.

## 5. First-principles engineering rules

Derived from the invariants in section 1:

- **Transient != fatal.** Network and lock acquisition are transient by nature;
  they get bounded retry with backoff, then a categorized `ProviderError`.
- **Persistent != silent.** Immutable storage conflicts, stale locks that exceed
  a time budget, and cross-partition duplicates are permanent conditions; they
  raise `IntegrityError` and stop the run with an actionable message.
- **Progress is always persisted.** Every fold and every partition writes its
  result atomically before moving on, so any crash resumes rather than restarts.
- **Every configuration is validated at load time.** Wrong keys fail fast with
  a `ConfigError` naming the exact path and allowed values — never a deep
  `KeyError`.
- **Every assumption is logged.** Model skips, empty folds, inferred strictness,
  retries, and resumptions all produce structured log records.

## 6. Changes applied in this revamp

1. `omega/errors.py` — error taxonomy: `OmegaError` base with
   `ConfigError`, `DataError`, `IntegrityError`, `ProviderError`,
   `OperationalError`, `ResourceError`.
2. `omega/config.py` — unified env expansion, versioned schema validation with
   cross-field invariants, clear `ConfigError` messages; `cloud_config.py`
   now delegates to it (DRY).
3. `omega/state.py` — stale-lock recovery with ownership metadata and a time
   budget; race-safe reads; lock files record host/pid.
4. `omega/utils.py` — `retry` decorator (exponential backoff + jitter), fsync in
   atomic writes, structured logging helper.
5. `omega/providers/oanda.py` — bounded retry/backoff, structured `ProviderError`,
   response size guard, explicit timeouts.
6. `omega/backtest.py` — honors `one_bar_latency` (decision at bar `t`,
   fill at bar `t+1`), exposing latency in the response report.
7. `omega/local_import.py` and `omega/partitions.py` — cross-partition
   duplicate/overlap detection so a bar can never exist in two monthly
   partitions.
8. `omega/pipeline.py` — structured logging, per-fold checkpoint/resume via a
   run ledger, explicit guard for empty result sets, logged model skips.
9. `tests/test_reliability.py` — red-team tests for every fix above.
10. README updated with the reliability contract and new commands.

## 7. Adversarial pressure-test round (replayable)

A 68-scenario adversarial campaign (`scripts/pressure_test.py`) was run against
the engine and against the round-1 fixes. It probes the first-principles
invariants under hostile inputs: tiny/empty/constant/string datasets, invalid
config ranges, OANDA payload defects, path traversal, corrupt ledgers, stale and
contended locks, cross-source overlap, and crash/resume idempotence. Every
scenario now passes. Findings fixed in round 2:

| # | Defect found by the campaign | Fix |
|---|---|---|
| 1 | `synthetic_fx` crashed for small `n` (weekday filter could return fewer bars than requested → column length mismatch) | Fixture now generates enough candidates and derives `n` from what survives (`omega/data.py`) |
| 2 | `validate()` on an empty frame raised a raw `TypeError` from `np.isfinite` on object dtype instead of `DataValidationError` | Layer-1 schema check now also asserts non-empty; prices are coerced numerically before checks (`omega/validation.py`) |
| 3 | `build_features` accepted `w=0`, silently emitting garbage NaN columns | Non-positive windows rejected (`omega/features.py`) |
| 4 | Config accepted `alpha`, `abstain_below`, and `calibration_fraction` outside their valid ranges | Full schema now range-checks every key the pipeline reads, with documented defaults (`omega/config.py`) |
| 5 | `calibration_fraction=1.0` or `train_bars=1` emptied the train split and would crash `model.fit` deep in the run | `walk_forward_splits` rejects these at construction (`omega/evaluation.py`) |
| 6 | NaN `forward_return` was silently turned into zero P&L by `nan_to_num` | The response now reports `missing_returns` so gaps are surfaced, not hidden (`omega/backtest.py`) |
| 7 | Cross-source overlap was invisible | `overlap_advisory` reports sibling sources covering the same months on every import (`omega/partitions.py`) |
| 8 | No regression lock on the above | `tests/test_pressure.py` pins every fix (15 tests) |

Concurrency was also validated: four threads updating one `StageLedger` serialize
cleanly with no lost or clobbered stage records.

## 8. Remaining roadmap (out of scope for this pass)

- Combinatorial purged CV and PBO diagnostics (`CONVERSATION_HOOK` in evaluation).
- Point-in-time ALFRED macro adapter; quote-level spread adapter.
- Adaptive conformal calibration with drift monitoring.
- Vendor coverage discovery before scaling partitions beyond smoke-test scale.
- TFT/TabPFN behind the same temporal-split and calibration interface.
