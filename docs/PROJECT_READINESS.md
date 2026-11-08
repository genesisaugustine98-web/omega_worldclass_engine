# OMEGA project readiness

Review date: 8 November 2026.

## Scope

A Colab-first research engine for estimating probabilities of six 30-minute FX market phenomena. Its scientific contract includes temporal evaluation, calibration, abstention, observable-at-prediction-time inputs, content-hashed raw files and a training-prevalence baseline.

## Evidence present

- A packaged Python project with pinned core dependencies and a pytest development extra.
- A documented synthetic demo.
- A test suite and a pressure-test script.
- Checkpointed data/research pipeline and explicit acquisition term gates.
- Multiple free and paid data-provider paths are distinguished, with explicit terms acceptance and limits described.

The README reports 76 tests and 68 pressure scenarios. These remain repository-reported counts until the current branch CI run confirms them.

## Not established

- No live trading service or broker execution is claimed.
- No independently reviewed trading performance is established by the README.
- No independent research replication or third-party validation is claimed.
- Provider free tiers and data availability are time- and account-dependent.
- No root licence file was verified in this review; a project/package metadata decision alone should not be treated as a licence grant.

## Priority evidence gates

1. Keep CI green on the current commit and retain the result link.
2. Publish a compact reproducibility bundle: commit, environment, source hashes, config, data coverage, run command and test output.
3. Keep midpoint-only/resampled data visibly distinct from executable bid/ask data.
4. Keep acquisition double-gated by provider terms and explicit user action.
5. Record compute, storage, provider calls and cache-hit rate for resource/credit applications.

## Funding positioning

OMEGA is best framed as reproducible research infrastructure and an educational FX market-state research tool, not a proven trading strategy or financial product.