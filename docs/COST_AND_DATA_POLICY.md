# OMEGA cost and data policy

- Do not claim all operating costs are zero; compute, storage and vendor usage may cost money even when some APIs have free tiers.
- Keep provider terms acceptance as an explicit precondition.
- Preserve raw input files and SHA-256 provenance.
- Distinguish native 30-minute data from daily data resampled to 30-minute bars.
- Report acquisition requests, retries, data volume, cache hits, CPU/GPU time and storage usage.
- Do not automatically switch providers when a quota is exhausted if the replacement changes data semantics, time resolution or licensing.
- Stop on provider errors or incomplete data rather than silently manufacturing observations.
- Never embed provider credentials in notebooks or repository configuration.