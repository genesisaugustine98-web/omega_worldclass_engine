# Setup instructions for Google Colab

1. Choose **Runtime → Change runtime type**. GPU is optional for V1.
2. Mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Set `OMEGA_REPO_URL` to your Git repository URL and run `colab/00_BOOTSTRAP_COLAB.ipynb`.
4. The bootstrap installs pinned requirements and sets Drive-backed data and run roots.
5. Add tokens under their exact `OMEGA_*` names in Colab Secrets. Bootstrap imports only missing values and reports booleans, never values. Keep secrets out of notebooks, YAML, output, and Git.
6. Run `!python scripts/run_demo.py` to verify the environment with synthetic data.
7. Place legally obtained CSV/Parquet data in Drive; use `omega.data.ingest_local`; never overwrite raw partitions.
8. Run validation before feature engineering. Any failed layer stops the pipeline.

Free Colab is ephemeral and quotas vary. A T4 and uninterrupted 4–6 hour runtime are not guaranteed. V1 uses CPU-friendly baselines and atomic stage artifacts.

The configured provider remains disabled while `explicit_terms_accepted: false`. Change it only after reviewing the provider's terms and confirming actual instrument/date coverage. The OANDA adapter is an integration path, not evidence that 20–30 years are available.

Bootstrap prints a capacity report. `raw/` and `manifests/` are never automatic retention candidates. Only disposable cache/temporary/old-derived paths may be proposed, and deletion remains a deliberate user operation.
