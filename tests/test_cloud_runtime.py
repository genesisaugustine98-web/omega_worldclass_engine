import json

import pytest

from omega.bootstrap import bootstrap, check_dependencies, check_storage
from omega.capacity import inspect_capacity
from omega.secrets import load_platform_secrets
from omega.state import StageLedger


def test_bootstrap_uses_explicit_local_roots(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    run_root = tmp_path / "runs"
    monkeypatch.setenv("OMEGA_DATA_ROOT", str(data_root))
    monkeypatch.setenv("OMEGA_RUN_ROOT", str(run_root))

    report = bootstrap(tmp_path, required_free_bytes=1)

    assert report.platform == "local"
    assert report.data_root == str(data_root)
    assert report.run_root == str(run_root)
    assert report.dependencies_ok
    assert data_root.is_dir()
    assert run_root.is_dir()


def test_dependency_check_reports_missing_package():
    ok, missing = check_dependencies(("module_that_does_not_exist_omega",))
    assert not ok
    assert missing == ["module_that_does_not_exist_omega"]


def test_storage_check_fails_closed_when_requirement_is_impossible(tmp_path):
    with pytest.raises(RuntimeError, match="Insufficient storage"):
        check_storage(tmp_path, required_free_bytes=10**30)


def test_stage_ledger_recovers_after_failed_stage(tmp_path):
    ledger_path = tmp_path / "run" / "ledger.json"
    ledger = StageLedger(ledger_path, "research-run-1")

    with pytest.raises(ValueError):
        with ledger.stage("download:EUR_USD:2024-01") as should_run:
            assert should_run
            raise ValueError("simulated disconnect")

    assert ledger.status("download:EUR_USD:2024-01") == "failed"

    with ledger.stage("download:EUR_USD:2024-01") as should_run:
        assert should_run

    assert ledger.is_complete("download:EUR_USD:2024-01")
    with ledger.stage("download:EUR_USD:2024-01") as should_run:
        assert not should_run

    payload = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert payload["run_id"] == "research-run-1"


def test_secret_loader_reports_status_without_returning_values(monkeypatch):
    monkeypatch.delenv("OMEGA_OANDA_TOKEN", raising=False)
    status = load_platform_secrets(("OMEGA_OANDA_TOKEN",), getter=lambda name: "private-token")
    assert status == {"OMEGA_OANDA_TOKEN": True}
    assert "private-token" not in json.dumps(status)


def test_bootstrap_reports_provider_key_presence_only(monkeypatch):
    monkeypatch.setenv("OMEGA_DATA_ROOT", "/tmp/omega-boot-provider-data")
    monkeypatch.setenv("OMEGA_RUN_ROOT", "/tmp/omega-boot-provider-runs")
    monkeypatch.delenv("OMEGA_TWELVEDATA_API_KEY", raising=False)
    monkeypatch.delenv("OMEGA_POLYGON_API_KEY", raising=False)
    report = bootstrap("/tmp", required_free_bytes=1)
    assert report.twelvedata_key_present is False
    assert report.polygon_key_present is False
    monkeypatch.setenv("OMEGA_TWELVEDATA_API_KEY", "not-a-real-key")
    report2 = bootstrap("/tmp", required_free_bytes=1)
    assert report2.twelvedata_key_present is True
    assert report2.polygon_key_present is False
    assert "not-a-real-key" not in json.dumps(report2.as_dict())


def test_capacity_report_never_selects_raw_data(tmp_path):
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "immutable.bin").write_bytes(b"raw")
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "discardable.bin").write_bytes(b"cache")

    report = inspect_capacity(tmp_path, minimum_free_bytes=10**30)

    assert report.warning
    assert report.managed_bytes == 8
    assert str(tmp_path / "cache") in report.retention_candidates
    assert all("raw" not in candidate for candidate in report.retention_candidates)