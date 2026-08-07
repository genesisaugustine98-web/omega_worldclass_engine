from pathlib import Path

import pandas as pd
import pytest

from omega.local_import import ImportSchema, import_history_file, read_local_history
from omega.validation import DataValidationError


def write_fixture(path: Path, rows: list[dict]) -> Path:
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def valid_rows():
    return [
        {"Date": "2024-01-02 00:00", "O": 1.10, "H": 1.12, "L": 1.09, "C": 1.11, "S": 0.0001},
        {"Date": "2024-01-02 00:30", "O": 1.11, "H": 1.13, "L": 1.10, "C": 1.12, "S": 0.0001},
        {"Date": "2024-02-01 00:00", "O": 1.12, "H": 1.14, "L": 1.11, "C": 1.13, "S": 0.0001},
    ]


def schema():
    return ImportSchema(timestamp="Date", open="O", high="H", low="L", close="C", spread="S", timezone="UTC")


def test_import_maps_validates_partitions_hashes_and_resumes(tmp_path):
    source_file = write_fixture(tmp_path / "fixture.csv", valid_rows())
    data_root = tmp_path / "data"
    first = import_history_file(source_file, data_root, "fixture_v1", "EUR_USD", schema(), require_spread=True)
    second = import_history_file(source_file, data_root, "fixture_v1", "EUR_USD", schema(), require_spread=True)
    assert first["partition_count"] == 2
    assert first["dataset"]["dataset_root_sha256"] == second["dataset"]["dataset_root_sha256"]
    assert first["manifests"] == second["manifests"]
    assert len(list((data_root / "raw" / "local" / "fixture_v1" / "EUR_USD").glob("*.csv"))) == 1
    assert (data_root / "normalized" / "local-fixture_v1" / "EUR_USD" / "2024" / "01" / "bars.parquet").is_file()


def test_naive_timestamp_requires_timezone(tmp_path):
    source_file = write_fixture(tmp_path / "fixture.csv", valid_rows()[:1])
    with pytest.raises(ValueError, match="explicit --timezone"):
        read_local_history(source_file, ImportSchema(timestamp="Date", open="O", high="H", low="L", close="C"))


def test_duplicate_source_column_mapping_is_rejected():
    duplicate_mapping = ImportSchema(timestamp="Date", open="O", high="H", low="L", close="O", timezone="UTC")
    with pytest.raises(ValueError, match="only one canonical column"):
        duplicate_mapping.column_map()


def test_offset_timestamp_rejects_redundant_timezone(tmp_path):
    rows = valid_rows()[:1]
    rows[0]["Date"] = "2024-01-02T00:00:00+01:00"
    source_file = write_fixture(tmp_path / "fixture.csv", rows)
    with pytest.raises(ValueError, match="already include offsets"):
        read_local_history(source_file, schema())


def test_bad_ohlc_and_duplicate_timestamp_stop_before_commit(tmp_path):
    bad_rows = valid_rows()
    bad_rows[1]["H"] = 0.5
    source_file = write_fixture(tmp_path / "bad.csv", bad_rows)
    data_root = tmp_path / "data"
    with pytest.raises(DataValidationError):
        import_history_file(source_file, data_root, "fixture_v1", "EUR_USD", schema(), require_spread=True)
    assert not (data_root / "raw").exists()

    duplicate_rows = valid_rows()[:2]
    duplicate_rows[1]["Date"] = duplicate_rows[0]["Date"]
    source_file = write_fixture(tmp_path / "duplicate.csv", duplicate_rows)
    with pytest.raises(DataValidationError, match="duplicates=1"):
        import_history_file(source_file, data_root, "fixture_v1", "EUR_USD", schema(), require_spread=True)


def test_conflicting_partition_requires_new_source_identifier(tmp_path):
    source_file = write_fixture(tmp_path / "first.csv", valid_rows()[:2])
    data_root = tmp_path / "data"
    import_history_file(source_file, data_root, "fixture_v1", "EUR_USD", schema(), require_spread=True)
    changed = valid_rows()[:2]
    changed[1]["C"] = 1.125
    source_file = write_fixture(tmp_path / "changed.csv", changed)
    with pytest.raises(FileExistsError, match="different content"):
        import_history_file(source_file, data_root, "fixture_v1", "EUR_USD", schema(), require_spread=True)


def test_identifiers_cannot_escape_storage_root(tmp_path):
    source_file = write_fixture(tmp_path / "fixture.csv", valid_rows()[:1])
    with pytest.raises(ValueError, match="source must match"):
        import_history_file(source_file, tmp_path / "data", "../escape", "EUR_USD", schema())