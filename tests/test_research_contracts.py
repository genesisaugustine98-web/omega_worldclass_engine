import numpy as np

from omega.backtest import attribution, hypothetical_state_response
from omega.evaluation import probability_metrics


def test_response_uses_non_overlapping_entries_and_round_trip_costs():
    frame, report = hypothetical_state_response(
        np.full(10, 0.9), np.full(10, 0.01), holding_bars=4, spread_bps=1, slippage_bps=0.5, one_bar_latency=False
    )
    assert frame.index[frame.position.eq(1)].tolist() == [0, 4, 8]
    assert report["entries"] == 3
    assert np.allclose(frame.loc[[0, 4, 8], "cost"], 0.0003)
    assert frame.loc[[1, 2, 3, 5, 6, 7, 9], "pnl"].eq(0).all()


def test_one_bar_latency_defers_fill_to_next_bar():
    frame, report = hypothetical_state_response(
        np.full(10, 0.9), np.full(10, 0.01), holding_bars=4, spread_bps=1, slippage_bps=0.5, one_bar_latency=True
    )
    assert frame.index[frame.position.eq(1)].tolist() == [1, 6]
    assert report["one_bar_latency"] is True
    assert report["entries"] == 2
    assert frame.loc[0, "position"] == 0
    assert np.allclose(frame.loc[[1, 6], "cost"], 0.0003)


def test_response_validates_shapes_and_holding_period():
    try:
        hypothetical_state_response([0.8], [0.1, 0.2])
        assert False
    except ValueError as error:
        assert "equal shape" in str(error)
    try:
        hypothetical_state_response([0.8], [0.1], holding_bars=0)
        assert False
    except ValueError as error:
        assert "at least 1" in str(error)


def test_prevalence_is_a_defined_probability_baseline():
    y = np.array([0, 0, 1, 1])
    metrics = probability_metrics(y, np.full(4, y.mean()))
    assert metrics["brier"] == 0.25
    assert metrics["log_loss"] > 0


def test_attribution_language_rejects_causal_claims():
    report = attribution("trend_ignition", np.array([1, 0]), np.array([0.9, 0.8]))
    assert report["selected_windows"] == 2
    assert "not proof" in report["language"]