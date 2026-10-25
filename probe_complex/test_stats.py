"""Tests for the probe fixture (deliberately thin)."""

from probe_complex.stats import rolling_average, summarize


def test_rolling_average_runs():
    rolling_average([1, 2, 3, 4, 5])


def test_summarize_runs():
    summarize([{"score": 10}, {"score": 20}])
