"""The fundamentals assembler: the part most likely to be silently wrong.

These exercise the hard cases found against real filings -- year-to-date facts
that must be differenced into quarters, cover-page dates that must not hijack
the fiscal grid, restatements, and the earliest-versus-latest filing-date
distinction that point-in-time work depends on.
"""
from __future__ import annotations


import pandas as pd
import pytest

from stocklab.data.providers.sec import (
    _assemble, _cluster, _derive_quarters, _n_quarters, _record_first_filed,
    _snap, as_of_filter,
)


def fact(start, end, val, filed, form="10-Q"):
    return {"start": start, "end": end, "val": val, "filed": filed, "form": form}


class TestPeriodLength:
    @pytest.mark.parametrize("start,end,expected", [
        ("2025-01-01", "2025-03-31", 1),
        ("2025-01-01", "2025-06-30", 2),
        ("2025-01-01", "2025-09-30", 3),
        ("2025-01-01", "2025-12-31", 4),
        ("2024-09-29", "2025-09-27", 4),   # 52/53-week retail calendar
    ])
    def test_quarter_counting(self, start, end, expected):
        assert _n_quarters(pd.Timestamp(start), pd.Timestamp(end)) == expected

    def test_rejects_implausible_spans(self):
        assert _n_quarters(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-01-10")) is None
        assert _n_quarters(pd.Timestamp("2020-01-01"), pd.Timestamp("2025-01-01")) is None


class TestQuarterDerivation:
    def test_ytd_facts_are_differenced_into_quarters(self):
        # How US filers actually report: Q1 discrete, then H1, 9M and FY cumulative.
        durations = [
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31"), 100.0),
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-06-30"), 210.0),
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-09-30"), 330.0),
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), 460.0),
        ]
        quarters = _derive_quarters(durations)
        assert quarters[pd.Timestamp("2025-03-31")] == pytest.approx(100.0)
        assert quarters[pd.Timestamp("2025-06-30")] == pytest.approx(110.0)
        assert quarters[pd.Timestamp("2025-09-30")] == pytest.approx(120.0)
        # Q4 is almost never filed directly; it is the annual residual.
        assert quarters[pd.Timestamp("2025-12-31")] == pytest.approx(130.0)

    def test_quarters_sum_back_to_the_annual_total(self):
        durations = [
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31"), 100.0),
            (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30"), 110.0),
            (pd.Timestamp("2025-07-01"), pd.Timestamp("2025-09-30"), 120.0),
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), 460.0),
        ]
        quarters = _derive_quarters(durations)
        assert sum(quarters.values()) == pytest.approx(460.0)

    def test_discrete_quarters_pass_through(self):
        durations = [
            (pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31"), 100.0),
            (pd.Timestamp("2025-04-01"), pd.Timestamp("2025-06-30"), 110.0),
        ]
        assert len(_derive_quarters(durations)) == 2

    def test_insufficient_information_yields_nothing_rather_than_a_guess(self):
        durations = [(pd.Timestamp("2025-01-01"), pd.Timestamp("2025-12-31"), 460.0)]
        assert _derive_quarters(durations) == {}


class TestFiscalGrid:
    def test_nearby_dates_collapse_to_one_period(self):
        mapping = _cluster([pd.Timestamp("2025-09-27"), pd.Timestamp("2025-09-30"),
                            pd.Timestamp("2026-09-26")])
        assert mapping[pd.Timestamp("2025-09-27")] == mapping[pd.Timestamp("2025-09-30")]
        assert mapping[pd.Timestamp("2026-09-26")] != mapping[pd.Timestamp("2025-09-27")]

    def test_cover_page_date_snaps_back_to_the_period_it_describes(self):
        grid = [pd.Timestamp("2025-09-27"), pd.Timestamp("2026-09-26")]
        # A 10-K cover page is stamped a few weeks after year end.
        assert _snap(pd.Timestamp("2025-10-17"), grid) == pd.Timestamp("2025-09-27")

    def test_a_quarterly_instant_is_dropped_from_an_annual_grid(self):
        # Returning None is the point: a Q3 balance sheet must not be grafted
        # onto the annual frame just because it is the nearest row.
        grid = [pd.Timestamp("2025-09-27"), pd.Timestamp("2026-09-26")]
        assert _snap(pd.Timestamp("2026-06-27"), grid) is None

    def test_instant_slightly_before_period_end_snaps_forward(self):
        grid = [pd.Timestamp("2025-09-27"), pd.Timestamp("2026-09-26")]
        assert _snap(pd.Timestamp("2026-09-20"), grid) == pd.Timestamp("2026-09-26")


class TestFiledDates:
    def test_earliest_filing_wins(self):
        # Values use the latest filing (restatements supersede) but availability
        # must use the first: a 10-Q carries prior-year comparatives, so taking
        # the latest date back-dates a year of hindsight into the series.
        lookup: dict = {}
        period = pd.Timestamp("2025-03-31")
        _record_first_filed(lookup, period, "2026-04-30")
        _record_first_filed(lookup, period, "2025-04-30")
        _record_first_filed(lookup, period, "2027-04-30")
        assert lookup[period] == "2025-04-30"

    def test_missing_filing_date_is_ignored(self):
        lookup: dict = {}
        _record_first_filed(lookup, pd.Timestamp("2025-03-31"), None)
        assert lookup == {}


class TestAssembly:
    def _facts(self):
        return {
            "cik": "0000000001",
            "facts": {
                "us-gaap": {
                    "Revenues": {"units": {"USD": [
                        fact("2025-01-01", "2025-03-31", 100.0, "2025-04-30"),
                        fact("2025-01-01", "2025-06-30", 210.0, "2025-07-30"),
                        fact("2025-01-01", "2025-09-30", 330.0, "2025-10-30"),
                        fact("2025-01-01", "2025-12-31", 460.0, "2026-02-20", "10-K"),
                    ]}},
                    "Assets": {"units": {"USD": [
                        {"end": "2025-03-31", "val": 900.0, "filed": "2025-04-30"},
                        {"end": "2025-06-30", "val": 950.0, "filed": "2025-07-30"},
                        {"end": "2025-09-30", "val": 980.0, "filed": "2025-10-30"},
                        {"end": "2025-12-31", "val": 1000.0, "filed": "2026-02-20"},
                    ]}},
                    "StockholdersEquity": {"units": {"USD": [
                        {"end": "2025-12-31", "val": 600.0, "filed": "2026-02-20"},
                    ]}},
                },
                "dei": {
                    "EntityCommonStockSharesOutstanding": {"units": {"shares": [
                        # Cover-page date, three weeks after year end.
                        {"end": "2026-01-20", "val": 50.0, "filed": "2026-02-20"},
                    ]}},
                },
            },
        }

    def test_builds_annual_and_quarterly_frames(self):
        result = _assemble("TEST", "0000000001", self._facts(), {"sic": 3571})
        assert len(result.annual) == 1
        assert result.annual["revenue"].iloc[0] == pytest.approx(460.0)
        assert len(result.quarterly) == 4
        assert result.quarterly["revenue"].sum() == pytest.approx(460.0)

    def test_fiscal_year_label_is_the_period_end_not_the_cover_date(self):
        result = _assemble("TEST", "0000000001", self._facts(), {"sic": 3571})
        assert result.annual.index[0] == pd.Timestamp("2025-12-31")

    def test_cover_page_share_count_lands_on_the_fiscal_year(self):
        result = _assemble("TEST", "0000000001", self._facts(), {"sic": 3571})
        assert result.annual["shares_outstanding"].iloc[0] == pytest.approx(50.0)

    def test_total_liabilities_is_derived_from_the_identity(self):
        result = _assemble("TEST", "0000000001", self._facts(), {"sic": 3571})
        assert result.ttm.get("total_liabilities") == pytest.approx(400.0)
        assert result.annual["derived_total_liabilities"].iloc[0] == 1.0

    def test_ttm_prefers_four_quarters(self):
        result = _assemble("TEST", "0000000001", self._facts(), {"sic": 3571})
        assert result.ttm["ttm_basis"] == 2.0
        assert result.ttm["revenue"] == pytest.approx(460.0)

    def test_zero_share_counts_are_rejected(self):
        facts = self._facts()
        facts["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"].append(
            {"end": "2026-02-01", "val": 0.0, "filed": "2026-02-20"})
        result = _assemble("TEST", "0000000001", facts, {"sic": 3571})
        # A listed company cannot have zero shares; zero is finite, so without
        # the guard it would be accepted and collapse market cap to nothing.
        assert result.ttm.get("shares_outstanding") == pytest.approx(50.0)

    def test_restatement_uses_the_latest_filed_value(self):
        facts = self._facts()
        facts["facts"]["us-gaap"]["Revenues"]["units"]["USD"].append(
            fact("2025-01-01", "2025-12-31", 470.0, "2026-08-01", "10-K/A"))
        result = _assemble("TEST", "0000000001", facts, {"sic": 3571})
        assert result.annual["revenue"].iloc[0] == pytest.approx(470.0)

    def test_tags_are_merged_not_first_match(self):
        # Companies migrate tags when the FASB deprecates one; first-match
        # returns a series that stopped years ago.
        facts = self._facts()
        facts["facts"]["us-gaap"]["RevenueFromContractWithCustomerExcludingAssessedTax"] = {
            "units": {"USD": [fact("2023-01-01", "2023-12-31", 300.0, "2024-02-20", "10-K")]}
        }
        result = _assemble("TEST", "0000000001", facts, {"sic": 3571})
        assert len(result.annual) == 2
        assert set(result.sources["revenue"]) >= {"Revenues"}


class TestPointInTime:
    def test_as_of_filter_drops_later_filings(self):
        facts = {
            "cik": "1", "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
                fact("2025-01-01", "2025-03-31", 100.0, "2025-04-30"),
                fact("2025-01-01", "2025-06-30", 210.0, "2025-07-30"),
            ]}}}},
        }
        filtered = as_of_filter(facts, "2025-05-01")
        rows = filtered["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
        assert len(rows) == 1 and rows[0]["filed"] == "2025-04-30"

    def test_filtering_everything_leaves_no_tag(self):
        facts = {"cik": "1", "facts": {"us-gaap": {"Revenues": {"units": {"USD": [
            fact("2025-01-01", "2025-03-31", 100.0, "2025-04-30")]}}}}}
        assert as_of_filter(facts, "2020-01-01")["facts"] == {}


class TestFrameIndex:
    """The frames API is the cheap path for whole-market screens.

    Its index is a zero-padded CIK, which a CSV cache round-trip silently
    turns into an integer -- "0000320193" comes back as 320193, every lookup
    against the ticker map misses, and a screen over 5,000 filers matches a few
    hundred with no error anywhere.
    """

    def test_integer_index_is_repadded(self):
        from stocklab.data.providers.sec import _normalise_cik_index
        frame = pd.DataFrame({"val": [1.0, 2.0]}, index=[320193, 2098])
        out = _normalise_cik_index(frame)
        assert list(out.index) == ["0000320193", "0000002098"]
        assert out.index.name == "cik"

    def test_already_padded_index_is_unchanged(self):
        from stocklab.data.providers.sec import _normalise_cik_index
        frame = pd.DataFrame({"val": [1.0]}, index=["0000320193"])
        assert list(_normalise_cik_index(frame).index) == ["0000320193"]

    def test_empty_frame_survives(self):
        from stocklab.data.providers.sec import _normalise_cik_index
        assert _normalise_cik_index(pd.DataFrame()).empty
        assert _normalise_cik_index(None).empty

    def test_round_trip_through_the_cache_keeps_the_padding(self, tmp_path):
        from stocklab.data.cache import Cache
        from stocklab.data.providers.sec import _normalise_cik_index
        store = Cache(tmp_path / "c.sqlite")
        frame = pd.DataFrame({"val": [1.0]}, index=pd.Index(["0000320193"], name="cik"))
        store.set_df("frame", frame)
        assert list(_normalise_cik_index(store.get_df("frame")).index) == ["0000320193"]
