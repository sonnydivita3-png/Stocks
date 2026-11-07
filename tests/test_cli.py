"""CLI wiring. Network-dependent commands are only checked for help text."""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from click.testing import CliRunner

from stocklab.cli import main


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def invoke(runner, args, data_dir):
    return runner.invoke(main, ["--data-dir", str(data_dir)] + args)


class TestHelp:
    def test_top_level_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        for command in ("analyse", "value", "reverse", "screen", "regime",
                        "portfolio", "rebalance", "tax", "harvest", "risk"):
            assert command in result.output

    def test_not_advice_disclaimer_is_present(self, runner):
        result = runner.invoke(main, ["--help"])
        assert "Not investment advice" in result.output

    @pytest.mark.parametrize("command", [
        "quote", "analyse", "value", "reverse", "entry", "regime", "screen",
        "portfolio", "performance", "rebalance", "tax", "harvest", "risk",
        "cache",
    ])
    def test_each_command_has_help(self, runner, command):
        result = runner.invoke(main, [command, "--help"])
        assert result.exit_code == 0

    @pytest.mark.parametrize("group,sub", [
        ("account", "add"), ("account", "list"), ("txn", "add"),
        ("txn", "import"), ("txn", "list"), ("watch", "add"), ("watch", "show"),
    ])
    def test_each_subcommand_has_help(self, runner, group, sub):
        result = runner.invoke(main, [group, sub, "--help"])
        assert result.exit_code == 0


class TestScreenListing:
    def test_list_presets_needs_no_network(self, runner, tmp_path):
        result = invoke(runner, ["screen", "--list-presets"], tmp_path)
        assert result.exit_code == 0
        assert "magic-formula" in result.output
        assert "quality" in result.output

    def test_unknown_preset_exits_nonzero(self, runner, tmp_path):
        result = invoke(runner, ["screen", "nonsense", "--symbols", "AAA"], tmp_path)
        assert result.exit_code != 0


class TestPortfolioWorkflow:
    def test_end_to_end_account_and_transaction_flow(self, runner, tmp_path):
        created = invoke(runner, ["account", "add", "Brokerage",
                                  "--type", "taxable", "--basis-method", "hifo"],
                         tmp_path)
        assert created.exit_code == 0, created.output
        assert "Brokerage" in created.output

        roth = invoke(runner, ["account", "add", "Roth", "--type", "roth_ira"],
                      tmp_path)
        assert roth.exit_code == 0
        assert "Sheltered account" in roth.output   # the harvesting warning

        duplicate = invoke(runner, ["account", "add", "Brokerage",
                                    "--type", "taxable"], tmp_path)
        assert duplicate.exit_code != 0

        added = invoke(runner, ["txn", "add", "--account", "Brokerage",
                                "--date", "2024-03-01", "--type", "buy",
                                "--symbol", "AAA", "--quantity", "100",
                                "--price", "50"], tmp_path)
        assert added.exit_code == 0, added.output

        listed = invoke(runner, ["txn", "list"], tmp_path)
        assert listed.exit_code == 0 and "AAA" in listed.output

        accounts = invoke(runner, ["account", "list"], tmp_path)
        assert accounts.exit_code == 0
        assert "Roth IRA" in accounts.output and "HIFO" in accounts.output

    def test_transaction_against_a_missing_account_fails(self, runner, tmp_path):
        result = invoke(runner, ["txn", "add", "--account", "Nope",
                                 "--date", "2024-01-01", "--type", "buy",
                                 "--symbol", "AAA", "--quantity", "1",
                                 "--price", "1"], tmp_path)
        assert result.exit_code != 0

    def test_csv_import_dry_run(self, runner, tmp_path):
        invoke(runner, ["account", "add", "Main", "--type", "taxable"], tmp_path)
        csv = tmp_path / "trades.csv"
        # Deliberately uses broker-style column names and one bad row.
        csv.write_text(
            "Trade Date,Action,Ticker,Shares,Share Price,Commission\n"
            "2024-01-02,buy,AAA,10,100.0,1.0\n"
            "2024-02-02,sell,AAA,5,120.0,1.0\n"
            "2024-03-02,reorganisation,AAA,1,1.0,0\n"
        )
        result = invoke(runner, ["txn", "import", str(csv), "--account", "Main",
                                 "--dry-run"], tmp_path)
        assert result.exit_code == 0, result.output
        assert "Parsed 2 transaction(s); 1 skipped" in result.output
        assert "Dry run" in result.output
        # Nothing should have been saved.
        assert "No transactions" in invoke(runner, ["txn", "list"], tmp_path).output

    def test_csv_import_saves(self, runner, tmp_path):
        invoke(runner, ["account", "add", "Main", "--type", "taxable"], tmp_path)
        csv = tmp_path / "trades.csv"
        csv.write_text("date,type,symbol,quantity,price\n"
                       "2024-01-02,buy,AAA,10,100.0\n")
        result = invoke(runner, ["txn", "import", str(csv), "--account", "Main"],
                        tmp_path)
        assert result.exit_code == 0
        assert "Imported 1" in result.output

    def test_csv_missing_required_columns_fails(self, runner, tmp_path):
        invoke(runner, ["account", "add", "Main", "--type", "taxable"], tmp_path)
        csv = tmp_path / "bad.csv"
        csv.write_text("ticker,shares\nAAA,10\n")
        result = invoke(runner, ["txn", "import", str(csv), "--account", "Main"],
                        tmp_path)
        assert result.exit_code != 0
        assert "missing required column" in result.output

    def test_empty_portfolio_is_a_message_not_a_crash(self, runner, tmp_path):
        result = invoke(runner, ["portfolio"], tmp_path)
        assert result.exit_code == 0
        assert "No open positions" in result.output

    def test_tax_report_with_no_disposals(self, runner, tmp_path):
        invoke(runner, ["account", "add", "Main", "--type", "taxable"], tmp_path)
        result = invoke(runner, ["tax", "--year", "2026"], tmp_path)
        assert result.exit_code == 0
        assert "not tax advice" in result.output.lower()

    def test_harvest_with_nothing_to_harvest(self, runner, tmp_path):
        invoke(runner, ["account", "add", "Main", "--type", "taxable"], tmp_path)
        result = invoke(runner, ["harvest"], tmp_path)
        assert result.exit_code == 0
        assert "not deductible" in result.output

    def test_rebalance_without_targets_fails_clearly(self, runner, tmp_path):
        invoke(runner, ["account", "add", "Main", "--type", "taxable"], tmp_path)
        result = invoke(runner, ["rebalance"], tmp_path)
        assert result.exit_code != 0
        assert "No targets" in result.output

    def test_watchlist_round_trip(self, runner, tmp_path):
        added = invoke(runner, ["watch", "add", "AAA", "--thesis", "cheap"], tmp_path)
        assert added.exit_code == 0
        removed = invoke(runner, ["watch", "remove", "AAA"], tmp_path)
        assert removed.exit_code == 0
        assert "empty" in invoke(runner, ["watch", "show"], tmp_path).output.lower()


class TestCache:
    def test_cache_stats_on_an_empty_cache(self, runner, tmp_path):
        result = invoke(runner, ["cache"], tmp_path)
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_cache_purge(self, runner, tmp_path):
        result = invoke(runner, ["cache", "--purge"], tmp_path)
        assert result.exit_code == 0
        assert "Removed" in result.output
