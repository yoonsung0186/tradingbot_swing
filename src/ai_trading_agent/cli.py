from __future__ import annotations

import argparse
import csv
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from time import sleep

from .alpaca import AlpacaError, AlpacaPaperClient
from .backtest import BacktestConfig, buy_and_hold_curve, latest_target_weights, run_backtest
from .charting import write_daily_return_artifacts, write_model_comparison_artifacts, write_simulation_artifacts
from .config import LEVERAGED_ETF_UNIVERSE, MARKET_CONTEXT_SYMBOLS, SECTOR_ETF_UNIVERSE, app_config, resolve_universe
from .data import DataError, YahooChartClient
from .daytrade import DayTradeConfig, NEW_YORK, run_daytrade_backtest
from .daytrade_feedback import (
    format_feedback_table_ko,
    run_feedback_optimization,
    select_best_active_model,
    write_feedback_artifacts,
)
from .execution import AlpacaBroker, DryRunBroker, ExecutionOrder, build_execution_plan, format_execution_plan, save_execution_plan
from .features import compute_features, format_features_table
from .macro import FredMacroProvider
from .official_research import SecClient, format_research
from .optimizer import format_optimization_table, format_walk_forward_table, optimize_strategy, walk_forward_validate
from .paper import PaperPortfolio
from .reporting import (
    format_daytrade_summary_ko,
    format_daytrade_trade_table_ko,
    format_result_summary_ko,
    format_risk_decisions,
    format_signal_table,
    format_swing_summary_ko,
    format_swing_trade_table_ko,
    format_trade_table_ko,
    write_daytrade_trade_log_csv,
    write_markdown_report,
    write_swing_trade_log_csv,
    write_trade_log_csv,
)
from .realtime import RealtimeConfig, run_tick_file_dry_run
from .realtime_session import (
    RealtimeSessionConfig,
    parse_session_until,
    run_realtime_paper_session,
    summarize_realtime_session,
)
from .risk import RiskManager
from .strategy import MomentumStrategy
from .swing import default_swing_configs, run_swing_backtest
from .swing_research import SwingResearchProvider


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent.py",
        description="US stock research and paper trading agent",
    )
    parser.add_argument("--universe", default="etf", choices=["etf", "sector", "mega", "stock", "leveraged", "all"])
    parser.add_argument("--symbols", nargs="*", default=[], help="Extra symbols to include")
    parser.add_argument("--cash", type=float, default=None, help="Override starting cash")

    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("report", help="Generate signals and write a markdown report")
    subparsers.add_parser("paper", help="Run one local paper-trading cycle")
    scan = subparsers.add_parser("scan", help="Collect more history and rank symbols by risk-adjusted features")
    scan.add_argument("--days", type=int, default=1800, help="How many calendar days to fetch")
    scan.add_argument("--limit", type=int, default=20, help="How many rows to print")
    research = subparsers.add_parser("official-research", help="Fetch SEC official filings and official-source queries")
    research.add_argument("--limit", type=int, default=5, help="How many filings per symbol")
    pulse = subparsers.add_parser("market-pulse", help="Check fast intraday market movement and official filing context")
    pulse.add_argument("--interval", default="5m", choices=["1m", "2m", "5m", "15m", "30m", "60m"])
    pulse.add_argument("--range", default="1d", choices=["1d", "5d"])
    pulse.add_argument("--limit", type=int, default=10)
    pulse.add_argument("--research-top", type=int, default=3, help="Fetch SEC context for top N candidates")
    toss = subparsers.add_parser("toss-plan", help="Create a manual Toss Securities order checklist")
    toss.add_argument("--start", default="2023-01-01", help="Simulation start date")
    toss.add_argument("--profile", choices=["balanced", "stable", "aggressive"], default="stable")
    toss.add_argument("--max-mdd", type=float, default=18)
    toss.add_argument("--limit", type=int, default=5, help="How many candidate symbols to print")
    order_plan = subparsers.add_parser("auto-order-plan", help="Create an automated broker order plan without submitting it")
    order_plan.add_argument("--start", default="2023-01-01")
    order_plan.add_argument("--profile", choices=["balanced", "stable", "aggressive"], default="stable")
    order_plan.add_argument("--max-mdd", type=float, default=18)
    order_plan.add_argument("--limit", type=int, default=3)
    order_plan.add_argument("--allocation", type=float, default=0.60, help="Fraction of cash to allocate across candidates")
    order_plan.add_argument("--broker", choices=["alpaca"], default="alpaca")
    submit_plan = subparsers.add_parser("submit-plan", help="Submit an execution plan to dry-run, Alpaca paper, or gated Alpaca live")
    submit_plan.add_argument("plan_path")
    submit_plan.add_argument("--mode", choices=["dry-run", "paper", "live"], default="dry-run")
    submit_plan.add_argument("--i-understand-real-money-risk", action="store_true")
    subparsers.add_parser("alpaca-account", help="Check Alpaca paper account connection")

    alpaca_submit = subparsers.add_parser("alpaca-submit", help="Preview or submit Alpaca paper orders")
    alpaca_submit.add_argument("--confirm", action="store_true", help="Actually submit approved orders")

    backtest = subparsers.add_parser("backtest", help="Run a simple historical strategy backtest")
    backtest.add_argument("--start", default=None, help="Start date, for example 2024-01-01")
    backtest.add_argument("--days", type=int, default=2200, help="How many calendar days to fetch")
    backtest.add_argument("--top", type=int, default=3, help="Number of holdings")
    backtest.add_argument("--rebalance-interval", type=int, default=5, help="Trading days between rebalances")
    backtest.add_argument("--long-window", type=int, default=200, help="Trend filter window")
    backtest.add_argument("--min-momentum", type=float, default=0.0, help="Minimum medium-term momentum")
    backtest.add_argument("--max-volatility", type=float, default=0.45, help="Maximum annualized 3m volatility")
    backtest.add_argument("--stop-loss", type=float, default=0.10, help="Per-position stop loss")
    backtest.add_argument("--cost-bps", type=float, default=5.0, help="Estimated one-way trading cost in bps")
    backtest.add_argument("--slippage-bps", type=float, default=10.0, help="Estimated one-way slippage in bps")
    backtest.add_argument("--min-dollar-volume", type=float, default=20_000_000.0, help="Minimum 20d dollar volume")

    optimize = subparsers.add_parser("optimize", help="Search strategy parameters for return vs drawdown")
    optimize.add_argument("--start", default="2020-01-01", help="Start date, for example 2020-01-01")
    optimize.add_argument("--days", type=int, default=2600, help="How many calendar days to fetch")
    optimize.add_argument("--max-results", type=int, default=10, help="How many optimized configurations to show")
    optimize.add_argument(
        "--profile",
        choices=["balanced", "stable", "aggressive"],
        default="balanced",
        help="Optimization style. Stable limits drawdown; aggressive allows concentration.",
    )
    optimize.add_argument(
        "--max-mdd",
        type=float,
        default=None,
        help="Optional maximum drawdown limit. Use 0.18 or 18 for 18 percent.",
    )
    optimize.add_argument("--cost-bps", type=float, default=5.0, help="Estimated one-way trading cost in bps")
    optimize.add_argument("--slippage-bps", type=float, default=10.0, help="Estimated one-way slippage in bps")
    optimize.add_argument("--min-dollar-volume", type=float, default=20_000_000.0, help="Minimum 20d dollar volume")

    walk = subparsers.add_parser("walk-forward", help="Optimize on past windows and test on later windows")
    walk.add_argument("--start", default="2020-01-01", help="Walk-forward start date")
    walk.add_argument("--end", default=None, help="Walk-forward end date")
    walk.add_argument("--days", type=int, default=3000, help="How many calendar days to fetch")
    walk.add_argument("--train-days", type=int, default=730, help="Calendar days in each training window")
    walk.add_argument("--test-days", type=int, default=180, help="Calendar days in each test window")
    walk.add_argument("--profile", choices=["balanced", "stable", "aggressive"], default="stable")
    walk.add_argument("--max-mdd", type=float, default=None, help="Training maximum drawdown limit")
    walk.add_argument("--cost-bps", type=float, default=5.0, help="Estimated one-way trading cost in bps")
    walk.add_argument("--slippage-bps", type=float, default=10.0, help="Estimated one-way slippage in bps")
    walk.add_argument("--min-dollar-volume", type=float, default=20_000_000.0, help="Minimum 20d dollar volume")

    compare = subparsers.add_parser("compare-models", help="Train profiles on one period and test on a later holdout")
    compare.add_argument("--train-start", default="2020-01-01")
    compare.add_argument("--train-end", default="2023-12-31")
    compare.add_argument("--test-start", default="2024-01-01")
    compare.add_argument("--test-end", default=None)
    compare.add_argument("--days", type=int, default=3000)
    compare.add_argument("--profiles", nargs="*", default=["stable", "balanced", "aggressive"])
    compare.add_argument("--max-mdd", type=float, default=None)
    compare.add_argument("--cost-bps", type=float, default=5.0)
    compare.add_argument("--slippage-bps", type=float, default=15.0)
    compare.add_argument("--min-dollar-volume", type=float, default=20_000_000.0)

    swing = subparsers.add_parser("swing-backtest", help="Run stable/aggressive/catalyst daily swing simulations")
    swing.add_argument("--start", default="2023-01-01", help="Simulation start date")
    swing.add_argument("--end", default=None, help="Simulation end date")
    swing.add_argument("--days", type=int, default=2200, help="Calendar days to fetch before the end date")
    swing.add_argument(
        "--model",
        choices=[
            "stable",
            "aggressive",
            "catalyst",
            "catalyst-rsi",
            "catalyst-pullback4",
            "catalyst-atr",
            "catalyst-atr-strength",
            "catalyst-atr-strength-extend",
            "catalyst-atr-weak-time",
            "catalyst-hold",
            "catalyst-exits",
            "leveraged-overlay-aggressive",
            "leveraged-overlay-improved",
            "leveraged-overlay-regime4",
            "leveraged-overlay",
            "both",
            "all",
        ],
        default="both",
    )
    swing.add_argument("--max-symbols", type=int, default=0, help="Limit stock symbols; 0 means all selected symbols")
    swing.add_argument("--cost-bps", type=float, default=5.0)
    swing.add_argument("--slippage-bps", type=float, default=15.0)
    swing.add_argument("--min-dollar-volume", type=float, default=20_000_000.0)
    swing.add_argument("--research-filter", action=argparse.BooleanOptionalAction, default=True)
    swing.add_argument("--research-source", choices=["sec", "sec-alpaca", "off"], default="sec-alpaca")
    swing.add_argument("--research-news-lookback-days", type=int, default=7)
    swing.add_argument("--research-filing-lookback-days", type=int, default=14)
    swing.add_argument("--research-industry-lookback-days", type=int, default=14)
    swing.add_argument("--research-max-news-pages", type=int, default=8)
    swing.add_argument("--research-block-risk-score", type=int, default=4)
    swing.add_argument("--research-caution-risk-score", type=int, default=2)
    swing.add_argument("--filing-text", action=argparse.BooleanOptionalAction, default=True)
    swing.add_argument("--macro-filter", action=argparse.BooleanOptionalAction, default=True)
    swing.add_argument("--macro-lookback-days", type=int, default=30)
    swing.add_argument("--macro-block-risk-score", type=int, default=1)
    swing.add_argument("--macro-caution-risk-score", type=int, default=1)

    daytrade = subparsers.add_parser("daytrade-backtest", help="Run intraday day-trading simulation")
    daytrade.add_argument("--interval", default="5m", choices=["1m", "2m", "5m", "15m", "30m"])
    daytrade.add_argument("--range", default="60d", choices=["5d", "30d", "60d"])
    daytrade.add_argument("--max-symbols", type=int, default=24, help="Limit symbols to avoid intraday rate limits")
    daytrade.add_argument("--fresh", action="store_true", help="Ignore intraday cache and fetch fresh market data")
    daytrade.add_argument(
        "--primary-indicator",
        default="ma_bollinger",
        choices=["orb_breakout", "vwap_trend", "volume_momentum", "pullback_vwap", "ma_bollinger"],
    )
    daytrade.add_argument("--filters", nargs="*", default=["volume", "bullish"])
    daytrade.add_argument("--market-confirmation", action=argparse.BooleanOptionalAction, default=True)
    daytrade.add_argument("--opening-minutes", type=int, default=30)
    daytrade.add_argument("--max-trades-per-day", type=int, default=3)
    daytrade.add_argument("--per-trade-risk", type=float, default=0.01)
    daytrade.add_argument("--max-position-pct", type=float, default=0.40)
    daytrade.add_argument("--max-daily-loss", type=float, default=0.03)
    daytrade.add_argument("--stop-pct", type=float, default=0.005)
    daytrade.add_argument("--take-profit-r", type=float, default=2.0)
    daytrade.add_argument("--volume-ratio", type=float, default=1.6)
    daytrade.add_argument("--min-bar-dollar-volume", type=float, default=3_000_000.0)
    daytrade.add_argument("--min-momentum", type=float, default=0.0015)
    daytrade.add_argument("--fast-ema", type=int, default=9)
    daytrade.add_argument("--slow-ema", type=int, default=20)
    daytrade.add_argument("--bollinger-window", type=int, default=20)
    daytrade.add_argument("--bollinger-std", type=float, default=2.0)
    daytrade.add_argument("--bollinger-position", type=float, default=0.85)
    daytrade.add_argument("--vwap-buffer", type=float, default=0.0008)
    daytrade.add_argument("--last-entry-minutes-before-close", type=int, default=60)
    daytrade.add_argument("--cost-bps", type=float, default=5.0)
    daytrade.add_argument("--slippage-bps", type=float, default=15.0)

    feedback = subparsers.add_parser("daytrade-feedback-optimize", help="Run 10+ feedback simulations and rank intraday models")
    feedback.add_argument("--interval", default="1m", choices=["1m", "2m", "5m", "15m", "30m"])
    feedback.add_argument("--range", default="5d", choices=["5d", "30d", "60d"])
    feedback.add_argument("--max-symbols", type=int, default=24)
    feedback.add_argument("--fresh", action="store_true")
    feedback.add_argument("--train-fraction", type=float, default=0.6)
    feedback.add_argument("--cost-bps", type=float, default=5.0)
    feedback.add_argument("--slippage-bps", type=float, default=15.0)

    indicator = subparsers.add_parser("daytrade-indicators", help="Compare intraday indicators and select a model")
    indicator.add_argument("--interval", default="5m", choices=["1m", "2m", "5m", "15m", "30m"])
    indicator.add_argument("--range", default="60d", choices=["5d", "30d", "60d"])
    indicator.add_argument("--max-symbols", type=int, default=24)
    indicator.add_argument("--fresh", action="store_true")
    indicator.add_argument("--train-fraction", type=float, default=0.5)
    indicator.add_argument("--cost-bps", type=float, default=5.0)
    indicator.add_argument("--slippage-bps", type=float, default=15.0)

    realtime = subparsers.add_parser("realtime-dry-run", help="Read 10-second ticks from a JSONL feed and create dry-run decisions")
    realtime.add_argument("--tick-file", default="data/toss_ticks.jsonl")
    realtime.add_argument("--watch-seconds", type=int, default=0)
    realtime.add_argument("--poll-seconds", type=int, default=10)
    realtime.add_argument("--lookback-ticks", type=int, default=6)
    realtime.add_argument("--entry-momentum", type=float, default=0.003)
    realtime.add_argument("--stop-loss", type=float, default=0.004)
    realtime.add_argument("--take-profit", type=float, default=0.008)
    realtime.add_argument("--trailing-stop", type=float, default=0.004)
    realtime.add_argument("--max-position-pct", type=float, default=0.10)
    paper_live = subparsers.add_parser("realtime-paper", help="Run a live local paper-trading session from market data")
    paper_live.add_argument("--provider", choices=["alpaca", "yahoo"], default="alpaca")
    paper_live.add_argument("--feed", default="iex", help="Alpaca stock data feed, usually iex for free/basic accounts")
    paper_live.add_argument("--until", default=None, help="KST time like 09:00 or ISO datetime")
    paper_live.add_argument("--poll-seconds", type=int, default=10)
    paper_live.add_argument(
        "--risk-mode",
        choices=["stable", "balanced", "aggressive", "surge_runner", "hybrid_runner", "profit_runner"],
        default="stable",
    )
    paper_live.add_argument("--max-symbols", type=int, default=10)
    paper_live.add_argument("--wait-for-credentials", action="store_true")
    paper_live.add_argument("--credential-check-seconds", type=int, default=60)
    paper_live.add_argument("--dynamic-universe", action="store_true", help="Continuously add volume/dollar-volume surge candidates")
    paper_live.add_argument("--scan-interval-seconds", type=int, default=180)
    paper_live.add_argument("--scan-max-symbols", type=int, default=60)
    paper_live.add_argument("--dynamic-max-symbols", type=int, default=16)
    paper_live.add_argument("--top-surging", type=int, default=6)
    paper_live.add_argument("--min-volume-ratio", type=float, default=1.6)
    paper_live.add_argument("--min-recent-dollar-volume", type=float, default=3_000_000.0)
    paper_live.add_argument("--min-short-return", type=float, default=0.0015)
    paper_live.add_argument(
        "--research-filter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use official SEC filings and Alpaca news as an auxiliary entry risk filter",
    )
    paper_live.add_argument("--research-interval-seconds", type=int, default=300)
    paper_live.add_argument("--research-max-symbols", type=int, default=20)
    paper_live.add_argument("--research-news-lookback-minutes", type=int, default=180)
    paper_live.add_argument("--research-block-risk-score", type=int, default=4)
    paper_live.add_argument("--research-caution-risk-score", type=int, default=2)

    supervise = subparsers.add_parser("supervise-realtime", help="Run realtime paper trading with automatic restart on errors")
    supervise.add_argument("--provider", choices=["alpaca", "yahoo"], default="alpaca")
    supervise.add_argument("--feed", default="iex")
    supervise.add_argument("--until", default=None)
    supervise.add_argument("--poll-seconds", type=int, default=10)
    supervise.add_argument(
        "--risk-mode",
        choices=["stable", "balanced", "aggressive", "surge_runner", "hybrid_runner", "profit_runner"],
        default="hybrid_runner",
    )
    supervise.add_argument("--max-symbols", type=int, default=10)
    supervise.add_argument("--wait-for-credentials", action="store_true")
    supervise.add_argument("--credential-check-seconds", type=int, default=60)
    supervise.add_argument("--dynamic-universe", action="store_true")
    supervise.add_argument("--scan-interval-seconds", type=int, default=180)
    supervise.add_argument("--scan-max-symbols", type=int, default=60)
    supervise.add_argument("--dynamic-max-symbols", type=int, default=16)
    supervise.add_argument("--top-surging", type=int, default=6)
    supervise.add_argument("--min-volume-ratio", type=float, default=1.6)
    supervise.add_argument("--min-recent-dollar-volume", type=float, default=3_000_000.0)
    supervise.add_argument("--min-short-return", type=float, default=0.0015)
    supervise.add_argument("--research-filter", action=argparse.BooleanOptionalAction, default=True)
    supervise.add_argument("--research-interval-seconds", type=int, default=300)
    supervise.add_argument("--research-max-symbols", type=int, default=20)
    supervise.add_argument("--research-news-lookback-minutes", type=int, default=180)
    supervise.add_argument("--research-block-risk-score", type=int, default=4)
    supervise.add_argument("--research-caution-risk-score", type=int, default=2)
    supervise.add_argument("--restart-delay-seconds", type=int, default=15)
    supervise.add_argument("--max-restarts", type=int, default=20)

    paper_summary = subparsers.add_parser("realtime-summary", help="Summarize the latest live paper-trading session")
    paper_summary.add_argument("--session-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = app_config()
    starting_cash = args.cash if args.cash is not None else cfg.starting_cash
    symbols = resolve_universe(args.universe, args.symbols)

    try:
        if args.command == "alpaca-account":
            return handle_alpaca_account()
        if args.command == "backtest":
            return handle_backtest(args, symbols, starting_cash)
        if args.command == "optimize":
            return handle_optimize(args, symbols, starting_cash)
        if args.command == "walk-forward":
            return handle_walk_forward(args, symbols, starting_cash)
        if args.command == "compare-models":
            return handle_compare_models(args, symbols, starting_cash)
        if args.command == "swing-backtest":
            return handle_swing_backtest(args, symbols, starting_cash)
        if args.command == "daytrade-backtest":
            return handle_daytrade_backtest(args, symbols, starting_cash)
        if args.command == "daytrade-feedback-optimize":
            return handle_daytrade_feedback_optimize(args, symbols, starting_cash)
        if args.command == "daytrade-indicators":
            return handle_daytrade_indicators(args, symbols, starting_cash)
        if args.command == "realtime-dry-run":
            return handle_realtime_dry_run(args, symbols, starting_cash)
        if args.command == "realtime-paper":
            return handle_realtime_paper(args, symbols, starting_cash)
        if args.command == "supervise-realtime":
            return handle_supervise_realtime(args, symbols, starting_cash)
        if args.command == "realtime-summary":
            return handle_realtime_summary(args)
        if args.command == "official-research":
            return handle_official_research(args, symbols)
        if args.command == "market-pulse":
            return handle_market_pulse(args, symbols)
        if args.command == "toss-plan":
            return handle_toss_plan(args, symbols, starting_cash)
        if args.command == "auto-order-plan":
            return handle_auto_order_plan(args, symbols, starting_cash)
        if args.command == "submit-plan":
            return handle_submit_plan(args)
        if args.command == "scan":
            return handle_scan(args, symbols)
        return handle_signals(args, symbols, cfg.state_file, starting_cash)
    except (DataError, AlpacaError, ValueError) as exc:
        print(f"ERROR: {exc}")
        return 1


def fetch_histories(symbols: list[str], start: date | None = None, end: date | None = None) -> dict[str, list]:
    client = YahooChartClient()
    histories = {}
    for symbol in symbols:
        try:
            histories[symbol] = client.history(symbol, start=start, end=end)
        except DataError as exc:
            print(f"WARNING: skipping {symbol}: {exc}")
    if not histories:
        raise DataError("No price data could be fetched")
    return histories


def fetch_intraday_histories(
    symbols: list[str],
    interval: str,
    range_: str,
    cache_ttl_hours: int = 1,
) -> dict[str, list]:
    client = YahooChartClient(cache_ttl_hours=cache_ttl_hours)
    histories = {}
    for symbol in symbols:
        try:
            histories[symbol] = client.intraday_history(symbol, interval=interval, range_=range_)
        except DataError as exc:
            print(f"WARNING: skipping {symbol}: {exc}")
    if not histories:
        raise DataError("No intraday price data could be fetched")
    return histories


def with_context_symbols(symbols: list[str]) -> list[str]:
    required = ["SPY", "^VIX", "SHY"]
    seen: set[str] = set()
    merged: list[str] = []
    for symbol in [*symbols, *required]:
        upper = symbol.upper()
        if upper not in seen:
            seen.add(upper)
            merged.append(upper)
    return merged


def handle_signals(args: argparse.Namespace, symbols: list[str], state_file, starting_cash: float) -> int:
    histories = fetch_histories(symbols)
    if "SPY" not in histories:
        histories["SPY"] = YahooChartClient().history("SPY")

    strategy = MomentumStrategy()
    risk_on, regime_note = strategy.market_regime(histories["SPY"])
    signals = strategy.generate(histories, risk_on)
    latest_prices = {symbol: bars[-1].close for symbol, bars in histories.items()}

    portfolio = PaperPortfolio(state_file, starting_cash)
    snapshot = portfolio.snapshot(latest_prices)
    risk = RiskManager()
    decisions = [risk.review_buy(signal, snapshot) for signal in signals]
    report_path = write_markdown_report(regime_note, signals, decisions, snapshot)

    print(f"Market regime: {regime_note}")
    print()
    print(format_signal_table(signals))
    print()
    print(format_risk_decisions(decisions))
    print()
    print(f"Report written: {report_path}")

    if args.command == "paper":
        approved = [decision.order for decision in decisions if decision.allowed and decision.order]
        for order in approved:
            trade = portfolio.execute(order)
            print(
                f"PAPER BUY {trade['symbol']} qty={trade['qty']} "
                f"price={trade['price']:.2f} notional={trade['notional']:.2f}"
            )
        if not approved:
            print("No approved paper orders.")
        return 0

    if args.command == "alpaca-submit":
        approved = [decision.order for decision in decisions if decision.allowed and decision.order]
        if not approved:
            print("No approved Alpaca paper orders.")
            return 0
        if not args.confirm:
            print("Preview only. Add --confirm to submit these orders to Alpaca paper trading.")
            return 0
        client = AlpacaPaperClient()
        for order in approved:
            response = client.submit_order(order)
            print(f"Submitted {order.symbol} qty={order.qty}: id={response.get('id', 'unknown')}")
        return 0

    return 0


def handle_scan(args: argparse.Namespace, symbols: list[str]) -> int:
    end = date.today()
    start_fetch = end - timedelta(days=args.days)
    histories = fetch_histories(with_context_symbols(symbols), start=start_fetch, end=end)
    features = compute_features(histories)
    print(format_features_table(features, limit=args.limit))
    return 0


def handle_official_research(args: argparse.Namespace, symbols: list[str]) -> int:
    client = SecClient()
    items = []
    for symbol in symbols:
        try:
            items.append(client.research(symbol, limit=args.limit))
        except Exception as exc:
            print(f"WARNING: official research skipped for {symbol}: {exc}")
    if not items:
        raise DataError("No official research items were produced")
    print(format_research(items))
    return 0


def handle_market_pulse(args: argparse.Namespace, symbols: list[str]) -> int:
    client = YahooChartClient(cache_ttl_hours=0)
    snapshots = []
    for symbol in symbols:
        try:
            item = client.intraday_snapshot(symbol, interval=args.interval, range_=args.range)
        except Exception as exc:
            print(f"WARNING: intraday snapshot skipped for {symbol}: {exc}")
            continue
        score = item["day_return"] * 2 + item["short_return"]
        snapshots.append((score, item))
    snapshots.sort(key=lambda item: item[0], reverse=True)
    if not snapshots:
        raise DataError("No market pulse snapshots were produced")

    print("rank symbol price day_return short_return recent_volume source")
    print("---- ------ ----- ---------- ------------ ------------- ------")
    for rank, (_, item) in enumerate(snapshots[: args.limit], start=1):
        print(
            f"{rank:4} {item['symbol']:6} {item['price']:7.2f} "
            f"{item['day_return']:10.2%} {item['short_return']:12.2%} "
            f"{item['recent_volume']:13} {item['source']}"
        )

    if args.research_top > 0:
        sec = SecClient()
        research_items = []
        for _, item in snapshots[: args.research_top]:
            try:
                research_items.append(sec.research(item["symbol"], limit=2))
            except Exception as exc:
                print(f"WARNING: SEC research skipped for {item['symbol']}: {exc}")
        if research_items:
            print()
            print("Official context")
            print(format_research(research_items))
    return 0


def handle_backtest(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    end = date.today()
    start_fetch = end - timedelta(days=args.days)
    histories = fetch_histories(with_context_symbols(symbols), start=start_fetch, end=end)
    start = date.fromisoformat(args.start) if args.start else None
    config = BacktestConfig(
        top_n=args.top,
        rebalance_interval=args.rebalance_interval,
        long_window=args.long_window,
        medium_window=min(126, args.long_window),
        min_momentum=args.min_momentum,
        max_volatility=args.max_volatility,
        stop_loss_pct=args.stop_loss,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        min_dollar_volume=args.min_dollar_volume,
        tradable_symbols=tuple(symbols),
    )
    result = run_backtest(histories, cash=cash, start=start, config=config)
    benchmark = buy_and_hold_curve(histories["SPY"], cash, start=start)
    csv_path, svg_path, png_path = write_simulation_artifacts(result["equity_curve"], benchmark)
    daily_csv_path, daily_png_path = write_daily_return_artifacts(result["equity_curve"])
    trade_csv_path = write_trade_log_csv(result["trade_log"])
    print(format_result_summary_ko(result, title="백테스트 결과"))
    print()
    print(f"CSV 저장 위치:      {csv_path}")
    print(f"총자산 그래프 SVG:  {svg_path}")
    if png_path:
        print(f"총자산 그래프 PNG:  {png_path}")
    print(f"일별 수익률 CSV:    {daily_csv_path}")
    if daily_png_path:
        print(f"일별 수익률 PNG:    {daily_png_path}")
    print(f"거래 내역 CSV:      {trade_csv_path}")
    print()
    print(format_trade_table_ko(result["trade_log"], limit=25))
    return 0


def handle_optimize(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    end = date.today()
    start_fetch = end - timedelta(days=args.days)
    start = date.fromisoformat(args.start) if args.start else None
    max_mdd = args.max_mdd
    if max_mdd and max_mdd > 1:
        max_mdd = max_mdd / 100
    histories = fetch_histories(with_context_symbols(symbols), start=start_fetch, end=end)
    results = optimize_strategy(
        histories,
        cash=cash,
        start=start,
        max_results=args.max_results,
        max_mdd=max_mdd,
        profile=args.profile,
        tradable_symbols=tuple(symbols),
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        min_dollar_volume=args.min_dollar_volume,
    )
    if not results:
        raise DataError("No optimization results were produced")
    print(format_optimization_table(results))
    best = results[0]
    benchmark = buy_and_hold_curve(histories["SPY"], cash, start=start)
    csv_path, svg_path, png_path = write_simulation_artifacts(
        best.result["equity_curve"],
        benchmark,
        label="optimized_simulation",
    )
    daily_csv_path, daily_png_path = write_daily_return_artifacts(
        best.result["equity_curve"],
        label="optimized_daily_returns",
    )
    trade_csv_path = write_trade_log_csv(best.result["trade_log"], label="optimized_trade_log")
    print()
    print("## 선택된 최적 설정")
    for key, value in best.result["config"].items():
        print(f"- {key}: {value}")
    print()
    print(format_result_summary_ko(best.result, title="최적화 모델 시뮬레이션 결과"))
    print()
    print(f"CSV 저장 위치:      {csv_path}")
    print(f"총자산 그래프 SVG:  {svg_path}")
    if png_path:
        print(f"총자산 그래프 PNG:  {png_path}")
    print(f"일별 수익률 CSV:    {daily_csv_path}")
    if daily_png_path:
        print(f"일별 수익률 PNG:    {daily_png_path}")
    print(f"거래 내역 CSV:      {trade_csv_path}")
    print()
    print(format_trade_table_ko(best.result["trade_log"], limit=25))
    return 0


def handle_walk_forward(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    end = date.fromisoformat(args.end) if args.end else date.today()
    start_fetch = end - timedelta(days=args.days)
    start = date.fromisoformat(args.start)
    max_mdd = args.max_mdd
    if max_mdd and max_mdd > 1:
        max_mdd = max_mdd / 100
    histories = fetch_histories(with_context_symbols(symbols), start=start_fetch, end=end)
    folds = walk_forward_validate(
        histories,
        cash=cash,
        start=start,
        end=end,
        train_days=args.train_days,
        test_days=args.test_days,
        profile=args.profile,
        max_mdd=max_mdd,
        tradable_symbols=tuple(symbols),
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        min_dollar_volume=args.min_dollar_volume,
    )
    if not folds:
        raise DataError("No walk-forward folds were produced")

    print(format_walk_forward_table(folds))
    test_returns = [fold.test_result["total_return"] for fold in folds]
    test_mdds = [fold.test_result["max_drawdown"] for fold in folds]
    test_sharpes = [fold.test_result["sharpe"] for fold in folds]
    positive = [item for item in test_returns if item > 0]
    print()
    print("## 워크포워드 검증 요약")
    print()
    print("| 항목 | 값 |")
    print("| --- | ---: |")
    print(f"| 검증 구간 수 | {len(folds):,}개 |")
    print(f"| 플러스 구간 | {len(positive):,}/{len(folds):,}개 |")
    print(f"| 평균 테스트 수익률 | {sum(test_returns) / len(test_returns):.2%} |")
    print(f"| 최악 테스트 수익률 | {min(test_returns):.2%} |")
    print(f"| 평균 테스트 MDD | {sum(test_mdds) / len(test_mdds):.2%} |")
    print(f"| 최악 테스트 MDD | {min(test_mdds):.2%} |")
    print(f"| 평균 테스트 Sharpe | {sum(test_sharpes) / len(test_sharpes):.2f} |")
    return 0


def handle_compare_models(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    test_end = date.fromisoformat(args.test_end) if args.test_end else date.today()
    train_start = date.fromisoformat(args.train_start)
    train_end = date.fromisoformat(args.train_end)
    test_start = date.fromisoformat(args.test_start)
    start_fetch = train_start - timedelta(days=900)
    histories = fetch_histories(with_context_symbols(symbols), start=start_fetch, end=test_end)
    max_mdd = args.max_mdd
    if max_mdd and max_mdd > 1:
        max_mdd = max_mdd / 100

    curves: dict[str, list] = {}
    rows = [
        "## 모델별 홀드아웃 검증 결과",
        "",
        "| 모델 | 훈련 수익률 | 홀드아웃 최종 총자산 | 홀드아웃 수익률 | 홀드아웃 MDD | Sharpe | Calmar | 거래 횟수 | 일별 그래프 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for profile in args.profiles:
        optimized = optimize_strategy(
            histories,
            cash=cash,
            start=train_start,
            end=train_end,
            max_results=1,
            max_mdd=max_mdd,
            profile=profile,
            tradable_symbols=tuple(symbols),
            cost_bps=args.cost_bps,
            slippage_bps=args.slippage_bps,
            min_dollar_volume=args.min_dollar_volume,
        )
        if not optimized:
            rows.append(f"| {profile} | 결과 없음 | - | - | - | - | - | - | - |")
            continue
        best = optimized[0]
        holdout = run_backtest(
            histories,
            cash=cash,
            start=test_start,
            end=test_end,
            config=best.config,
        )
        curves[profile] = holdout["equity_curve"]
        daily_csv, daily_png = write_daily_return_artifacts(
            holdout["equity_curve"],
            label=f"{profile}_holdout_daily_returns",
        )
        rows.append(
            f"| {profile} | {best.result['total_return']:.1%} | ${holdout['end_equity']:,.2f} | "
            f"{holdout['total_return']:.1%} | {holdout['max_drawdown']:.1%} | "
            f"{holdout['sharpe']:.2f} | {holdout['calmar']:.2f} | {holdout['trades']:,}회 | "
            f"{daily_png or daily_csv} |"
        )

    comparison_csv, comparison_png = write_model_comparison_artifacts(curves)
    print("\n".join(rows))
    print()
    print(f"모델 비교 CSV: {comparison_csv}")
    if comparison_png:
        print(f"모델별 총자산 그래프 PNG: {comparison_png}")
    return 0


def handle_swing_backtest(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    selected_symbols = symbols[: args.max_symbols] if args.max_symbols and args.max_symbols > 0 else symbols
    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date.today()
    start_fetch = min(start - timedelta(days=520), end - timedelta(days=args.days))
    if args.model == "both":
        model_names = ["stable", "aggressive"]
    elif args.model == "all":
        model_names = ["stable", "aggressive", "catalyst_rsi", "catalyst_pullback4", "catalyst_atr"]
    elif args.model == "catalyst-exits":
        model_names = ["catalyst_rsi", "catalyst_pullback4", "catalyst_atr"]
    elif args.model == "catalyst-hold":
        model_names = ["catalyst_atr_strength_extend", "catalyst_atr_weak_time"]
    elif args.model == "leveraged-overlay":
        model_names = ["leveraged_overlay_aggressive", "leveraged_overlay_improved", "leveraged_overlay_regime_4stage"]
    elif args.model == "leveraged-overlay-regime4":
        model_names = ["leveraged_overlay_regime_4stage"]
    elif args.model in {"catalyst-atr-strength", "catalyst-atr-strength-extend"}:
        model_names = ["catalyst_atr_strength_extend"]
    elif args.model == "catalyst-atr-weak-time":
        model_names = ["catalyst_atr_weak_time"]
    else:
        model_names = [args.model.replace("-", "_")]
    fetch_symbols = _merge_symbols(selected_symbols, [*MARKET_CONTEXT_SYMBOLS, *SECTOR_ETF_UNIVERSE])
    histories = fetch_histories(fetch_symbols, start=start_fetch, end=end)
    research_provider = None
    if args.research_filter and args.research_source != "off":
        print("뉴스/공시 보조지표 수집 중입니다. 백테스트에서는 신호일 기준 공개 자료만 사용합니다.")
        research_provider = SwingResearchProvider(
            selected_symbols,
            start=start,
            end=end,
            source=args.research_source,
            news_lookback_days=args.research_news_lookback_days,
            filing_lookback_days=args.research_filing_lookback_days,
            industry_lookback_days=args.research_industry_lookback_days,
            block_risk_score=args.research_block_risk_score,
            caution_risk_score=args.research_caution_risk_score,
            include_filing_text=args.filing_text,
            max_news_pages_per_chunk=args.research_max_news_pages,
        )
    macro_provider = None
    if args.macro_filter:
        print("FRED macro filter loading. Backtest uses only observations with realtime_start <= signal date.")
        macro_provider = FredMacroProvider(
            start=start,
            end=end,
            lookback_days=args.macro_lookback_days,
            block_risk_score=args.macro_block_risk_score,
            caution_risk_score=args.macro_caution_risk_score,
        )
    configs = default_swing_configs(
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        min_dollar_volume=args.min_dollar_volume,
        tradable_symbols=tuple(selected_symbols),
    )
    benchmark = buy_and_hold_curve(histories["SPY"], cash, start=start)
    curves: dict[str, list] = {}
    results: dict[str, dict] = {}

    print("## 스윙 모델 백테스트")
    print()
    print("| 항목 | 값 |")
    print("| --- | --- |")
    print(f"| 기간 | {start.isoformat()} ~ {end.isoformat()} |")
    print(f"| 대상 종목 수 | {len(selected_symbols):,}개 |")
    print(f"| 비교 모델 | {', '.join(model_names)} |")
    print(f"| 체결 규칙 | T일 종가로 신호 생성, T+1 시가 체결 |")
    print(f"| 거래비용+슬리피지 | {(args.cost_bps + args.slippage_bps):.1f}bps/방향 |")
    print(
        f"| 뉴스/공시 보조지표 | "
        f"{'사용' if research_provider else '미사용'}"
        f"{' / ' + args.research_source if research_provider else ''} |"
    )
    print()

    for name in model_names:
        config = configs[name]
        result = run_swing_backtest(
            histories,
            cash=cash,
            start=start,
            end=end,
            config=config,
            research_provider=research_provider,
            macro_provider=macro_provider,
        )
        results[name] = result
        curves[f"swing_{name}"] = result["equity_curve"]
        csv_path, svg_path, png_path = write_simulation_artifacts(
            result["equity_curve"],
            benchmark,
            label=f"swing_{name}_equity",
        )
        daily_csv_path, daily_png_path = write_daily_return_artifacts(
            result["equity_curve"],
            label=f"swing_{name}_daily_returns",
        )
        trade_csv_path = write_swing_trade_log_csv(
            result["closed_trades"],
            label=f"swing_{name}_trade_log",
        )

        model_title = _swing_model_label(name)
        print(format_swing_summary_ko(result, title=f"{model_title} 결과"))
        print()
        if result.get("research_summary"):
            print("### 뉴스/공시 보조지표 요약")
            print()
            print(_format_research_summary_table(result["research_summary"]))
            print()
        if result.get("macro_summary"):
            print("### FRED macro filter summary")
            print()
            print(_format_macro_summary_table(result["macro_summary"]))
            print()
        print("### 설정")
        print()
        print("| 항목 | 값 |")
        print("| --- | ---: |")
        print(f"| 종목당 최대 비중 | {config.max_position_pct:.0%} |")
        print(f"| 최대 보유 종목 | {config.max_positions:,}개 |")
        print(f"| 1차 익절 | +{config.partial_take_profit_pct:.1%} 도달 시 {config.partial_take_profit_fraction:.0%} 매도 |")
        print(f"| 트레일링 | 고점 대비 -{config.trailing_stop_pct:.1%} |")
        print(f"| 과열 기준 | RSI {config.overheat_rsi_threshold:.0f} 이상 |")
        print(f"| 과열 후 청산 | {_format_overheat_exit_rule(config)} |")
        print(f"| 최대 보유기간 | {config.max_hold_days:,}거래일 |")
        print(f"| 보유기간 처리 | {_format_time_exit_rule(config)} |")
        print(f"| 레버리지 ETF 최대 비중 | {config.leveraged_max_position_pct:.0%} |")
        print()
        print(f"총자산 CSV:       {csv_path}")
        print(f"총자산 그래프 SVG:{svg_path}")
        if png_path:
            print(f"총자산 그래프 PNG:{png_path}")
        print(f"일별 수익률 CSV:  {daily_csv_path}")
        if daily_png_path:
            print(f"일별 수익률 PNG:  {daily_png_path}")
        print(f"청산 거래 CSV:    {trade_csv_path}")
        print()
        print(format_swing_trade_table_ko(result["closed_trades"], limit=20))
        print()

    if len(curves) > 1:
        comparison_csv, comparison_png = write_model_comparison_artifacts(curves, label="swing_model_comparison")
        print("## 모델 비교 그래프")
        print()
        print(f"모델 비교 CSV: {comparison_csv}")
        if comparison_png:
            print(f"모델별 총자산 그래프 PNG: {comparison_png}")
        print()
        print("| 모델 | 최종 총자산 | 누적 수익률 | MDD | 거래 수 | 승률 | Profit Factor |")
        print("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
        for name, result in results.items():
            profit_factor = result["profit_factor"]
            profit_factor_text = "무한대" if profit_factor == float("inf") else f"{profit_factor:.2f}"
            label = _swing_model_label(name).replace(" 스윙", "")
            print(
                f"| {label} | ${result['end_equity']:,.2f} | {result['total_return']:.2%} | "
                f"{result['max_drawdown']:.2%} | {result['trades']:,} | "
                f"{result['win_rate']:.2%} | {profit_factor_text} |"
            )
    print()
    print("실제 주문은 실행하지 않았습니다. 이 명령은 과거 일봉 기반 스윙 백테스트입니다.")
    return 0


def _swing_model_label(name: str) -> str:
    labels = {
        "stable": "안정형 스윙",
        "aggressive": "공격형 스윙",
        "catalyst": "촉매 모멘텀 스윙",
        "catalyst_rsi": "촉매 RSI 즉시청산",
        "catalyst_pullback4": "촉매 고점-4% 청산",
        "catalyst_atr": "촉매 ATR 청산",
        "catalyst_atr_strength_extend": "ATR 추세연장",
        "catalyst_atr_weak_time": "ATR 약세만 시간청산",
        "leveraged_overlay_aggressive": "레버리지 오버레이 공격형",
        "leveraged_overlay_improved": "improved aggressive experiment",
        "leveraged_overlay_regime_4stage": "4-stage regime improved",
    }
    return labels.get(name, name)


def _format_overheat_exit_rule(config) -> str:
    if config.overheat_trailing_atr_multiple > 0:
        return f"과열 감지 후 고점 대비 ATR {config.overheat_trailing_atr_multiple:.1f}배 이탈"
    if config.overheat_trailing_stop_pct > 0:
        return f"과열 감지 후 고점 대비 -{config.overheat_trailing_stop_pct:.1%}"
    return "과열 즉시 청산"


def _format_time_exit_rule(config) -> str:
    if config.time_exit_mode == "evidence_extend":
        return (
            f"{config.max_hold_days}거래일 이후 추세/공식근거 유지 시 조건부 유예"
            f"(일반 {config.max_extended_hold_days}일, 레버리지 {config.leveraged_max_extended_hold_days}일 상한)"
        )
    if config.time_exit_mode == "strength_extend":
        return "35거래일 도달 후에도 EMA20 위 + 상대강도 유지 시 연장"
    if config.time_exit_mode == "weak_only":
        return f"35거래일 도달 시 손실/횡보(+{config.time_exit_sideways_return_pct:.1%} 이하)만 청산"
    return "35거래일 도달 시 청산"


def handle_daytrade_backtest(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    selected_symbols = symbols[: args.max_symbols] if args.max_symbols and args.max_symbols > 0 else symbols
    interval_minutes = _interval_minutes(args.interval)
    config = DayTradeConfig(
        model_name=args.primary_indicator,
        primary_indicator=args.primary_indicator,
        enabled_filters=tuple(args.filters),
        opening_minutes=args.opening_minutes,
        interval_minutes=interval_minutes,
        per_trade_risk=_normalize_pct(args.per_trade_risk),
        max_position_pct=_normalize_pct(args.max_position_pct),
        max_daily_loss_pct=_normalize_pct(args.max_daily_loss),
        max_trades_per_day=args.max_trades_per_day,
        stop_pct=_normalize_pct(args.stop_pct),
        take_profit_r=args.take_profit_r,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
        min_signal_volume_ratio=args.volume_ratio,
        min_bar_dollar_volume=args.min_bar_dollar_volume,
        min_momentum_pct=_normalize_pct(args.min_momentum),
        fast_ema_window=args.fast_ema,
        slow_ema_window=args.slow_ema,
        bollinger_window=args.bollinger_window,
        bollinger_std=args.bollinger_std,
        min_bollinger_position=args.bollinger_position,
        vwap_buffer_pct=_normalize_pct(args.vwap_buffer),
        require_market_confirmation=args.market_confirmation,
        last_entry_minutes_before_close=args.last_entry_minutes_before_close,
    )
    fetch_symbols = _merge_symbols(selected_symbols, list(config.market_symbols))
    histories = fetch_intraday_histories(
        fetch_symbols,
        interval=args.interval,
        range_=args.range,
        cache_ttl_hours=0 if args.fresh else 1,
    )
    result = run_daytrade_backtest(histories, cash=cash, config=config)
    csv_path, svg_path, png_path = write_simulation_artifacts(
        result["equity_curve"],
        [],
        label="daytrade_equity",
    )
    daily_csv_path, daily_png_path = write_daily_return_artifacts(
        result["equity_curve"],
        label="daytrade_daily_returns",
    )
    trade_csv_path = write_daytrade_trade_log_csv(result["trade_log"])

    print("## 단타 전략")
    print()
    print("| 항목 | 값 |")
    print("| --- | --- |")
    print("| 전략 | 장 초반 변동폭 돌파 + VWAP 확인 |")
    print(f"| 봉 주기 | {args.interval} |")
    print(f"| 데이터 범위 | {args.range} |")
    print(f"| 대상 종목 | {', '.join(selected_symbols)} |")
    print(f"| 시장 확인 | {', '.join(config.market_symbols) if config.require_market_confirmation else '사용 안 함'} |")
    print(f"| 하루 최대 거래 | {args.max_trades_per_day}회 |")
    print(f"| 하루 손실 제한 | {_normalize_pct(args.max_daily_loss):.2%} |")
    print(f"| 신규 진입 제한 | 장마감 {args.last_entry_minutes_before_close}분 전부터 금지 |")
    print()
    print(format_daytrade_summary_ko(result))
    print()
    print(f"총자산 CSV:       {csv_path}")
    print(f"총자산 그래프 SVG:{svg_path}")
    if png_path:
        print(f"총자산 그래프 PNG:{png_path}")
    print(f"일별 수익률 CSV:  {daily_csv_path}")
    if daily_png_path:
        print(f"일별 수익률 PNG:  {daily_png_path}")
    print(f"단타 거래 CSV:    {trade_csv_path}")
    print()
    print(format_daytrade_trade_table_ko(result["trade_log"], limit=30))
    print()
    print("실제 주문은 실행하지 않았습니다. 이 명령은 intraday 백테스트 전용입니다.")
    return 0


def handle_daytrade_feedback_optimize(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    selected_symbols = symbols[: args.max_symbols] if args.max_symbols and args.max_symbols > 0 else symbols
    interval_minutes = _interval_minutes(args.interval)
    context_symbols = ["SPY", "QQQ"]
    histories = fetch_intraday_histories(
        _merge_symbols(selected_symbols, context_symbols),
        interval=args.interval,
        range_=args.range,
        cache_ttl_hours=0 if args.fresh else 1,
    )
    evaluations = run_feedback_optimization(
        histories,
        cash=cash,
        interval_minutes=interval_minutes,
        train_fraction=args.train_fraction,
        cost_bps=args.cost_bps,
        slippage_bps=args.slippage_bps,
    )
    best = select_best_active_model(evaluations)
    feedback_csv, comparison_png, best_trade_csv = write_feedback_artifacts(evaluations)
    best_csv, best_svg, best_png = write_simulation_artifacts(
        best.full_result["equity_curve"],
        [],
        label="daytrade_feedback_best_equity",
    )
    daily_csv, daily_png = write_daily_return_artifacts(
        best.full_result["equity_curve"],
        label="daytrade_feedback_best_daily_returns",
    )

    print("## 단타 피드백 최적화")
    print()
    print(f"- 후보 모델 수: {len(evaluations)}개")
    print(f"- 데이터: {args.range} {args.interval}")
    print(f"- 대상 종목: {', '.join(selected_symbols)}")
    print(f"- 훈련/검증 분리: {args.train_fraction:.0%}/{1 - args.train_fraction:.0%}")
    print("- 점수 기준: 검증 수익률, 검증 MDD, 손익비, 훈련-검증 괴리, 과도한 거래 빈도 패널티")
    print()
    print(format_feedback_table_ko(evaluations))
    print()
    print("## 선택된 최적 활성 모델")
    print()
    print("| 항목 | 값 |")
    print("| --- | --- |")
    print(f"| 모델 | {best.name} |")
    print(f"| 신뢰도 | {best.confidence} |")
    print(f"| 주 지표 | {best.config.primary_indicator} |")
    print(f"| 보조 필터 | {', '.join(best.config.enabled_filters)} |")
    print(f"| 하루 최대 거래 | {best.config.max_trades_per_day}회 |")
    print(f"| 종목 최대 비중 | {best.config.max_position_pct:.0%} |")
    print(f"| 손절 | {best.config.stop_pct:.2%} |")
    print(f"| 목표가 | 손절폭의 {best.config.take_profit_r:.1f}배 |")
    print(f"| 거래량 배수 | {best.config.min_signal_volume_ratio:.2f}배 이상 |")
    print(f"| 1봉 거래대금 | ${best.config.min_bar_dollar_volume:,.0f} 이상 |")
    print(f"| VWAP 여유 | {best.config.vwap_buffer_pct:.2%} |")
    print(f"| 시장 확인 | {', '.join(best.config.market_symbols) if best.config.require_market_confirmation else '사용 안 함'} |")
    print()
    print(format_daytrade_summary_ko(best.full_result, title="선택 모델 전체 시뮬레이션"))
    print()
    print(f"피드백 결과 CSV:       {feedback_csv}")
    if comparison_png:
        print(f"상위 모델 비교 PNG:   {comparison_png}")
    print(f"선택 모델 총자산 CSV: {best_csv}")
    print(f"선택 모델 총자산 SVG: {best_svg}")
    if best_png:
        print(f"선택 모델 총자산 PNG: {best_png}")
    print(f"선택 모델 일별 CSV:   {daily_csv}")
    if daily_png:
        print(f"선택 모델 일별 PNG:   {daily_png}")
    print(f"선택 모델 거래 CSV:   {best_trade_csv}")
    print()
    print(format_daytrade_trade_table_ko(best.full_result["trade_log"], limit=30))
    print()
    print("실제 주문은 실행하지 않았습니다. 이 명령은 반복 피드백 백테스트 전용입니다.")
    return 0


def handle_daytrade_indicators(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    selected_symbols = symbols[: args.max_symbols] if args.max_symbols and args.max_symbols > 0 else symbols
    interval_minutes = _interval_minutes(args.interval)
    context_symbols = ["SPY", "QQQ"]
    histories = fetch_intraday_histories(
        _merge_symbols(selected_symbols, context_symbols),
        interval=args.interval,
        range_=args.range,
        cache_ttl_hours=0 if args.fresh else 1,
    )
    train_histories, holdout_histories, split_date = _split_intraday_histories(
        histories,
        selected_symbols,
        train_fraction=args.train_fraction,
    )
    configs = _daytrade_indicator_configs(args, interval_minutes)
    rows: list[dict] = []
    holdout_curves: dict[str, list] = {}

    for name, config in configs.items():
        train_result = run_daytrade_backtest(train_histories, cash=cash, config=config)
        holdout_result = run_daytrade_backtest(holdout_histories, cash=cash, config=config)
        holdout_curves[name] = holdout_result["equity_curve"]
        rows.append(
            {
                "name": name,
                "config": config,
                "train": train_result,
                "holdout": holdout_result,
                "score": _daytrade_selection_score(holdout_result),
            }
        )

    cash_train = _flat_daytrade_result(train_histories, selected_symbols, cash)
    cash_holdout = _flat_daytrade_result(holdout_histories, selected_symbols, cash)
    holdout_curves["cash_guard"] = cash_holdout["equity_curve"]
    rows.append(
        {
            "name": "cash_guard",
            "config": None,
            "train": cash_train,
            "holdout": cash_holdout,
            "score": _daytrade_selection_score(cash_holdout),
        }
    )

    rows.sort(key=lambda item: item["score"], reverse=True)
    selected = rows[0]
    full_result = (
        _flat_daytrade_result(histories, selected_symbols, cash)
        if selected["config"] is None
        else run_daytrade_backtest(histories, cash=cash, config=selected["config"])
    )
    comparison_csv = _write_daytrade_indicator_csv(rows)
    comparison_curve_csv, comparison_curve_png = write_model_comparison_artifacts(
        holdout_curves,
        label="daytrade_indicator_holdout",
    )
    full_csv_path, full_svg_path, full_png_path = write_simulation_artifacts(
        full_result["equity_curve"],
        [],
        label="selected_daytrade_indicator_equity",
    )
    daily_csv_path, daily_png_path = write_daily_return_artifacts(
        full_result["equity_curve"],
        label="selected_daytrade_indicator_daily_returns",
    )
    trade_csv_path = write_daytrade_trade_log_csv(
        full_result["trade_log"],
        label="selected_daytrade_indicator_trades",
    )

    print("## 지표 후보별 검증 결과")
    print()
    print(f"- 데이터 범위: {args.range} {args.interval}")
    print(f"- 대상 종목: {', '.join(selected_symbols)}")
    print(f"- 학습/검증 분리 기준일: {split_date.isoformat()}")
    print()
    print("| 순위 | 지표 모델 | 훈련 수익률 | 검증 최종 총자산 | 검증 수익률 | 검증 MDD | 승률 | 손익비 | 거래 |")
    print("| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for rank, item in enumerate(rows, start=1):
        holdout = item["holdout"]
        train = item["train"]
        profit_factor = holdout["profit_factor"]
        pf_text = "무한대" if profit_factor == float("inf") else f"{profit_factor:.2f}"
        print(
            f"| {rank} | {item['name']} | {train['total_return']:.2%} | "
            f"${holdout['end_equity']:,.2f} | {holdout['total_return']:.2%} | "
            f"{holdout['max_drawdown']:.2%} | {holdout['win_rate']:.2%} | "
            f"{pf_text} | {holdout['trades']:,}회 |"
        )

    selected_config = selected["config"]
    print()
    print("## 선택된 거래 모델")
    print()
    if selected_config is None:
        print("- 주 지표: 없음")
        print("- 보조 필터: 없음")
        print("- 시장 확인: 사용 안 함")
        print("- 모델명: cash_guard")
        print("- 판단: 검증 구간에서 유효한 양수 기대값 지표가 없어 거래를 보류합니다.")
    else:
        print(f"- 주 지표: {selected_config.primary_indicator}")
        print(f"- 보조 필터: {', '.join(selected_config.enabled_filters) or '없음'}")
        print(f"- 시장 확인: {', '.join(selected_config.market_symbols) if selected_config.require_market_confirmation else '사용 안 함'}")
        print(f"- 모델명: {selected_config.model_name}")
    print()
    print(format_daytrade_summary_ko(full_result, title="선택 지표 적용 전체 시뮬레이션"))
    print()
    print(f"지표 비교 CSV:        {comparison_csv}")
    print(f"검증 곡선 CSV:        {comparison_curve_csv}")
    if comparison_curve_png:
        print(f"검증 곡선 PNG:        {comparison_curve_png}")
    print(f"선택 모델 총자산 CSV: {full_csv_path}")
    print(f"선택 모델 총자산 SVG: {full_svg_path}")
    if full_png_path:
        print(f"선택 모델 총자산 PNG: {full_png_path}")
    print(f"선택 모델 일별 CSV:   {daily_csv_path}")
    if daily_png_path:
        print(f"선택 모델 일별 PNG:   {daily_png_path}")
    print(f"선택 모델 거래 CSV:   {trade_csv_path}")
    print()
    print(format_daytrade_trade_table_ko(full_result["trade_log"], limit=20))
    print()
    print("실제 주문은 실행하지 않았습니다. 선택된 지표 모델은 dry-run/백테스트 결과입니다.")
    return 0


def handle_realtime_dry_run(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    config = RealtimeConfig(
        lookback_ticks=args.lookback_ticks,
        entry_momentum_pct=_normalize_pct(args.entry_momentum),
        stop_loss_pct=_normalize_pct(args.stop_loss),
        take_profit_pct=_normalize_pct(args.take_profit),
        trailing_stop_pct=_normalize_pct(args.trailing_stop),
        max_position_pct=_normalize_pct(args.max_position_pct),
    )
    tick_path = Path(args.tick_file)
    model, decisions = run_tick_file_dry_run(
        tick_path,
        symbols=symbols,
        cash=cash,
        config=config,
        watch_seconds=args.watch_seconds,
        poll_seconds=args.poll_seconds,
    )

    print("## 10초 가격 반응 모델 dry-run")
    print()
    print(f"- 입력 파일: {tick_path}")
    print(f"- 대상 종목: {', '.join(symbols)}")
    print(f"- 남은 현금: ${model.cash:,.2f}")
    print()
    if not tick_path.exists():
        print("틱 파일이 없습니다. 토스에서 허용된 방식으로 가격을 내보낼 수 있다면 아래 JSONL 형식으로 저장해 주세요.")
        print('{"ts":"2026-05-05T09:00:00+09:00","symbol":"AAPL","price":190.12,"bid":190.10,"ask":190.14,"volume":1200,"source":"toss"}')
        return 0

    if not decisions:
        print("생성된 매수/매도 계획이 없습니다.")
        return 0

    print("| 시간 | 종목 | 행동 | 가격 | 수량 | 금액 | 사유 |")
    print("| --- | --- | --- | ---: | ---: | ---: | --- |")
    for decision in decisions:
        print(
            f"| {decision.timestamp.isoformat()} | {decision.symbol} | {decision.action} | "
            f"${decision.price:,.2f} | {decision.qty:,.4f} | ${decision.notional:,.2f} | {decision.reason} |"
        )
    print()
    print("실제 주문은 실행하지 않았습니다. 이 출력은 주문 후보 계획입니다.")
    return 0


def handle_realtime_paper(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    selected_symbols = symbols[: args.max_symbols] if args.max_symbols and args.max_symbols > 0 else symbols
    scan_symbols = symbols[: args.scan_max_symbols] if args.scan_max_symbols and args.scan_max_symbols > 0 else symbols
    until = parse_session_until(args.until)
    config = _realtime_session_config_from_args(args, selected_symbols, scan_symbols, cash, until)
    print("## 실시간 Paper Trading 시작")
    print()
    print("| 항목 | 값 |")
    print("| --- | --- |")
    print(f"| 제공자 | {args.provider} |")
    print(f"| 대상 종목 | {', '.join(selected_symbols)} |")
    print(f"| 종료 예정 | {until.isoformat()} |")
    print(f"| 주기 | {args.poll_seconds}초 |")
    print(f"| 위험 모드 | {args.risk_mode} |")
    print(f"| 동적 종목 편입 | {'사용' if args.dynamic_universe else '미사용'} |")
    print(f"| 뉴스/공시 보조지표 | {'사용' if args.research_filter else '미사용'} |")
    if args.dynamic_universe:
        print(f"| 스캔 대상 | {len(scan_symbols):,}개 |")
        print(f"| 최대 감시 종목 | {args.dynamic_max_symbols:,}개 |")
    print()
    session_dir = run_realtime_paper_session(config)
    print(summarize_realtime_session(session_dir))
    return 0


def handle_supervise_realtime(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    selected_symbols = symbols[: args.max_symbols] if args.max_symbols and args.max_symbols > 0 else symbols
    scan_symbols = symbols[: args.scan_max_symbols] if args.scan_max_symbols and args.scan_max_symbols > 0 else symbols
    until = parse_session_until(args.until)
    restarts = 0
    print("## supervise realtime paper trading")
    print(f"- risk_mode: {args.risk_mode}")
    print(f"- until: {until.isoformat()}")
    while datetime.now(until.tzinfo) < until:
        config = _realtime_session_config_from_args(args, selected_symbols, scan_symbols, cash, until)
        try:
            session_dir = run_realtime_paper_session(config)
            print(f"session finished: {session_dir}")
            if datetime.now(until.tzinfo) >= until:
                break
        except Exception as exc:
            restarts += 1
            print(f"session error; restart {restarts}/{args.max_restarts}: {exc}")
            if restarts >= args.max_restarts:
                raise
        sleep(max(args.restart_delay_seconds, 1))
    return 0


def _realtime_session_config_from_args(
    args: argparse.Namespace,
    selected_symbols: list[str],
    scan_symbols: list[str],
    cash: float,
    until: datetime,
) -> RealtimeSessionConfig:
    return RealtimeSessionConfig(
        symbols=selected_symbols,
        cash=cash,
        provider=args.provider,
        feed=args.feed,
        poll_seconds=args.poll_seconds,
        until=until,
        risk_mode=args.risk_mode,
        wait_for_credentials=args.wait_for_credentials,
        credential_check_seconds=args.credential_check_seconds,
        dynamic_universe=args.dynamic_universe,
        scan_symbols=scan_symbols,
        scan_interval_seconds=args.scan_interval_seconds,
        scan_max_symbols=args.scan_max_symbols,
        dynamic_max_symbols=args.dynamic_max_symbols,
        top_surging_symbols=args.top_surging,
        min_volume_ratio=args.min_volume_ratio,
        min_recent_dollar_volume=args.min_recent_dollar_volume,
        min_short_return_pct=_normalize_pct(args.min_short_return),
        research_filter=args.research_filter,
        research_interval_seconds=args.research_interval_seconds,
        research_max_symbols=args.research_max_symbols,
        research_news_lookback_minutes=args.research_news_lookback_minutes,
        research_block_risk_score=args.research_block_risk_score,
        research_caution_risk_score=args.research_caution_risk_score,
    )


def handle_realtime_summary(args: argparse.Namespace) -> int:
    session_dir = Path(args.session_dir) if args.session_dir else None
    print(summarize_realtime_session(session_dir))
    return 0


def handle_toss_plan(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    max_mdd = args.max_mdd / 100 if args.max_mdd and args.max_mdd > 1 else args.max_mdd
    end = date.today()
    start_fetch = end - timedelta(days=2600)
    start = date.fromisoformat(args.start)
    histories = fetch_histories(with_context_symbols(symbols), start=start_fetch, end=end)
    optimized = optimize_strategy(
        histories,
        cash=cash,
        start=start,
        max_results=1,
        max_mdd=max_mdd,
        profile=args.profile,
        tradable_symbols=tuple(symbols),
        cost_bps=5,
        slippage_bps=15,
    )
    if not optimized:
        raise DataError("No Toss checklist candidates were produced")
    best = optimized[0]
    candidates = []
    for symbol in symbols:
        bars = histories.get(symbol, [])
        if not bars:
            continue
        score = _latest_candidate_score(bars)
        if score is None:
            continue
        candidates.append((score, symbol, bars[-1].close))
    candidates.sort(reverse=True)

    print("Manual Toss Securities checklist only. No order was placed.")
    print(f"Profile: {args.profile}")
    print(f"Backtest return: {best.result['total_return']:.2%}, MDD: {best.result['max_drawdown']:.2%}, Sharpe: {best.result['sharpe']:.2f}")
    print()
    print("candidate symbol latest_price note")
    print("--------- ------ ------------ ----")
    for rank, (score, symbol, price) in enumerate(candidates[: args.limit], start=1):
        print(f"{rank:9} {symbol:6} {price:12.2f} review official filings and enter manually in Toss if you approve")
    return 0


def handle_auto_order_plan(args: argparse.Namespace, symbols: list[str], cash: float) -> int:
    max_mdd = args.max_mdd / 100 if args.max_mdd and args.max_mdd > 1 else args.max_mdd
    end = date.today()
    start_fetch = end - timedelta(days=2600)
    start = date.fromisoformat(args.start)
    histories = fetch_histories(with_context_symbols(symbols), start=start_fetch, end=end)
    optimized = optimize_strategy(
        histories,
        cash=cash,
        start=start,
        max_results=1,
        max_mdd=max_mdd,
        profile=args.profile,
        tradable_symbols=tuple(symbols),
        cost_bps=5,
        slippage_bps=15,
    )
    if not optimized:
        raise DataError("No automated order plan candidates were produced")

    best = optimized[0]
    weights = latest_target_weights(histories, best.config)
    selected = [
        (weight, symbol, histories[symbol][-1].close)
        for symbol, weight in sorted(weights.items(), key=lambda item: item[1], reverse=True)
        if symbol in symbols and histories.get(symbol) and histories[symbol][-1].close > 0
    ][: args.limit]
    if not selected:
        raise DataError("Validated model currently recommends cash / no new equity orders")

    selected_weight_total = sum(weight for weight, _, _ in selected)
    if selected_weight_total <= 0:
        raise DataError("Selected model weights are zero")

    orders = [
        ExecutionOrder(
            symbol=symbol,
            side="buy",
            qty=max((cash * args.allocation * (weight / selected_weight_total)) / price, 0),
            reason=f"{args.profile} validated target_weight={weight:.2%}",
        )
        for weight, symbol, price in selected
        if price > 0
    ]
    plan = build_execution_plan(orders, broker=args.broker, mode="dry-run")
    path = save_execution_plan(plan)
    print(format_execution_plan(plan))
    print()
    print(f"Execution plan written: {path}")
    print("No order was submitted. Use submit-plan --mode dry-run or --mode paper after reviewing the JSON.")
    return 0


def handle_submit_plan(args: argparse.Namespace) -> int:
    payload = json.loads(Path(args.plan_path).read_text(encoding="utf-8"))
    orders = [
        ExecutionOrder(
            symbol=item["symbol"],
            side=item["side"],
            qty=float(item["qty"]),
            order_type=item.get("order_type", "market"),
            time_in_force=item.get("time_in_force", "day"),
            limit_price=item.get("limit_price"),
            reason=item.get("reason", ""),
        )
        for item in payload.get("orders", [])
    ]
    if not orders:
        raise DataError("Execution plan has no orders")
    if args.mode == "live" and not args.i_understand_real_money_risk:
        raise DataError("Live mode requires --i-understand-real-money-risk")

    if args.mode == "dry-run":
        broker = DryRunBroker()
    elif args.mode == "paper":
        broker = AlpacaBroker(paper=True)
    else:
        broker = AlpacaBroker(paper=False)

    results = broker.submit_orders(orders)
    print("symbol side qty status broker_response")
    print("------ ---- --- ------ ---------------")
    for order, response in zip(orders, results):
        status = response.get("status", "submitted")
        print(f"{order.symbol:6} {order.side:4} {order.qty:.6f} {status:6} {json.dumps(response)[:180]}")
    return 0


def _latest_candidate_score(bars: list) -> float | None:
    if len(bars) < 260:
        return None
    closes = [bar.close for bar in bars]
    if closes[-1] <= 0:
        return None
    momentum_3m = closes[-1] / closes[-63] - 1
    momentum_6m = closes[-1] / closes[-126] - 1
    trend = closes[-1] / (sum(closes[-200:]) / 200) - 1
    return momentum_3m * 2 + momentum_6m + trend


def _format_research_summary_table(summary: dict) -> str:
    errors = summary.get("errors") or []
    error_text = "; ".join(errors[:2]) if errors else "-"
    rows = [
        "| 항목 | 값 |",
        "| --- | ---: |",
        f"| 자료 소스 | {summary.get('source', '-')} |",
        f"| 수집 뉴스 수 | {int(summary.get('news_items', 0)):,}건 |",
        f"| 수집 공시 수 | {int(summary.get('filings', 0)):,}건 |",
        f"| 보조지표 검사 신호 | {int(summary.get('checked_signals', 0)):,}건 |",
        f"| 진입 차단 | {int(summary.get('blocked_signals', 0)):,}건 |",
        f"| 비중 축소 주의 | {int(summary.get('caution_signals', 0)):,}건 |",
        f"| 긍정/산업 가산 | {int(summary.get('positive_signals', 0)):,}건 |",
        f"| 뉴스 룩백 | {int(summary.get('news_lookback_days', 0)):,}일 |",
        f"| 공시 룩백 | {int(summary.get('filing_lookback_days', 0)):,}일 |",
        f"| 경고 | {error_text} |",
    ]
    return "\n".join(rows)


def _format_macro_summary_table(summary: dict) -> str:
    errors = summary.get("errors") or []
    error_text = "; ".join(errors[:2]) if errors else "-"
    observations = summary.get("observations") or {}
    observation_text = ", ".join(f"{key}:{value}" for key, value in observations.items()) if observations else "-"
    rows = [
        "| item | value |",
        "| --- | ---: |",
        f"| source | {summary.get('source', '-')} |",
        f"| series | {', '.join(summary.get('series', [])) or '-'} |",
        f"| observations | {observation_text} |",
        f"| checked signals | {int(summary.get('checked_signals', 0)):,} |",
        f"| blocked signals | {int(summary.get('blocked_signals', 0)):,} |",
        f"| caution signals | {int(summary.get('caution_signals', 0)):,} |",
        f"| positive signals | {int(summary.get('positive_signals', 0)):,} |",
        f"| lookback days | {int(summary.get('lookback_days', 0)):,} |",
        f"| as-of rule | {summary.get('asof_rule', '-')} |",
        f"| warnings | {error_text} |",
    ]
    return "\n".join(rows)


def _interval_minutes(interval: str) -> int:
    if not interval.endswith("m"):
        raise ValueError("Only minute intraday intervals are supported")
    return int(interval[:-1])


def _normalize_pct(value: float) -> float:
    return value / 100 if value > 1 else value


def _daytrade_indicator_configs(args: argparse.Namespace, interval_minutes: int) -> dict[str, DayTradeConfig]:
    common = {
        "interval_minutes": interval_minutes,
        "per_trade_risk": 0.004,
        "max_position_pct": 0.15,
        "max_daily_loss_pct": 0.01,
        "max_trades_per_day": 1,
        "stop_pct": 0.006,
        "take_profit_r": 3.0,
        "cost_bps": args.cost_bps,
        "slippage_bps": args.slippage_bps,
        "last_entry_minutes_before_close": 60,
        "market_symbols": ("SPY", "QQQ"),
    }
    def config(**overrides) -> DayTradeConfig:
        values = dict(common)
        values.update(overrides)
        return DayTradeConfig(**values)

    return {
        "ma_bollinger_runner": config(
            model_name="ma_bollinger_runner",
            primary_indicator="ma_bollinger",
            enabled_filters=("volume", "bullish"),
            require_market_confirmation=True,
            opening_minutes=30,
            max_position_pct=0.40,
            max_daily_loss_pct=0.03,
            max_trades_per_day=3,
            stop_pct=0.005,
            take_profit_r=2.0,
            min_signal_volume_ratio=1.6,
            min_bar_dollar_volume=3_000_000,
            min_momentum_pct=0.0015,
            fast_ema_window=9,
            slow_ema_window=20,
            bollinger_window=20,
            bollinger_std=2.0,
            min_bollinger_position=0.85,
            vwap_buffer_pct=0.0008,
        ),
        "profit_vwap_runner": config(
            model_name="profit_vwap_runner",
            primary_indicator="vwap_trend",
            enabled_filters=("volume", "bullish", "momentum"),
            require_market_confirmation=True,
            opening_minutes=30,
            max_position_pct=0.40,
            max_daily_loss_pct=0.03,
            max_trades_per_day=3,
            stop_pct=0.005,
            take_profit_r=2.0,
            min_signal_volume_ratio=1.6,
            min_bar_dollar_volume=3_000_000,
            min_momentum_pct=0.0015,
            vwap_buffer_pct=0.0008,
        ),
        "orb_breakout": config(
            model_name="orb_breakout",
            primary_indicator="orb_breakout",
            enabled_filters=("vwap", "volume", "bullish"),
            require_market_confirmation=False,
            opening_minutes=45,
            min_signal_volume_ratio=1.8,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.0015,
        ),
        "market_orb": config(
            model_name="market_orb",
            primary_indicator="orb_breakout",
            enabled_filters=("vwap", "volume", "bullish"),
            require_market_confirmation=True,
            opening_minutes=45,
            min_signal_volume_ratio=1.8,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.0015,
        ),
        "vwap_trend": config(
            model_name="vwap_trend",
            primary_indicator="vwap_trend",
            enabled_filters=("volume", "bullish", "momentum"),
            require_market_confirmation=True,
            opening_minutes=30,
            min_signal_volume_ratio=1.6,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.001,
            min_momentum_pct=0.002,
        ),
        "volume_momentum": config(
            model_name="volume_momentum",
            primary_indicator="volume_momentum",
            enabled_filters=("vwap", "bullish"),
            require_market_confirmation=True,
            opening_minutes=30,
            min_signal_volume_ratio=2.0,
            min_bar_dollar_volume=5_000_000,
            vwap_buffer_pct=0.001,
            min_momentum_pct=0.0025,
        ),
        "pullback_vwap": config(
            model_name="pullback_vwap",
            primary_indicator="pullback_vwap",
            enabled_filters=("volume", "bullish"),
            require_market_confirmation=True,
            opening_minutes=45,
            min_signal_volume_ratio=1.7,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.001,
            take_profit_r=2.4,
        ),
        "strict_hybrid": config(
            model_name="strict_hybrid",
            primary_indicator="volume_momentum",
            enabled_filters=("orb_breakout", "vwap", "bullish"),
            require_market_confirmation=True,
            min_market_return_pct=0.0005,
            market_vwap_buffer_pct=0.0003,
            opening_minutes=45,
            min_signal_volume_ratio=2.2,
            min_bar_dollar_volume=5_000_000,
            vwap_buffer_pct=0.0015,
            breakeven_after_r=1.2,
            max_hold_minutes=180,
        ),
    }


def _split_intraday_histories(
    histories: dict[str, list],
    tradable_symbols: list[str],
    train_fraction: float,
) -> tuple[dict[str, list], dict[str, list], date]:
    dates = sorted(
        {
            bar.timestamp.astimezone(NEW_YORK).date()
            for symbol in tradable_symbols
            for bar in histories.get(symbol, [])
        }
    )
    if len(dates) < 4:
        raise DataError("Not enough intraday sessions for indicator validation")
    split_index = min(max(int(len(dates) * train_fraction), 1), len(dates) - 1)
    train_dates = set(dates[:split_index])
    holdout_dates = set(dates[split_index:])
    return (
        _filter_intraday_histories(histories, train_dates),
        _filter_intraday_histories(histories, holdout_dates),
        dates[split_index],
    )


def _filter_intraday_histories(histories: dict[str, list], allowed_dates: set[date]) -> dict[str, list]:
    return {
        symbol: [bar for bar in bars if bar.timestamp.astimezone(NEW_YORK).date() in allowed_dates]
        for symbol, bars in histories.items()
    }


def _daytrade_selection_score(result: dict) -> float:
    if result["trades"] == 0:
        return 0.0
    trade_penalty = 0.25 if result["trades"] < 5 else 0.0
    profit_factor = result["profit_factor"]
    if profit_factor == float("inf"):
        profit_factor = 3.0
    return (
        result["total_return"] * 10
        + (min(profit_factor, 3.0) - 1.0) * 0.20
        + result["win_rate"] * 0.05
        + result["max_drawdown"] * 2.0
        - trade_penalty
    )


def _flat_daytrade_result(histories: dict[str, list], tradable_symbols: list[str], cash: float) -> dict:
    dates = sorted(
        {
            bar.timestamp.astimezone(NEW_YORK).date()
            for symbol in tradable_symbols
            for bar in histories.get(symbol, [])
        }
    )
    equity_curve = [(item_date, cash) for item_date in dates]
    return {
        "start_equity": cash,
        "end_equity": cash,
        "total_return": 0.0,
        "cagr": 0.0,
        "max_drawdown": 0.0,
        "annual_volatility": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "days": len(dates),
        "trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "average_trade_return": 0.0,
        "best_trade_return": 0.0,
        "worst_trade_return": 0.0,
        "trade_log": [],
        "daily_rows": [],
        "equity_curve": equity_curve,
        "config": {"model_name": "cash_guard"},
    }


def _write_daytrade_indicator_csv(rows: list[dict], output_dir: Path = Path("reports")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"daytrade_indicator_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fieldnames = [
        "name",
        "primary_indicator",
        "enabled_filters",
        "train_return",
        "holdout_return",
        "holdout_end_equity",
        "holdout_mdd",
        "holdout_win_rate",
        "holdout_profit_factor",
        "holdout_trades",
        "score",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in rows:
            config = item["config"]
            train = item["train"]
            holdout = item["holdout"]
            writer.writerow(
                {
                    "name": item["name"],
                    "primary_indicator": config.primary_indicator if config else "",
                    "enabled_filters": ",".join(config.enabled_filters) if config else "",
                    "train_return": f"{train['total_return']:.8f}",
                    "holdout_return": f"{holdout['total_return']:.8f}",
                    "holdout_end_equity": f"{holdout['end_equity']:.2f}",
                    "holdout_mdd": f"{holdout['max_drawdown']:.8f}",
                    "holdout_win_rate": f"{holdout['win_rate']:.8f}",
                    "holdout_profit_factor": f"{holdout['profit_factor']:.8f}",
                    "holdout_trades": holdout["trades"],
                    "score": f"{item['score']:.8f}",
                }
            )
    return path


def _merge_symbols(primary: list[str], extra: list[str]) -> list[str]:
    seen: set[str] = set()
    merged: list[str] = []
    for symbol in [*primary, *extra]:
        upper = symbol.upper()
        if upper not in seen:
            seen.add(upper)
            merged.append(upper)
    return merged


def handle_alpaca_account() -> int:
    client = AlpacaPaperClient()
    account = client.account()
    print(f"Account status: {account.get('status')}")
    print(f"Buying power:   {account.get('buying_power')}")
    print(f"Portfolio value:{account.get('portfolio_value')}")
    return 0
