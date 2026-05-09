from __future__ import annotations

import sys
import unittest
from datetime import date, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ai_trading_agent.alpaca import AlpacaNewsItem
from ai_trading_agent.daytrade import DayTradeConfig, run_daytrade_backtest
from ai_trading_agent.models import Bar, IntradayBar, PortfolioSnapshot
from ai_trading_agent.official_research import Filing, OfficialResearch
from ai_trading_agent.realtime import PriceTick, RealtimeConfig, RealtimeReactiveModel
from ai_trading_agent.realtime_session import (
    SurgeScanConfig,
    classify_research_signals,
    realtime_config_for_mode,
    score_surging_symbols,
)
from ai_trading_agent.risk import RiskManager
from ai_trading_agent.strategy import MomentumStrategy
from ai_trading_agent.backtest import BacktestConfig, latest_target_weights, run_backtest
from ai_trading_agent.macro import FredObservation, _latest_realtime_value
from ai_trading_agent.metrics import performance_stats
from ai_trading_agent.swing import (
    SwingConfig,
    _macd,
    leveraged_overlay_improved_config,
    leveraged_overlay_regime_4stage_config,
    run_swing_backtest,
)


def make_bars(symbol: str, days: int = 70, step: float = 0.8) -> list[Bar]:
    start = date(2025, 1, 1)
    bars: list[Bar] = []
    for idx in range(days):
        close = 100 + idx * step
        volume = 1_000_000
        if idx == days - 1:
            volume = 2_000_000
        bars.append(
            Bar(
                symbol=symbol,
                date=start + timedelta(days=idx),
                open=close - 0.5,
                high=close + 1,
                low=close - 1,
                close=close,
                volume=volume,
            )
        )
    return bars


def make_intraday_bars(symbol: str) -> list[IntradayBar]:
    ny = ZoneInfo("America/New_York")
    start = datetime(2025, 1, 2, 9, 30, tzinfo=ny)
    prices = [
        (100.00, 100.20, 99.80, 100.00, 1_000),
        (100.00, 100.30, 99.90, 100.10, 1_000),
        (100.10, 100.35, 99.95, 100.15, 1_000),
        (100.15, 100.40, 100.00, 100.20, 1_000),
        (100.20, 100.35, 100.00, 100.25, 1_000),
        (100.25, 100.30, 100.05, 100.10, 1_000),
        (100.20, 100.90, 100.10, 100.80, 5_000),
        (100.90, 102.00, 100.70, 101.60, 2_000),
        (101.60, 101.70, 101.10, 101.20, 1_000),
    ]
    bars: list[IntradayBar] = []
    for idx, (open_price, high, low, close, volume) in enumerate(prices):
        bars.append(
            IntradayBar(
                symbol=symbol,
                timestamp=(start + timedelta(minutes=idx * 5)).astimezone(ZoneInfo("UTC")),
                open=open_price,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
    return bars


class StrategyAndRiskTests(unittest.TestCase):
    def test_momentum_strategy_generates_buy(self) -> None:
        strategy = MomentumStrategy()
        signals = strategy.generate({"SPY": make_bars("SPY")}, market_risk_on=True)
        self.assertEqual(signals[0].action, "BUY")

    def test_risk_manager_sizes_order(self) -> None:
        strategy = MomentumStrategy()
        signal = strategy.generate({"SPY": make_bars("SPY")}, market_risk_on=True)[0]
        snapshot = PortfolioSnapshot(cash=10_000, equity=10_000, positions={}, prices={"SPY": signal.price})
        decision = RiskManager().review_buy(signal, snapshot)
        self.assertTrue(decision.allowed)
        self.assertIsNotNone(decision.order)
        self.assertGreater(decision.order.qty, 0)

    def test_backtest_returns_risk_metrics(self) -> None:
        histories = {
            "SPY": make_bars("SPY", days=320),
            "SHY": make_bars("SHY", days=320),
            "^VIX": make_bars("^VIX", days=320),
        }
        config = BacktestConfig(long_window=100, medium_window=80, short_window=40)
        result = run_backtest(histories, cash=10_000, config=config)
        self.assertGreater(result["end_equity"], 0)
        self.assertIn("sharpe", result)
        self.assertIn("equity_curve", result)

    def test_backtest_delays_signal_execution_to_next_day(self) -> None:
        bars = make_bars("AAPL", days=80)
        config = BacktestConfig(
            short_window=5,
            medium_window=10,
            long_window=20,
            min_history_days=20,
            rebalance_interval=30,
            min_dollar_volume=0,
            max_volatility=2,
            use_regime_filter=False,
            tradable_symbols=("AAPL",),
        )
        result = run_backtest(
            {"AAPL": bars},
            cash=10_000,
            start=bars[30].date,
            end=bars[32].date,
            config=config,
        )
        self.assertEqual(result["equity_curve"][0][1], 10_000)
        self.assertGreater(result["equity_curve"][-1][1], 10_000)

    def test_swing_backtest_delays_entry_to_next_day(self) -> None:
        bars = make_bars("AAPL", days=120, step=0.6)
        config = SwingConfig(
            model_name="swing_test",
            ema_fast_window=5,
            ema_slow_window=10,
            exit_ema_window=5,
            bollinger_window=10,
            atr_window=5,
            rsi_window=5,
            momentum_window=5,
            relative_strength_window=5,
            breakout_window=5,
            volume_window=5,
            min_volume_ratio=0.5,
            min_dollar_volume=0,
            min_momentum=-1,
            min_relative_strength=-1,
            max_rsi=100,
            max_position_pct=0.5,
            max_positions=1,
            use_market_filter=False,
            cost_bps=0,
            slippage_bps=0,
            tradable_symbols=("AAPL",),
        )
        result = run_swing_backtest(
            {"AAPL": bars},
            cash=10_000,
            start=bars[30].date,
            end=bars[40].date,
            config=config,
        )
        self.assertGreaterEqual(result["transaction_count"], 1)
        self.assertEqual(result["trade_log"][0]["side"], "BUY")
        self.assertEqual(result["trade_log"][0]["date"], bars[31].date.isoformat())

    def test_swing_research_uses_signal_date_not_entry_date(self) -> None:
        bars = make_bars("AAPL", days=120, step=0.6)
        cutoff = bars[31].date

        class FakeResearchProvider:
            def score(self, symbol: str, as_of: date):
                return SimpleNamespace(
                    allow_entry=as_of < cutoff,
                    risk_level="ok" if as_of < cutoff else "blocked",
                    risk_score=0,
                    positive_score=0,
                    industry_score=0.0,
                    score_adjustment=0.0,
                    position_multiplier=1.0,
                    reasons=("test research",),
                )

            def summary(self) -> dict:
                return {"source": "test"}

        config = SwingConfig(
            model_name="swing_research_test",
            ema_fast_window=5,
            ema_slow_window=10,
            exit_ema_window=5,
            bollinger_window=10,
            atr_window=5,
            rsi_window=5,
            momentum_window=5,
            relative_strength_window=5,
            breakout_window=5,
            volume_window=5,
            min_volume_ratio=0.5,
            min_dollar_volume=0,
            min_momentum=-1,
            min_relative_strength=-1,
            max_rsi=100,
            max_position_pct=0.5,
            max_positions=1,
            use_market_filter=False,
            cost_bps=0,
            slippage_bps=0,
            tradable_symbols=("AAPL",),
        )
        result = run_swing_backtest(
            {"AAPL": bars},
            cash=10_000,
            start=bars[30].date,
            end=bars[40].date,
            config=config,
            research_provider=FakeResearchProvider(),
        )
        self.assertGreaterEqual(result["transaction_count"], 1)
        self.assertEqual(result["trade_log"][0]["date"], cutoff.isoformat())

    def test_fred_macro_uses_realtime_start_not_future_observation(self) -> None:
        observations = [
            FredObservation(
                series_id="DGS10",
                observation_date=date(2025, 1, 9),
                realtime_start=date(2025, 1, 10),
                realtime_end=date(2025, 1, 31),
                value=3.0,
            ),
            FredObservation(
                series_id="DGS10",
                observation_date=date(2025, 1, 10),
                realtime_start=date(2025, 1, 13),
                realtime_end=date(2025, 1, 31),
                value=5.0,
            ),
        ]
        as_of_value = _latest_realtime_value(observations, date(2025, 1, 10))
        self.assertIsNotNone(as_of_value)
        self.assertEqual(as_of_value.value, 3.0)

    def test_macd_histogram_detects_rising_momentum(self) -> None:
        prices = [100 + idx * 0.1 for idx in range(30)] + [103 + idx * 0.8 for idx in range(20)]
        state = _macd(prices, 12, 26, 9)
        self.assertIsNotNone(state)
        self.assertGreater(state["macd"], state["signal"])
        self.assertGreater(state["histogram"], 0)

    def test_improved_model_combines_only_valid_experiment_controls(self) -> None:
        config = leveraged_overlay_improved_config()
        self.assertTrue(config.use_strong_market_sizing)
        self.assertTrue(config.allow_pyramiding)
        self.assertEqual(config.pyramid_trigger_pct, 0.15)
        self.assertEqual(config.pyramid_add_fraction, 0.35)
        self.assertEqual(config.time_exit_mode, "evidence_extend")
        self.assertEqual(config.min_extension_unrealized_return, 0.08)
        self.assertEqual(config.min_extension_relative_strength, 0.03)
        self.assertEqual(config.max_extended_hold_days, 60)
        self.assertEqual(config.leveraged_max_extended_hold_days, 45)
        self.assertEqual(config.event_positive_score_bonus, 0.0)
        self.assertEqual(config.leveraged_underlying_min_short_momentum, -1.0)
        self.assertEqual(config.max_positions, 2)

    def test_regime_4stage_model_keeps_improved_controls(self) -> None:
        config = leveraged_overlay_regime_4stage_config()
        self.assertTrue(config.use_four_stage_regime)
        self.assertFalse(config.use_strong_market_sizing)
        self.assertEqual(config.time_exit_mode, "evidence_extend")
        self.assertTrue(config.allow_pyramiding)
        self.assertEqual(config.regime_neutral_position_pct, 0.40)
        self.assertEqual(config.regime_bull_position_pct, 0.45)
        self.assertEqual(config.regime_strong_bull_position_pct, 0.55)
        self.assertEqual(config.regime_neutral_risk_multiplier, 0.85)
        self.assertEqual(config.regime_strong_bull_risk_multiplier, 1.12)
        self.assertEqual(config.regime_risk_off_vix_jump, 99.0)
        self.assertEqual(config.regime_leveraged_entry_min, "")
        self.assertEqual(config.regime_pyramid_entry_min, "")
        self.assertEqual(config.regime_risk_off_trailing_stop_pct, 0.0)
        self.assertTrue(config.regime_use_ema200_filter)
        self.assertTrue(config.leveraged_allow_pyramiding)
        self.assertEqual(config.leveraged_fast_exit_days, 0)
        self.assertEqual(config.leveraged_allowed_symbols, ())
        self.assertEqual(config.leveraged_blocked_symbols, ("BULZ",))
        self.assertEqual(config.leveraged_strict_symbols, ())
        self.assertEqual(config.early_weak_exit_days, 0)
        self.assertEqual(config.second_pyramid_trigger_pct, 0.0)
        self.assertEqual(config.second_pyramid_max_position_pct, 0.0)
        self.assertFalse(config.second_pyramid_leveraged_allowed)
        self.assertFalse(config.second_pyramid_after_partial_allowed)

    def test_latest_target_weights_uses_strategy_filters(self) -> None:
        histories = {
            "AAPL": make_bars("AAPL", days=320, step=1.2),
            "MSFT": make_bars("MSFT", days=320, step=0.3),
        }
        config = BacktestConfig(
            short_window=20,
            medium_window=60,
            long_window=100,
            min_history_days=100,
            top_n=1,
            min_dollar_volume=0,
            max_volatility=2,
            use_regime_filter=False,
            tradable_symbols=("AAPL", "MSFT"),
        )
        weights = latest_target_weights(histories, config)
        self.assertEqual(list(weights), ["AAPL"])
        self.assertGreater(weights["AAPL"], 0)
        self.assertLessEqual(sum(weights.values()), 1)

    def test_daytrade_enters_on_next_bar_and_exits_same_day(self) -> None:
        config = DayTradeConfig(
            model_name="orb_breakout_test",
            primary_indicator="orb_breakout",
            enabled_filters=("vwap", "volume", "bullish"),
            opening_minutes=30,
            interval_minutes=5,
            cost_bps=0,
            slippage_bps=0,
            min_bar_dollar_volume=0,
            min_signal_volume_ratio=1.2,
            stop_pct=0.005,
            take_profit_r=1.0,
            max_trades_per_day=1,
            require_market_confirmation=False,
        )
        result = run_daytrade_backtest({"AAPL": make_intraday_bars("AAPL")}, cash=10_000, config=config)
        self.assertEqual(result["trades"], 1)
        trade = result["trade_log"][0]
        self.assertEqual(trade["entry_time"], "01-03 00:05 KST")
        self.assertEqual(trade["exit_time"], "01-03 00:05 KST")
        self.assertEqual(trade["reason"], "take_profit")
        self.assertGreater(result["end_equity"], result["start_equity"])

    def test_realtime_model_creates_dry_run_plan_from_10s_momentum(self) -> None:
        model = RealtimeReactiveModel(
            cash=10_000,
            config=RealtimeConfig(lookback_ticks=3, entry_momentum_pct=0.003, cooldown_ticks=1),
        )
        start = datetime(2025, 1, 2, 14, 30, tzinfo=ZoneInfo("UTC"))
        decisions = []
        for idx, price in enumerate([100.0, 100.2, 100.5]):
            decisions.extend(
                model.on_tick(
                    PriceTick(
                        timestamp=start + timedelta(seconds=idx * 10),
                        symbol="AAPL",
                        price=price,
                        bid=price - 0.01,
                        ask=price + 0.01,
                        source="test",
                    )
                )
            )
        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0].action, "BUY_PLAN")
        self.assertEqual(decisions[0].symbol, "AAPL")

    def test_realtime_runner_partially_exits_then_trails_remainder(self) -> None:
        model = RealtimeReactiveModel(
            cash=10_000,
            config=RealtimeConfig(
                lookback_ticks=2,
                entry_momentum_pct=0.001,
                take_profit_pct=0.05,
                stop_loss_pct=0.01,
                trailing_stop_pct=0.009,
                cooldown_ticks=1,
                partial_take_profit_pct=0.012,
                partial_take_profit_fraction=0.4,
                breakeven_after_pct=0.015,
                breakeven_offset_pct=0.001,
            ),
        )
        start = datetime(2025, 1, 2, 14, 30, tzinfo=ZoneInfo("UTC"))
        decisions = []
        for idx, price in enumerate([100.0, 100.2, 101.5, 103.0, 102.0]):
            decisions.extend(
                model.on_tick(
                    PriceTick(
                        timestamp=start + timedelta(seconds=idx * 10),
                        symbol="AAPL",
                        price=price,
                        bid=price - 0.01,
                        ask=price + 0.01,
                        source="test",
                    )
                )
            )
        actions = [decision.action for decision in decisions]
        self.assertEqual(actions, ["BUY_PLAN", "SELL_PARTIAL_PLAN", "SELL_PLAN"])
        self.assertEqual(decisions[1].reason, "partial take profit")
        self.assertEqual(decisions[2].reason, "trailing stop")
        self.assertNotIn("AAPL", model.positions)

    def test_surge_scanner_scores_volume_and_dollar_volume_breakout(self) -> None:
        ny = ZoneInfo("America/New_York")
        start = datetime(2025, 1, 2, 9, 30, tzinfo=ny)
        quiet_bars = []
        surge_bars = []
        for idx in range(55):
            quiet_bars.append(
                IntradayBar(
                    symbol="QUIET",
                    timestamp=(start + timedelta(minutes=idx)).astimezone(ZoneInfo("UTC")),
                    open=100,
                    high=100.1,
                    low=99.9,
                    close=100,
                    volume=1_000,
                )
            )
            volume = 1_000 if idx < 50 else 80_000
            close = 100 + max(idx - 49, 0) * 0.25
            surge_bars.append(
                IntradayBar(
                    symbol="SURG",
                    timestamp=(start + timedelta(minutes=idx)).astimezone(ZoneInfo("UTC")),
                    open=close - 0.05,
                    high=close + 0.1,
                    low=close - 0.1,
                    close=close,
                    volume=volume,
                )
            )
        candidates = score_surging_symbols(
            {"QUIET": quiet_bars, "SURG": surge_bars},
            SurgeScanConfig(min_volume_ratio=2.0, min_recent_dollar_volume=1_000_000, min_short_return_pct=0.0001),
        )
        self.assertEqual([item.symbol for item in candidates], ["SURG"])
        self.assertGreater(candidates[0].volume_ratio, 2.0)

    def test_research_filter_blocks_high_risk_official_event(self) -> None:
        now = datetime(2025, 1, 3, 14, 30, tzinfo=ZoneInfo("UTC"))
        news = [
            AlpacaNewsItem(
                id="1",
                headline="ACME receives SEC investigation notice after accounting probe",
                summary="Company says it will cooperate.",
                url="https://example.com/news",
                source="TrustedWire",
                created_at=now,
                updated_at=now,
                symbols=("ACME",),
            )
        ]
        official = OfficialResearch(
            symbol="ACME",
            cik="0000000001",
            company_name="ACME Corp",
            latest_filings=[
                Filing(
                    symbol="ACME",
                    form="8-K",
                    filed="2025-01-03",
                    accession="0000000001-25-000001",
                    document="form8k.htm",
                    url="https://www.sec.gov/Archives/example",
                )
            ],
            facts={},
            official_queries=[],
        )
        snapshot = classify_research_signals("ACME", news, official, now)
        self.assertFalse(snapshot.allow_entry)
        self.assertEqual(snapshot.risk_level, "blocked")
        self.assertGreaterEqual(snapshot.risk_score, 4)

    def test_research_filter_treats_positive_news_as_auxiliary_only(self) -> None:
        now = datetime(2025, 1, 3, 14, 30, tzinfo=ZoneInfo("UTC"))
        news = [
            AlpacaNewsItem(
                id="2",
                headline="ACME raises guidance after record revenue",
                summary="Management also announces a buyback.",
                url="https://example.com/news",
                source="TrustedWire",
                created_at=now,
                updated_at=now,
                symbols=("ACME",),
            )
        ]
        snapshot = classify_research_signals("ACME", news, None, now)
        self.assertTrue(snapshot.allow_entry)
        self.assertEqual(snapshot.risk_level, "ok")
        self.assertGreater(snapshot.positive_score, 0)
        self.assertEqual(snapshot.risk_score, 0)

    def test_hybrid_runner_sits_between_stable_and_surge_risk(self) -> None:
        stable = realtime_config_for_mode("stable")
        hybrid = realtime_config_for_mode("hybrid_runner")
        surge = realtime_config_for_mode("surge_runner")
        self.assertEqual(stable.max_position_pct, 0.40)
        self.assertEqual(hybrid.max_position_pct, 0.40)
        self.assertEqual(surge.max_position_pct, 0.40)
        self.assertGreater(hybrid.take_profit_pct, stable.take_profit_pct)
        self.assertLess(hybrid.take_profit_pct, surge.take_profit_pct)
        self.assertIsNotNone(hybrid.partial_take_profit_pct)
        self.assertEqual(hybrid.max_tick_age_seconds, 75)

    def test_stale_tick_guard_blocks_hybrid_entry(self) -> None:
        model = RealtimeReactiveModel(
            cash=10_000,
            config=RealtimeConfig(
                lookback_ticks=2,
                entry_momentum_pct=0.001,
                max_tick_age_seconds=1,
                min_tick_volume=2,
            ),
        )
        old = datetime(2025, 1, 2, 14, 30, tzinfo=ZoneInfo("UTC"))
        decisions = []
        for idx, price in enumerate([100.0, 100.4]):
            decisions.extend(
                model.on_tick(
                    PriceTick(
                        timestamp=old + timedelta(seconds=idx),
                        symbol="AAPL",
                        price=price,
                        bid=price - 0.01,
                        ask=price + 0.01,
                        volume=10,
                        source="test",
                    )
                )
            )
        self.assertEqual(decisions, [])

    def test_sec_filing_text_can_block_entry(self) -> None:
        now = datetime(2025, 1, 3, 14, 30, tzinfo=ZoneInfo("UTC"))
        official = OfficialResearch(
            symbol="ACME",
            cik="0000000001",
            company_name="ACME Corp",
            latest_filings=[],
            facts={},
            official_queries=[],
        )
        snapshot = classify_research_signals(
            "ACME",
            [],
            official,
            now,
            official_texts=["The company discloses an SEC investigation and accounting probe."],
        )
        self.assertFalse(snapshot.allow_entry)
        self.assertEqual(snapshot.risk_level, "blocked")

    def test_performance_stats_handles_curve(self) -> None:
        curve = [
            (date(2025, 1, 1), 10_000),
            (date(2025, 1, 2), 10_100),
            (date(2025, 1, 3), 9_900),
            (date(2025, 1, 4), 10_300),
        ]
        stats = performance_stats(curve)
        self.assertGreater(stats.end_equity, stats.start_equity)
        self.assertLess(stats.max_drawdown, 0)


if __name__ == "__main__":
    unittest.main()
