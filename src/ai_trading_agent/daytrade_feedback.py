from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

from .charting import write_model_comparison_artifacts
from .daytrade import DayTradeConfig, NEW_YORK, run_daytrade_backtest
from .models import IntradayBar
from .reporting import write_daytrade_trade_log_csv


@dataclass(frozen=True)
class FeedbackCandidate:
    name: str
    config: DayTradeConfig


@dataclass(frozen=True)
class FeedbackEvaluation:
    rank: int
    name: str
    score: float
    confidence: str
    train_result: dict
    test_result: dict
    full_result: dict
    config: DayTradeConfig


def build_feedback_candidates(interval_minutes: int, cost_bps: float, slippage_bps: float) -> list[FeedbackCandidate]:
    common = {
        "interval_minutes": interval_minutes,
        "cost_bps": cost_bps,
        "slippage_bps": slippage_bps,
        "market_symbols": ("SPY", "QQQ"),
        "max_position_pct": 0.40,
        "per_trade_risk": 0.01,
        "max_daily_loss_pct": 0.015,
        "stop_pct": 0.008,
        "take_profit_r": 2.5,
        "last_entry_minutes_before_close": 60,
    }

    def candidate(name: str, **overrides) -> FeedbackCandidate:
        values = dict(common)
        values.update(overrides)
        return FeedbackCandidate(name, DayTradeConfig(model_name=name, **values))

    return [
        candidate(
            "ma_bollinger_runner",
            primary_indicator="ma_bollinger",
            enabled_filters=("volume", "bullish"),
            opening_minutes=30,
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
            require_market_confirmation=True,
        ),
        candidate(
            "profit_vwap_runner",
            primary_indicator="vwap_trend",
            enabled_filters=("volume", "bullish", "momentum"),
            opening_minutes=30,
            max_daily_loss_pct=0.03,
            max_trades_per_day=3,
            stop_pct=0.005,
            take_profit_r=2.0,
            min_signal_volume_ratio=1.6,
            min_bar_dollar_volume=3_000_000,
            min_momentum_pct=0.0015,
            vwap_buffer_pct=0.0008,
            require_market_confirmation=True,
        ),
        candidate(
            "orb_1trade_strict",
            primary_indicator="orb_breakout",
            enabled_filters=("vwap", "volume", "bullish"),
            max_trades_per_day=1,
            min_signal_volume_ratio=1.8,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.0015,
            require_market_confirmation=True,
        ),
        candidate(
            "orb_2trade_guarded",
            primary_indicator="orb_breakout",
            enabled_filters=("vwap", "volume", "bullish"),
            max_trades_per_day=2,
            min_signal_volume_ratio=1.8,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.0015,
            require_market_confirmation=True,
        ),
        candidate(
            "orb_daily_loss_guard",
            primary_indicator="orb_breakout",
            enabled_filters=("vwap", "volume", "bullish"),
            max_trades_per_day=999,
            max_daily_loss_pct=0.01,
            min_signal_volume_ratio=1.8,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.0015,
            require_market_confirmation=True,
        ),
        candidate(
            "orb_liquid_only",
            primary_indicator="orb_breakout",
            enabled_filters=("vwap", "volume", "bullish"),
            max_trades_per_day=2,
            min_signal_volume_ratio=2.0,
            min_bar_dollar_volume=5_000_000,
            vwap_buffer_pct=0.0015,
            require_market_confirmation=True,
        ),
        candidate(
            "orb_fast_takeprofit",
            primary_indicator="orb_breakout",
            enabled_filters=("vwap", "volume", "bullish"),
            max_trades_per_day=2,
            take_profit_r=1.8,
            min_signal_volume_ratio=1.8,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.001,
            require_market_confirmation=True,
        ),
        candidate(
            "orb_wide_runner",
            primary_indicator="orb_breakout",
            enabled_filters=("vwap", "volume", "bullish", "momentum"),
            max_trades_per_day=1,
            stop_pct=0.01,
            take_profit_r=3.2,
            min_signal_volume_ratio=1.8,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.0015,
            require_market_confirmation=True,
            breakeven_after_r=1.4,
        ),
        candidate(
            "vwap_trend_guarded",
            primary_indicator="vwap_trend",
            enabled_filters=("volume", "bullish", "momentum"),
            opening_minutes=30,
            max_trades_per_day=2,
            min_signal_volume_ratio=1.6,
            min_bar_dollar_volume=3_000_000,
            min_momentum_pct=0.002,
            vwap_buffer_pct=0.001,
            require_market_confirmation=True,
        ),
        candidate(
            "vwap_trend_strict",
            primary_indicator="vwap_trend",
            enabled_filters=("volume", "bullish", "momentum"),
            opening_minutes=30,
            max_trades_per_day=1,
            min_signal_volume_ratio=1.8,
            min_bar_dollar_volume=5_000_000,
            min_momentum_pct=0.0025,
            vwap_buffer_pct=0.0015,
            require_market_confirmation=True,
        ),
        candidate(
            "volume_momentum_guarded",
            primary_indicator="volume_momentum",
            enabled_filters=("vwap", "bullish"),
            opening_minutes=30,
            max_trades_per_day=2,
            min_signal_volume_ratio=2.0,
            min_bar_dollar_volume=5_000_000,
            min_momentum_pct=0.0025,
            vwap_buffer_pct=0.001,
            require_market_confirmation=True,
        ),
        candidate(
            "volume_momentum_aggressive",
            primary_indicator="volume_momentum",
            enabled_filters=("vwap", "bullish"),
            opening_minutes=30,
            max_trades_per_day=3,
            max_daily_loss_pct=0.02,
            stop_pct=0.007,
            take_profit_r=2.2,
            min_signal_volume_ratio=1.6,
            min_bar_dollar_volume=2_000_000,
            min_momentum_pct=0.002,
            vwap_buffer_pct=0.0008,
            require_market_confirmation=True,
        ),
        candidate(
            "pullback_vwap_guarded",
            primary_indicator="pullback_vwap",
            enabled_filters=("volume", "bullish"),
            max_trades_per_day=2,
            min_signal_volume_ratio=1.7,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.001,
            take_profit_r=2.4,
            require_market_confirmation=True,
        ),
        candidate(
            "strict_hybrid",
            primary_indicator="volume_momentum",
            enabled_filters=("orb_breakout", "vwap", "bullish"),
            max_trades_per_day=1,
            min_market_return_pct=0.0005,
            market_vwap_buffer_pct=0.0003,
            min_signal_volume_ratio=2.2,
            min_bar_dollar_volume=5_000_000,
            vwap_buffer_pct=0.0015,
            breakeven_after_r=1.2,
            max_hold_minutes=180,
            require_market_confirmation=True,
        ),
        candidate(
            "late_guarded_orb",
            primary_indicator="orb_breakout",
            enabled_filters=("vwap", "volume", "bullish"),
            opening_minutes=60,
            max_trades_per_day=1,
            min_signal_volume_ratio=1.8,
            min_bar_dollar_volume=3_000_000,
            vwap_buffer_pct=0.0015,
            require_market_confirmation=True,
            last_entry_minutes_before_close=90,
        ),
    ]


def run_feedback_optimization(
    histories: dict[str, list[IntradayBar]],
    cash: float,
    interval_minutes: int,
    train_fraction: float = 0.6,
    cost_bps: float = 5.0,
    slippage_bps: float = 15.0,
) -> list[FeedbackEvaluation]:
    train_histories, test_histories = split_intraday_histories(histories, train_fraction)
    evaluations: list[FeedbackEvaluation] = []
    for candidate in build_feedback_candidates(interval_minutes, cost_bps, slippage_bps):
        train_result = run_daytrade_backtest(train_histories, cash=cash, config=candidate.config)
        test_result = run_daytrade_backtest(test_histories, cash=cash, config=candidate.config)
        full_result = run_daytrade_backtest(histories, cash=cash, config=candidate.config)
        score = score_feedback_result(train_result, test_result)
        evaluations.append(
            FeedbackEvaluation(
                rank=0,
                name=candidate.name,
                score=score,
                confidence=confidence_label(test_result),
                train_result=train_result,
                test_result=test_result,
                full_result=full_result,
                config=candidate.config,
            )
        )
    evaluations.sort(key=lambda item: (item.score, item.test_result["total_return"]), reverse=True)
    return [
        FeedbackEvaluation(
            rank=idx,
            name=item.name,
            score=item.score,
            confidence=item.confidence,
            train_result=item.train_result,
            test_result=item.test_result,
            full_result=item.full_result,
            config=item.config,
        )
        for idx, item in enumerate(evaluations, start=1)
    ]


def select_best_active_model(evaluations: Iterable[FeedbackEvaluation]) -> FeedbackEvaluation:
    active = [item for item in evaluations if item.test_result["trades"] > 0]
    if not active:
        raise ValueError("No active feedback candidate produced trades")
    return max(active, key=lambda item: (item.score, item.test_result["total_return"]))


def score_feedback_result(train_result: dict, test_result: dict) -> float:
    test_return = float(test_result["total_return"])
    test_mdd = abs(float(test_result["max_drawdown"]))
    train_return = float(train_result["total_return"])
    train_mdd = abs(float(train_result["max_drawdown"]))
    profit_factor = float(test_result.get("profit_factor") or 0.0)
    if profit_factor == float("inf"):
        profit_factor = 3.0
    trades = int(test_result.get("trades", 0))
    if trades <= 0:
        return -999.0
    overfit_penalty = max(train_return - test_return, 0.0) * 2.0
    trade_penalty = max(trades - 12, 0) * 0.001
    return (
        test_return * 100
        - test_mdd * 55
        - train_mdd * 10
        - overfit_penalty * 25
        - trade_penalty
        + min(profit_factor, 3.0) * 0.12
    )


def confidence_label(result: dict) -> str:
    total_return = float(result["total_return"])
    trades = int(result.get("trades", 0))
    profit_factor = float(result.get("profit_factor") or 0.0)
    if trades <= 0:
        return "거래 없음"
    if total_return > 0 and profit_factor >= 1.1 and trades >= 5:
        return "보통"
    if total_return > 0:
        return "낮음"
    return "낮음/방어 필요"


def split_intraday_histories(
    histories: dict[str, list[IntradayBar]],
    train_fraction: float,
) -> tuple[dict[str, list[IntradayBar]], dict[str, list[IntradayBar]]]:
    dates = sorted(
        {
            bar.timestamp.astimezone(NEW_YORK).date()
            for bars in histories.values()
            for bar in bars
        }
    )
    if len(dates) < 2:
        return histories, histories
    split_idx = min(max(int(len(dates) * train_fraction), 1), len(dates) - 1)
    split_date = dates[split_idx]
    train: dict[str, list[IntradayBar]] = {}
    test: dict[str, list[IntradayBar]] = {}
    for symbol, bars in histories.items():
        train[symbol] = [bar for bar in bars if bar.timestamp.astimezone(NEW_YORK).date() < split_date]
        test[symbol] = [bar for bar in bars if bar.timestamp.astimezone(NEW_YORK).date() >= split_date]
    return train, test


def write_feedback_artifacts(
    evaluations: list[FeedbackEvaluation],
    output_dir: Path = Path("reports"),
) -> tuple[Path, Path | None, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = output_dir / f"daytrade_feedback_results_{timestamp}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "rank",
                "model",
                "score",
                "confidence",
                "train_return",
                "test_return",
                "test_mdd",
                "test_trades",
                "test_win_rate",
                "test_profit_factor",
                "full_return",
                "full_mdd",
                "full_trades",
                "max_trades_per_day",
                "max_position_pct",
                "volume_ratio",
                "min_bar_dollar_volume",
                "vwap_buffer_pct",
                "primary_indicator",
                "filters",
            ]
        )
        for item in evaluations:
            writer.writerow(
                [
                    item.rank,
                    item.name,
                    f"{item.score:.6f}",
                    item.confidence,
                    f"{float(item.train_result['total_return']):.8f}",
                    f"{float(item.test_result['total_return']):.8f}",
                    f"{float(item.test_result['max_drawdown']):.8f}",
                    int(item.test_result["trades"]),
                    f"{float(item.test_result.get('win_rate', 0.0)):.8f}",
                    f"{float(item.test_result.get('profit_factor', 0.0)):.8f}",
                    f"{float(item.full_result['total_return']):.8f}",
                    f"{float(item.full_result['max_drawdown']):.8f}",
                    int(item.full_result["trades"]),
                    item.config.max_trades_per_day,
                    f"{item.config.max_position_pct:.4f}",
                    f"{item.config.min_signal_volume_ratio:.4f}",
                    f"{item.config.min_bar_dollar_volume:.2f}",
                    f"{item.config.vwap_buffer_pct:.6f}",
                    item.config.primary_indicator,
                    "+".join(item.config.enabled_filters),
                ]
            )

    top_curves = {item.name: item.full_result["equity_curve"] for item in evaluations[:5]}
    _, comparison_png = write_model_comparison_artifacts(
        top_curves,
        output_dir=output_dir,
        label="daytrade_feedback_top_models",
    )
    best = select_best_active_model(evaluations)
    trade_path = write_daytrade_trade_log_csv(
        best.full_result["trade_log"],
        output_dir=output_dir,
        label="daytrade_feedback_best_trades",
    )
    return csv_path, comparison_png, trade_path


def format_feedback_table_ko(evaluations: list[FeedbackEvaluation], limit: int = 15) -> str:
    rows = [
        "## 피드백 최적화 결과",
        "",
        "| 순위 | 모델 | 점수 | 신뢰도 | 검증 수익률 | 검증 MDD | 검증 거래 | 검증 승률 | 손익비 | 전체 수익률 |",
        "| ---: | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in evaluations[:limit]:
        profit_factor = float(item.test_result.get("profit_factor", 0.0))
        profit_factor_text = "무한" if profit_factor == float("inf") else f"{profit_factor:.2f}"
        rows.append(
            f"| {item.rank} | {item.name} | {item.score:.2f} | {item.confidence} | "
            f"{float(item.test_result['total_return']):.2%} | "
            f"{float(item.test_result['max_drawdown']):.2%} | "
            f"{int(item.test_result['trades']):,}회 | "
            f"{float(item.test_result.get('win_rate', 0.0)):.2%} | "
            f"{profit_factor_text} | "
            f"{float(item.full_result['total_return']):.2%} |"
        )
    return "\n".join(rows)


def _curve_dates(curve: list[tuple[date, float]]) -> tuple[date | None, date | None]:
    if not curve:
        return None, None
    return curve[0][0], curve[-1][0]
