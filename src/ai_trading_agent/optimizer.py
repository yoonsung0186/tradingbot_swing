from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from itertools import product

from .backtest import BacktestConfig, run_backtest
from .models import Bar


@dataclass(frozen=True)
class OptimizationResult:
    score: float
    result: dict
    config: BacktestConfig


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: date
    train_end: date
    test_start: date
    test_end: date
    train_result: dict
    test_result: dict
    config: BacktestConfig


def optimize_strategy(
    histories: dict[str, list[Bar]],
    cash: float,
    start: date | None = None,
    end: date | None = None,
    max_results: int = 10,
    max_mdd: float | None = None,
    profile: str = "balanced",
    tradable_symbols: tuple[str, ...] = (),
    cost_bps: float = 5.0,
    slippage_bps: float = 10.0,
    min_dollar_volume: float = 20_000_000.0,
) -> list[OptimizationResult]:
    results: list[OptimizationResult] = []
    grid = _profile_grid(profile)
    for top_n, rebalance, long_window, min_momentum, max_volatility, stop_loss, max_weight, vix_threshold in grid:
        config = BacktestConfig(
            top_n=top_n,
            rebalance_interval=rebalance,
            long_window=long_window,
            medium_window=min(126, long_window),
            short_window=63,
            min_momentum=min_momentum,
            max_volatility=max_volatility,
            stop_loss_pct=stop_loss,
            max_weight=max_weight,
            vix_threshold=vix_threshold,
            risk_off_symbol="CASH",
            tradable_symbols=tradable_symbols,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            min_dollar_volume=min_dollar_volume,
        )
        result = run_backtest(histories, cash=cash, start=start, end=end, config=config)
        if result["days"] < 180:
            continue
        if max_mdd is not None and result["max_drawdown"] < -abs(max_mdd):
            continue
        score = objective_score(result)
        if profile == "aggressive":
            score += result["cagr"] * 1.2
        if profile == "stable":
            score += result["calmar"] * 0.5 - abs(result["max_drawdown"]) * 1.2
        results.append(OptimizationResult(score=score, result=result, config=config))

    results.sort(key=lambda item: item.score, reverse=True)
    return results[:max_results]


def walk_forward_validate(
    histories: dict[str, list[Bar]],
    cash: float,
    start: date,
    end: date,
    train_days: int = 504,
    test_days: int = 126,
    profile: str = "stable",
    max_mdd: float | None = None,
    tradable_symbols: tuple[str, ...] = (),
    cost_bps: float = 5.0,
    slippage_bps: float = 10.0,
    min_dollar_volume: float = 20_000_000.0,
) -> list[WalkForwardFold]:
    folds: list[WalkForwardFold] = []
    train_start = start
    while True:
        train_end = train_start + timedelta(days=train_days)
        test_start = train_end + timedelta(days=1)
        test_end = test_start + timedelta(days=test_days)
        if test_end > end:
            break

        optimized = optimize_strategy(
            histories,
            cash=cash,
            start=train_start,
            end=train_end,
            max_results=1,
            max_mdd=max_mdd,
            profile=profile,
            tradable_symbols=tradable_symbols,
            cost_bps=cost_bps,
            slippage_bps=slippage_bps,
            min_dollar_volume=min_dollar_volume,
        )
        if optimized:
            best = optimized[0]
            test_result = run_backtest(
                histories,
                cash=cash,
                start=test_start,
                end=test_end,
                config=best.config,
            )
            folds.append(
                WalkForwardFold(
                    train_start=train_start,
                    train_end=train_end,
                    test_start=test_start,
                    test_end=test_end,
                    train_result=best.result,
                    test_result=test_result,
                    config=best.config,
                )
            )

        train_start = train_start + timedelta(days=test_days)

    return folds


def _profile_grid(profile: str):
    if profile == "stable":
        return product(
            [3, 4],
            [10, 21],
            [150, 200],
            [0.08, 0.12],
            [0.35],
            [0.06, 0.08, 0.10],
            [0.20, 0.25],
            [30.0],
        )
    if profile == "aggressive":
        return product(
            [1, 2],
            [5, 10],
            [100, 150],
            [0.04, 0.08],
            [0.65, 0.80],
            [0.12, 0.20],
            [0.50, 0.70],
            [35.0],
        )
    return product(
        [2, 3, 4],
        [10, 21],
        [150, 200],
        [0.00, 0.04, 0.08],
        [0.35, 0.50],
        [0.08, 0.12],
        [0.25, 0.40],
        [30.0],
    )


def objective_score(result: dict) -> float:
    drawdown_penalty = max(0.0, abs(result["max_drawdown"]) - 0.18) * 3
    volatility_penalty = max(0.0, result["annual_volatility"] - 0.28) * 1.5
    return (
        result["calmar"]
        + result["sharpe"] * 0.30
        + result["cagr"] * 1.50
        - drawdown_penalty
        - volatility_penalty
    )


def format_optimization_table(results: list[OptimizationResult]) -> str:
    rows = [
        "## 후보 설정별 성과",
        "",
        "| 순위 | 점수 | 최종 총자산 | 누적 수익률 | CAGR | MDD | Sharpe | Calmar | 변동성 | 거래 | 보유종목 | 종목 최대비중 | 리밸런싱 | 장기선 | 최소 모멘텀 | 최대 변동성 | 손절 | VIX 기준 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for rank, item in enumerate(results, start=1):
        result = item.result
        config = item.config
        rows.append(
            f"| {rank} | {item.score:.2f} | ${result['end_equity']:,.2f} | "
            f"{result['total_return']:.1%} | {result['cagr']:.1%} | {result['max_drawdown']:.1%} | "
            f"{result['sharpe']:.2f} | {result['calmar']:.2f} | {result['annual_volatility']:.1%} | "
            f"{result['trades']:,}회 | {config.top_n}개 | {config.max_weight:.0%} | "
            f"{config.rebalance_interval}일 | {config.long_window}일 | {config.min_momentum:.1%} | "
            f"{config.max_volatility:.1%} | {config.stop_loss_pct:.1%} | {config.vix_threshold:.0f} |"
        )
    return "\n".join(rows)


def format_walk_forward_table(folds: list[WalkForwardFold]) -> str:
    rows = [
        "## 워크포워드 구간별 검증",
        "",
        "| 구간 | 훈련 기간 | 테스트 기간 | 훈련 수익률 | 테스트 최종 총자산 | 테스트 수익률 | 테스트 MDD | 테스트 Sharpe | 테스트 Calmar | 보유종목 | 종목 최대비중 | 리밸런싱 | 장기선 | 최소 모멘텀 | 손절 |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for idx, fold in enumerate(folds, start=1):
        result = fold.test_result
        config = fold.config
        rows.append(
            f"| {idx} | {fold.train_start:%Y-%m-%d}~{fold.train_end:%Y-%m-%d} | "
            f"{fold.test_start:%Y-%m-%d}~{fold.test_end:%Y-%m-%d} | "
            f"{fold.train_result['total_return']:.1%} | ${result['end_equity']:,.2f} | "
            f"{result['total_return']:.1%} | {result['max_drawdown']:.1%} | "
            f"{result['sharpe']:.2f} | {result['calmar']:.2f} | {config.top_n}개 | "
            f"{config.max_weight:.0%} | {config.rebalance_interval}일 | {config.long_window}일 | "
            f"{config.min_momentum:.1%} | {config.stop_loss_pct:.1%} |"
        )
    return "\n".join(rows)
