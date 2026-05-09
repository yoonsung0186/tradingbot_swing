from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date
from statistics import mean, pstdev


TRADING_DAYS = 252


@dataclass(frozen=True)
class PerformanceStats:
    start_equity: float
    end_equity: float
    total_return: float
    cagr: float
    max_drawdown: float
    annual_volatility: float
    sharpe: float
    sortino: float
    calmar: float
    days: int


def pct_change(values: list[float]) -> list[float]:
    returns: list[float] = []
    for previous, current in zip(values, values[1:]):
        if previous:
            returns.append(current / previous - 1)
    return returns


def annualized_volatility(returns: list[float]) -> float:
    if len(returns) < 2:
        return 0.0
    return pstdev(returns) * math.sqrt(TRADING_DAYS)


def downside_volatility(returns: list[float]) -> float:
    downside = [item for item in returns if item < 0]
    if len(downside) < 2:
        return 0.0
    return pstdev(downside) * math.sqrt(TRADING_DAYS)


def max_drawdown(values: list[float]) -> float:
    if not values:
        return 0.0
    peak = values[0]
    worst = 0.0
    for value in values:
        peak = max(peak, value)
        if peak:
            worst = min(worst, value / peak - 1)
    return worst


def compound_annual_growth_rate(start_value: float, end_value: float, days: int) -> float:
    if start_value <= 0 or end_value <= 0 or days <= 0:
        return 0.0
    years = days / TRADING_DAYS
    if years <= 0:
        return 0.0
    return (end_value / start_value) ** (1 / years) - 1


def correlation(left: list[float], right: list[float]) -> float:
    size = min(len(left), len(right))
    if size < 3:
        return 0.0
    left = left[-size:]
    right = right[-size:]
    left_mean = mean(left)
    right_mean = mean(right)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left, right))
    left_denominator = math.sqrt(sum((a - left_mean) ** 2 for a in left))
    right_denominator = math.sqrt(sum((b - right_mean) ** 2 for b in right))
    if not left_denominator or not right_denominator:
        return 0.0
    return numerator / (left_denominator * right_denominator)


def beta(asset_returns: list[float], benchmark_returns: list[float]) -> float:
    size = min(len(asset_returns), len(benchmark_returns))
    if size < 3:
        return 0.0
    asset_returns = asset_returns[-size:]
    benchmark_returns = benchmark_returns[-size:]
    benchmark_mean = mean(benchmark_returns)
    asset_mean = mean(asset_returns)
    variance = sum((item - benchmark_mean) ** 2 for item in benchmark_returns)
    if not variance:
        return 0.0
    covariance = sum(
        (asset - asset_mean) * (benchmark - benchmark_mean)
        for asset, benchmark in zip(asset_returns, benchmark_returns)
    )
    return covariance / variance


def performance_stats(equity_curve: list[tuple[date, float]]) -> PerformanceStats:
    if not equity_curve:
        return PerformanceStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)

    values = [value for _, value in equity_curve]
    returns = pct_change(values)
    start_value = values[0]
    end_value = values[-1]
    total_return = end_value / start_value - 1 if start_value else 0.0
    cagr = compound_annual_growth_rate(start_value, end_value, len(values))
    annual_vol = annualized_volatility(returns)
    downside_vol = downside_volatility(returns)
    daily_mean = mean(returns) if returns else 0.0
    daily_std = pstdev(returns) if len(returns) > 1 else 0.0
    daily_downside = pstdev([item for item in returns if item < 0]) if len([item for item in returns if item < 0]) > 1 else 0.0
    sharpe = (daily_mean / daily_std * math.sqrt(TRADING_DAYS)) if daily_std else 0.0
    sortino = (daily_mean / daily_downside * math.sqrt(TRADING_DAYS)) if daily_downside else 0.0
    drawdown = max_drawdown(values)
    calmar = cagr / abs(drawdown) if drawdown else 0.0
    if not downside_vol:
        sortino = 0.0

    return PerformanceStats(
        start_equity=start_value,
        end_equity=end_value,
        total_return=total_return,
        cagr=cagr,
        max_drawdown=drawdown,
        annual_volatility=annual_vol,
        sharpe=sharpe,
        sortino=sortino,
        calmar=calmar,
        days=len(values),
    )
