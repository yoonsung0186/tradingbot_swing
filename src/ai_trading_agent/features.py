from __future__ import annotations

from dataclasses import dataclass
from statistics import mean

from .metrics import annualized_volatility, beta, correlation, max_drawdown, pct_change
from .models import Bar


@dataclass(frozen=True)
class AssetFeature:
    symbol: str
    price: float
    momentum_1m: float
    momentum_3m: float
    momentum_6m: float
    momentum_12m: float
    volatility_3m: float
    drawdown_12m: float
    volume_ratio: float
    correlation_spy: float
    beta_spy: float
    quality_score: float


def compute_features(histories: dict[str, list[Bar]], benchmark_symbol: str = "SPY") -> list[AssetFeature]:
    benchmark = histories.get(benchmark_symbol, [])
    benchmark_returns = pct_change([bar.close for bar in benchmark])
    features: list[AssetFeature] = []
    for symbol, bars in histories.items():
        if symbol.startswith("^") or len(bars) < 260:
            continue
        closes = [bar.close for bar in bars]
        returns = pct_change(closes)
        feature = AssetFeature(
            symbol=symbol,
            price=closes[-1],
            momentum_1m=_momentum(closes, 21),
            momentum_3m=_momentum(closes, 63),
            momentum_6m=_momentum(closes, 126),
            momentum_12m=_momentum(closes, 252),
            volatility_3m=annualized_volatility(returns[-63:]),
            drawdown_12m=max_drawdown(closes[-252:]),
            volume_ratio=_volume_ratio(bars),
            correlation_spy=correlation(returns[-252:], benchmark_returns[-252:]),
            beta_spy=beta(returns[-252:], benchmark_returns[-252:]),
            quality_score=0.0,
        )
        features.append(_with_score(feature))
    features.sort(key=lambda item: item.quality_score, reverse=True)
    return features


def format_features_table(features: list[AssetFeature], limit: int = 20) -> str:
    rows = [
        "symbol price score 1m 3m 6m 12m vol dd corr beta vol_ratio",
        "------ ----- ----- -- -- -- --- --- -- ---- ---- ---------",
    ]
    for item in features[:limit]:
        rows.append(
            f"{item.symbol:6} {item.price:7.2f} {item.quality_score:5.2f} "
            f"{item.momentum_1m:6.1%} {item.momentum_3m:6.1%} "
            f"{item.momentum_6m:6.1%} {item.momentum_12m:7.1%} "
            f"{item.volatility_3m:6.1%} {item.drawdown_12m:6.1%} "
            f"{item.correlation_spy:5.2f} {item.beta_spy:5.2f} {item.volume_ratio:9.2f}"
        )
    return "\n".join(rows)


def _momentum(closes: list[float], days: int) -> float:
    if len(closes) <= days or closes[-days] == 0:
        return 0.0
    return closes[-1] / closes[-days] - 1


def _volume_ratio(bars: list[Bar], window: int = 20) -> float:
    if len(bars) <= window + 1:
        return 1.0
    recent = bars[-1].volume
    previous = [bar.volume for bar in bars[-window - 1 : -1] if bar.volume > 0]
    if not previous:
        return 1.0
    return recent / mean(previous)


def _with_score(feature: AssetFeature) -> AssetFeature:
    score = (
        feature.momentum_3m * 5
        + feature.momentum_6m * 3
        + feature.momentum_12m * 2
        - feature.volatility_3m * 1.8
        + feature.drawdown_12m * 1.2
        + min(feature.volume_ratio, 2.5) * 0.25
    )
    return AssetFeature(
        symbol=feature.symbol,
        price=feature.price,
        momentum_1m=feature.momentum_1m,
        momentum_3m=feature.momentum_3m,
        momentum_6m=feature.momentum_6m,
        momentum_12m=feature.momentum_12m,
        volatility_3m=feature.volatility_3m,
        drawdown_12m=feature.drawdown_12m,
        volume_ratio=feature.volume_ratio,
        correlation_spy=feature.correlation_spy,
        beta_spy=feature.beta_spy,
        quality_score=score,
    )
