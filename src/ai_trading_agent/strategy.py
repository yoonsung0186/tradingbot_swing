from __future__ import annotations

from statistics import mean

from .config import DEFENSIVE_SYMBOLS, StrategyConfig
from .models import Bar, Signal


def sma(values: list[float], window: int) -> float:
    if len(values) < window:
        raise ValueError("not enough values for moving average")
    return mean(values[-window:])


class MomentumStrategy:
    def __init__(self, config: StrategyConfig | None = None) -> None:
        self.config = config or StrategyConfig()

    def market_regime(self, spy_bars: list[Bar]) -> tuple[bool, str]:
        closes = [bar.close for bar in spy_bars]
        if len(closes) < self.config.long_window:
            return False, "SPY history is too short for regime check"
        spy_close = closes[-1]
        spy_sma = sma(closes, self.config.long_window)
        risk_on = spy_close > spy_sma
        label = "risk-on" if risk_on else "risk-off"
        return risk_on, f"{label}: SPY close {spy_close:.2f} vs {self.config.long_window}d SMA {spy_sma:.2f}"

    def generate(
        self,
        histories: dict[str, list[Bar]],
        market_risk_on: bool,
    ) -> list[Signal]:
        signals: list[Signal] = []
        for symbol, bars in histories.items():
            signal = self._signal_for_symbol(symbol, bars, market_risk_on)
            if signal:
                signals.append(signal)
        signals.sort(key=lambda item: item.score, reverse=True)
        return signals[: self.config.max_candidates]

    def _signal_for_symbol(
        self,
        symbol: str,
        bars: list[Bar],
        market_risk_on: bool,
    ) -> Signal | None:
        required = max(self.config.long_window, self.config.volume_window) + 1
        if len(bars) < required:
            return None

        latest = bars[-1]
        closes = [bar.close for bar in bars]
        volumes = [bar.volume for bar in bars]
        short_sma = sma(closes, self.config.short_window)
        long_sma = sma(closes, self.config.long_window)
        volume_avg = mean(volumes[-self.config.volume_window - 1 : -1])
        volume_ratio = latest.volume / volume_avg if volume_avg > 0 else 1.0
        high_20 = max(closes[-self.config.short_window :])
        momentum_20 = latest.close / closes[-self.config.short_window] - 1
        above_trend = latest.close > short_sma > long_sma
        near_high = latest.close >= high_20 * self.config.near_high_pct
        volume_ok = volume_ratio >= self.config.min_volume_ratio
        allowed_by_regime = market_risk_on or symbol in DEFENSIVE_SYMBOLS

        score = (
            momentum_20 * 100
            + max(latest.close / long_sma - 1, 0) * 60
            + min(volume_ratio, 3.0) * 4
        )
        action = "BUY" if above_trend and near_high and volume_ok and allowed_by_regime else "HOLD"
        reasons = [
            f"20d momentum {momentum_20:.1%}",
            f"close/SMA50 {latest.close / long_sma:.2f}",
            f"volume ratio {volume_ratio:.2f}",
        ]
        if not market_risk_on and symbol not in DEFENSIVE_SYMBOLS:
            reasons.append("blocked by risk-off regime")
        if action == "HOLD":
            score *= 0.35

        stop_loss = latest.close * (1 - self.config.stop_loss_pct)
        take_profit = latest.close * (1 + self.config.take_profit_pct)
        return Signal(
            symbol=symbol,
            action=action,
            score=score,
            price=latest.close,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reason="; ".join(reasons),
        )
