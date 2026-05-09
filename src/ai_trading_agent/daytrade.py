from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from statistics import mean
from zoneinfo import ZoneInfo

from .metrics import performance_stats
from .models import IntradayBar


NEW_YORK = ZoneInfo("America/New_York")
KST = ZoneInfo("Asia/Seoul")
MARKET_OPEN = time(9, 30)
MARKET_CLOSE = time(16, 0)


@dataclass(frozen=True)
class DayTradeConfig:
    model_name: str = "ma_bollinger_runner"
    primary_indicator: str = "ma_bollinger"
    enabled_filters: tuple[str, ...] = ("volume", "bullish")
    opening_minutes: int = 30
    interval_minutes: int = 5
    per_trade_risk: float = 0.01
    max_position_pct: float = 0.40
    max_daily_loss_pct: float = 0.03
    max_trades_per_day: int = 3
    stop_pct: float = 0.005
    take_profit_r: float = 2.0
    cost_bps: float = 5.0
    slippage_bps: float = 15.0
    min_signal_volume_ratio: float = 1.60
    min_bar_dollar_volume: float = 3_000_000.0
    momentum_lookback_bars: int = 6
    min_momentum_pct: float = 0.0015
    fast_ema_window: int = 9
    slow_ema_window: int = 20
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    min_bollinger_position: float = 0.85
    min_opening_range_pct: float = 0.002
    max_opening_range_pct: float = 0.035
    vwap_buffer_pct: float = 0.0008
    require_market_confirmation: bool = True
    market_symbols: tuple[str, ...] = ("SPY", "QQQ")
    min_market_confirmations: int = 1
    min_market_return_pct: float = 0.0
    market_vwap_buffer_pct: float = 0.0
    breakeven_after_r: float = 0.0
    max_hold_minutes: int = 0
    last_entry_minutes_before_close: int = 60
    flatten_minutes_before_close: int = 10


@dataclass(frozen=True)
class DayTradeCandidate:
    symbol: str
    side: str
    signal_time: datetime
    entry_time: datetime
    exit_time: datetime
    raw_entry_price: float
    raw_exit_price: float
    stop_price: float
    take_profit_price: float
    score: float
    exit_reason: str


def run_daytrade_backtest(
    histories: dict[str, list[IntradayBar]],
    cash: float,
    config: DayTradeConfig | None = None,
) -> dict:
    config = config or DayTradeConfig()
    sessions = _sessions_by_day(histories)
    all_days = sorted({session_date for symbol_days in sessions.values() for session_date in symbol_days})
    if not all_days:
        return _empty_result(cash, config)

    equity = cash
    equity_curve: list[tuple[date, float]] = []
    trade_log: list[dict] = []
    daily_rows: list[dict] = []

    for session_date in all_days:
        start_equity = equity
        day_loss_limit = start_equity * config.max_daily_loss_pct
        day_pnl = 0.0
        day_trades = 0
        traded_symbols: set[str] = set()
        candidates: list[DayTradeCandidate] = []

        context_symbols = {symbol.upper() for symbol in config.market_symbols}
        for symbol, by_day in sessions.items():
            if symbol.upper() in context_symbols:
                continue
            bars = by_day.get(session_date, [])
            candidate = _opening_range_breakout_candidate(symbol, bars, config)
            if candidate and _market_confirmed(sessions, session_date, candidate.signal_time, config):
                candidates.append(candidate)

        candidates.sort(key=lambda item: (item.entry_time, -item.score, item.symbol))
        for candidate in candidates:
            if day_trades >= config.max_trades_per_day:
                break
            if candidate.symbol in traded_symbols:
                continue
            if day_pnl <= -day_loss_limit:
                break

            entry_price = _buy_fill(candidate.raw_entry_price, config)
            exit_price = _sell_fill(candidate.raw_exit_price, config)
            risk_per_share = max(entry_price - candidate.stop_price, entry_price * 0.001)
            qty_by_risk = (start_equity * config.per_trade_risk) / risk_per_share
            qty_by_exposure = (equity * config.max_position_pct) / entry_price
            qty = max(min(qty_by_risk, qty_by_exposure), 0.0)
            if qty <= 0.000001:
                continue

            notional = qty * entry_price
            pnl = qty * (exit_price - entry_price)
            pnl_pct = pnl / notional if notional else 0.0
            equity += pnl
            day_pnl += pnl
            day_trades += 1
            traded_symbols.add(candidate.symbol)
            trade_log.append(
                {
                    "date": session_date.isoformat(),
                    "symbol": candidate.symbol,
                    "side": candidate.side,
                    "entry_time": _format_time(candidate.entry_time),
                    "exit_time": _format_time(candidate.exit_time),
                    "qty": qty,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "notional": notional,
                    "realized_pnl": pnl,
                    "pnl_pct": pnl_pct,
                    "stop_price": candidate.stop_price,
                    "take_profit_price": candidate.take_profit_price,
                    "reason": candidate.exit_reason,
                    "model": config.model_name,
                }
            )

        equity_curve.append((session_date, equity))
        daily_rows.append(
            {
                "date": session_date.isoformat(),
                "start_equity": start_equity,
                "end_equity": equity,
                "daily_pnl": equity - start_equity,
                "daily_return": (equity / start_equity - 1) if start_equity else 0.0,
                "trades": day_trades,
            }
        )

    stats_curve = [(all_days[0] - timedelta(days=1), cash), *equity_curve]
    stats = performance_stats(stats_curve)
    wins = [trade for trade in trade_log if trade["realized_pnl"] > 0]
    losses = [trade for trade in trade_log if trade["realized_pnl"] < 0]
    gross_profit = sum(trade["realized_pnl"] for trade in wins)
    gross_loss = abs(sum(trade["realized_pnl"] for trade in losses))
    returns = [trade["pnl_pct"] for trade in trade_log]

    return {
        "start_equity": cash,
        "end_equity": equity,
        "total_return": equity / cash - 1 if cash else 0.0,
        "cagr": stats.cagr,
        "max_drawdown": stats.max_drawdown,
        "annual_volatility": stats.annual_volatility,
        "sharpe": stats.sharpe,
        "sortino": stats.sortino,
        "calmar": stats.calmar,
        "days": len(equity_curve),
        "trades": len(trade_log),
        "win_rate": len(wins) / len(trade_log) if trade_log else 0.0,
        "profit_factor": gross_profit / gross_loss if gross_loss else (float("inf") if gross_profit else 0.0),
        "average_trade_return": mean(returns) if returns else 0.0,
        "best_trade_return": max(returns) if returns else 0.0,
        "worst_trade_return": min(returns) if returns else 0.0,
        "trade_log": trade_log,
        "daily_rows": daily_rows,
        "equity_curve": equity_curve,
        "config": config.__dict__,
    }


def _opening_range_breakout_candidate(
    symbol: str,
    bars: list[IntradayBar],
    config: DayTradeConfig,
) -> DayTradeCandidate | None:
    session_bars = _regular_session_bars(bars)
    opening_count = max(config.opening_minutes // config.interval_minutes, 1)
    if len(session_bars) <= opening_count + 1:
        return None

    opening_bars = session_bars[:opening_count]
    opening_high = max(bar.high for bar in opening_bars)
    opening_low = min(bar.low for bar in opening_bars)
    reference_price = opening_bars[-1].close
    if reference_price <= 0:
        return None
    opening_range_pct = (opening_high - opening_low) / reference_price
    if opening_range_pct < config.min_opening_range_pct or opening_range_pct > config.max_opening_range_pct:
        return None

    cumulative_volume = 0
    cumulative_price_volume = 0.0
    for bar in opening_bars:
        typical = (bar.high + bar.low + bar.close) / 3
        cumulative_volume += bar.volume
        cumulative_price_volume += typical * bar.volume

    for idx in range(opening_count, len(session_bars) - 1):
        bar = session_bars[idx]
        typical = (bar.high + bar.low + bar.close) / 3
        cumulative_volume += bar.volume
        cumulative_price_volume += typical * bar.volume
        if cumulative_volume <= 0:
            continue

        vwap = cumulative_price_volume / cumulative_volume
        lookback = session_bars[max(0, idx - 20) : idx]
        avg_volume = mean([item.volume for item in lookback]) if lookback else 0.0
        volume_ratio = bar.volume / avg_volume if avg_volume else 0.0
        dollar_volume = bar.close * bar.volume
        if dollar_volume < config.min_bar_dollar_volume:
            continue
        momentum = _bar_momentum(session_bars, idx, config.momentum_lookback_bars)
        ma = _ma_bollinger_state(session_bars, idx, config)
        indicators = {
            "orb_breakout": bar.close > opening_high,
            "vwap_trend": bar.close > vwap * (1 + config.vwap_buffer_pct),
            "volume_momentum": volume_ratio >= config.min_signal_volume_ratio and momentum >= config.min_momentum_pct,
            "pullback_vwap": bar.low <= vwap * (1 + config.vwap_buffer_pct) and bar.close > vwap * (1 + config.vwap_buffer_pct),
            "ma_bollinger": ma["trend_ok"] and ma["band_position"] >= config.min_bollinger_position,
            "ema_trend": ma["trend_ok"],
            "bollinger_upper": ma["upper"] > 0 and bar.close >= ma["upper"],
            "bollinger_near_upper": ma["band_position"] >= config.min_bollinger_position,
            "volume": volume_ratio >= config.min_signal_volume_ratio,
            "vwap": bar.close > vwap * (1 + config.vwap_buffer_pct),
            "bullish": bar.close > bar.open,
            "momentum": momentum >= config.min_momentum_pct,
        }
        if not indicators.get(config.primary_indicator, False):
            continue
        if any(not indicators.get(filter_name, False) for filter_name in config.enabled_filters):
            continue

        entry_bar = session_bars[idx + 1]
        if entry_bar.timestamp.astimezone(NEW_YORK).time() >= _last_entry_time(config):
            continue
        raw_entry = entry_bar.open
        if raw_entry <= 0:
            continue
        stop_price = max(opening_low, raw_entry * (1 - config.stop_pct))
        if stop_price >= raw_entry:
            stop_price = raw_entry * (1 - config.stop_pct)
        take_profit = raw_entry + (raw_entry - stop_price) * config.take_profit_r
        raw_exit, exit_time, exit_reason = _simulate_long_exit(
            session_bars,
            idx + 1,
            raw_entry,
            stop_price,
            take_profit,
            config,
        )
        breakout_strength = bar.close / opening_high - 1 if opening_high else 0.0
        vwap_strength = bar.close / vwap - 1 if vwap else 0.0
        ema_spread = ma["fast_ema"] / ma["slow_ema"] - 1 if ma["slow_ema"] > 0 else 0.0
        band_score = max(ma["band_position"], 0.0)
        score = breakout_strength * 100 + vwap_strength * 80 + momentum * 60 + ema_spread * 100 + band_score + volume_ratio
        return DayTradeCandidate(
            symbol=symbol,
            side="LONG",
            signal_time=bar.timestamp,
            entry_time=entry_bar.timestamp,
            exit_time=exit_time,
            raw_entry_price=raw_entry,
            raw_exit_price=raw_exit,
            stop_price=stop_price,
            take_profit_price=take_profit,
            score=score,
            exit_reason=exit_reason,
        )

    return None


def _simulate_long_exit(
    bars: list[IntradayBar],
    entry_idx: int,
    entry_price: float,
    stop_price: float,
    take_profit: float,
    config: DayTradeConfig,
) -> tuple[float, datetime, str]:
    flatten_time = _flatten_time(config)
    max_hold_until = bars[entry_idx].timestamp + timedelta(minutes=config.max_hold_minutes) if config.max_hold_minutes else None
    active_stop = stop_price
    risk_per_share = max(entry_price - stop_price, entry_price * 0.001)
    for bar in bars[entry_idx:]:
        local_time = bar.timestamp.astimezone(NEW_YORK).time()
        if bar.low <= active_stop:
            return active_stop, bar.timestamp, "stop_loss"
        if bar.high >= take_profit:
            return take_profit, bar.timestamp, "take_profit"
        if config.breakeven_after_r > 0 and bar.high >= entry_price + risk_per_share * config.breakeven_after_r:
            active_stop = max(active_stop, entry_price)
        if max_hold_until and bar.timestamp >= max_hold_until:
            return bar.close, bar.timestamp, "max_hold"
        if local_time >= flatten_time:
            return bar.close, bar.timestamp, "end_of_day"
    last = bars[-1]
    return last.close, last.timestamp, "end_of_day"


def _sessions_by_day(histories: dict[str, list[IntradayBar]]) -> dict[str, dict[date, list[IntradayBar]]]:
    sessions: dict[str, dict[date, list[IntradayBar]]] = {}
    for symbol, bars in histories.items():
        by_day: dict[date, list[IntradayBar]] = {}
        for bar in bars:
            local = bar.timestamp.astimezone(NEW_YORK)
            if MARKET_OPEN <= local.time() <= MARKET_CLOSE:
                by_day.setdefault(local.date(), []).append(bar)
        for session_bars in by_day.values():
            session_bars.sort(key=lambda item: item.timestamp)
        sessions[symbol] = by_day
    return sessions


def _regular_session_bars(bars: list[IntradayBar]) -> list[IntradayBar]:
    result = []
    for bar in bars:
        local_time = bar.timestamp.astimezone(NEW_YORK).time()
        if MARKET_OPEN <= local_time <= MARKET_CLOSE:
            result.append(bar)
    return result


def _bar_momentum(bars: list[IntradayBar], idx: int, lookback: int) -> float:
    previous_idx = idx - max(lookback, 1)
    if previous_idx < 0 or bars[previous_idx].close <= 0:
        return 0.0
    return bars[idx].close / bars[previous_idx].close - 1


def _ma_bollinger_state(bars: list[IntradayBar], idx: int, config: DayTradeConfig) -> dict[str, float | bool]:
    closes = [bar.close for bar in bars[: idx + 1]]
    fast_ema = _ema(closes, config.fast_ema_window)
    slow_ema = _ema(closes, config.slow_ema_window)
    previous_slow_ema = _ema(closes[:-1], config.slow_ema_window) if len(closes) > 1 else 0.0
    band_window = max(config.bollinger_window, 2)
    if len(closes) < max(config.fast_ema_window, config.slow_ema_window, band_window):
        return {
            "fast_ema": fast_ema,
            "slow_ema": slow_ema,
            "upper": 0.0,
            "lower": 0.0,
            "band_position": 0.0,
            "trend_ok": False,
        }

    window = closes[-band_window:]
    middle = mean(window)
    variance = sum((close - middle) ** 2 for close in window) / len(window)
    std_dev = variance ** 0.5
    upper = middle + config.bollinger_std * std_dev
    lower = middle - config.bollinger_std * std_dev
    band_width = upper - lower
    band_position = (closes[-1] - lower) / band_width if band_width > 0 else 0.0
    trend_ok = (
        closes[-1] > slow_ema
        and closes[-1] > fast_ema
        and fast_ema > slow_ema
        and slow_ema >= previous_slow_ema
    )
    return {
        "fast_ema": fast_ema,
        "slow_ema": slow_ema,
        "upper": upper,
        "lower": lower,
        "band_position": band_position,
        "trend_ok": trend_ok,
    }


def _ema(values: list[float], window: int) -> float:
    if not values:
        return 0.0
    period = max(window, 1)
    alpha = 2 / (period + 1)
    ema = values[0]
    for value in values[1:]:
        ema = value * alpha + ema * (1 - alpha)
    return ema


def _market_confirmed(
    sessions: dict[str, dict[date, list[IntradayBar]]],
    session_date: date,
    signal_time: datetime,
    config: DayTradeConfig,
) -> bool:
    if not config.require_market_confirmation:
        return True

    confirmed = 0
    for symbol in config.market_symbols:
        bars = sessions.get(symbol.upper(), {}).get(session_date, [])
        if _market_symbol_confirmed(bars, signal_time, config):
            confirmed += 1
    return confirmed >= config.min_market_confirmations


def _market_symbol_confirmed(
    bars: list[IntradayBar],
    signal_time: datetime,
    config: DayTradeConfig,
) -> bool:
    observed = [bar for bar in _regular_session_bars(bars) if bar.timestamp <= signal_time]
    if len(observed) < 3:
        return False

    cumulative_volume = sum(bar.volume for bar in observed)
    if cumulative_volume <= 0:
        return False
    cumulative_price_volume = sum(((bar.high + bar.low + bar.close) / 3) * bar.volume for bar in observed)
    vwap = cumulative_price_volume / cumulative_volume
    latest = observed[-1]
    session_open = observed[0].open
    if latest.close < vwap * (1 + config.market_vwap_buffer_pct):
        return False
    if session_open and latest.close / session_open - 1 < config.min_market_return_pct:
        return False
    return True


def _buy_fill(price: float, config: DayTradeConfig) -> float:
    return price * (1 + _cost_rate(config))


def _sell_fill(price: float, config: DayTradeConfig) -> float:
    return price * (1 - _cost_rate(config))


def _cost_rate(config: DayTradeConfig) -> float:
    return (config.cost_bps + config.slippage_bps) / 10_000


def _flatten_time(config: DayTradeConfig) -> time:
    close_dt = datetime.combine(date.today(), MARKET_CLOSE)
    return (close_dt - timedelta(minutes=config.flatten_minutes_before_close)).time()


def _last_entry_time(config: DayTradeConfig) -> time:
    close_dt = datetime.combine(date.today(), MARKET_CLOSE)
    return (close_dt - timedelta(minutes=config.last_entry_minutes_before_close)).time()


def _format_time(value: datetime) -> str:
    return value.astimezone(KST).strftime("%m-%d %H:%M KST")


def _empty_result(cash: float, config: DayTradeConfig) -> dict:
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
        "days": 0,
        "trades": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "average_trade_return": 0.0,
        "best_trade_return": 0.0,
        "worst_trade_return": 0.0,
        "trade_log": [],
        "daily_rows": [],
        "equity_curve": [],
        "config": config.__dict__,
    }
