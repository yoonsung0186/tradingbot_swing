from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from statistics import mean

from .metrics import annualized_volatility, max_drawdown, pct_change, performance_stats
from .models import Bar
from .strategy import sma


@dataclass(frozen=True)
class BacktestConfig:
    short_window: int = 63
    medium_window: int = 126
    long_window: int = 200
    top_n: int = 3
    rebalance_interval: int = 5
    min_momentum: float = 0.0
    max_volatility: float = 0.45
    max_weight: float = 0.40
    stop_loss_pct: float = 0.10
    cost_bps: float = 5.0
    slippage_bps: float = 10.0
    min_dollar_volume: float = 20_000_000.0
    min_history_days: int = 252
    use_regime_filter: bool = True
    market_symbol: str = "SPY"
    volatility_symbol: str = "^VIX"
    vix_threshold: float = 30.0
    risk_off_symbol: str = "SHY"
    tradable_symbols: tuple[str, ...] = ()


def run_backtest(
    histories: dict[str, list[Bar]],
    cash: float,
    start: date | None = None,
    end: date | None = None,
    config: BacktestConfig | None = None,
) -> dict:
    config = config or BacktestConfig()
    tradable_symbols = _tradable_symbols(histories, config)
    tradable_histories = {symbol: histories[symbol] for symbol in tradable_symbols}
    all_dates = sorted({bar.date for bars in tradable_histories.values() for bar in bars})
    if start:
        all_dates = [item for item in all_dates if item >= start]
    if end:
        all_dates = [item for item in all_dates if item <= end]

    if not all_dates:
        return _empty_result(cash, config)

    indexes = {symbol: -1 for symbol in histories}
    portfolio_cash = cash
    positions: dict[str, float] = {}
    entry_prices: dict[str, float] = {}
    cost_basis_prices: dict[str, float] = {}
    equity_curve: list[tuple[date, float]] = []
    trade_log: list[dict] = []
    trades = 0
    last_signal_day = -10_000
    pending_weights: dict[str, float] | None = None

    for day_index, current_date in enumerate(all_dates):
        _advance_indexes(histories, indexes, current_date)
        close_prices = _prices_for_indexes(histories, indexes, field="close")
        open_prices = _prices_for_indexes(histories, indexes, field="open")
        if not close_prices:
            continue

        if pending_weights is not None:
            portfolio_cash, rebalance_trades = _rebalance(
                portfolio_cash,
                positions,
                entry_prices,
                cost_basis_prices,
                open_prices,
                pending_weights,
                config,
                trade_log,
                current_date,
            )
            trades += rebalance_trades
            pending_weights = None

        portfolio_cash, exit_trades = _apply_stop_losses(
            portfolio_cash,
            positions,
            entry_prices,
            cost_basis_prices,
            histories,
            indexes,
            config,
            trade_log,
            current_date,
        )
        trades += exit_trades

        equity = _portfolio_equity(portfolio_cash, positions, close_prices)
        equity_curve.append((current_date, equity))

        if day_index - last_signal_day < config.rebalance_interval:
            continue

        # Signals are generated after today's close. Any portfolio change is
        # queued for the next available trading day's open.
        weights = _target_weights(histories, indexes, config)
        if weights is not None:
            pending_weights = weights
            last_signal_day = day_index

    stats = performance_stats(equity_curve)
    return {
        "start_equity": stats.start_equity,
        "end_equity": stats.end_equity,
        "total_return": stats.total_return,
        "cagr": stats.cagr,
        "max_drawdown": stats.max_drawdown,
        "annual_volatility": stats.annual_volatility,
        "sharpe": stats.sharpe,
        "sortino": stats.sortino,
        "calmar": stats.calmar,
        "days": stats.days,
        "trades": trades,
        "trade_log": trade_log,
        "config": asdict(config),
        "equity_curve": equity_curve,
    }


def buy_and_hold_curve(bars: list[Bar], cash: float, start: date | None = None) -> list[tuple[date, float]]:
    filtered = [bar for bar in bars if start is None or bar.date >= start]
    if not filtered:
        return []
    shares = cash / filtered[0].open
    return [(bar.date, shares * bar.close) for bar in filtered]


def latest_target_weights(histories: dict[str, list[Bar]], config: BacktestConfig) -> dict[str, float]:
    indexes = {symbol: len(bars) - 1 for symbol, bars in histories.items() if bars}
    if not indexes:
        return {}
    return _target_weights(histories, indexes, config) or {}


def _empty_result(cash: float, config: BacktestConfig) -> dict:
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
        "trade_log": [],
        "config": asdict(config),
        "equity_curve": [],
    }


def _advance_indexes(histories: dict[str, list[Bar]], indexes: dict[str, int], current_date: date) -> None:
    for symbol, bars in histories.items():
        while indexes[symbol] + 1 < len(bars) and bars[indexes[symbol] + 1].date <= current_date:
            indexes[symbol] += 1


def _prices_for_indexes(
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    field: str,
) -> dict[str, float]:
    prices: dict[str, float] = {}
    for symbol, idx in indexes.items():
        if idx >= 0:
            prices[symbol] = float(getattr(histories[symbol][idx], field))
    return prices


def _portfolio_equity(cash: float, positions: dict[str, float], prices: dict[str, float]) -> float:
    return cash + sum(qty * prices.get(symbol, 0.0) for symbol, qty in positions.items())


def _apply_stop_losses(
    cash: float,
    positions: dict[str, float],
    entry_prices: dict[str, float],
    cost_basis_prices: dict[str, float],
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: BacktestConfig,
    trade_log: list[dict],
    current_date: date,
) -> tuple[float, int]:
    trades = 0
    cost = _cost_rate(config)
    for symbol in list(positions):
        idx = indexes.get(symbol, -1)
        if idx < 0 or symbol not in entry_prices:
            continue
        bar = histories[symbol][idx]
        stop_price = entry_prices[symbol] * (1 - config.stop_loss_pct)
        if bar.low <= stop_price:
            fill_price = min(stop_price, bar.open)
            qty = positions.pop(symbol)
            proceeds = qty * fill_price * (1 - cost)
            basis = cost_basis_prices.get(symbol, entry_prices[symbol])
            realized_pnl = proceeds - qty * basis
            cash += proceeds
            entry_prices.pop(symbol, None)
            cost_basis_prices.pop(symbol, None)
            trade_log.append(
                _trade_record(
                    current_date,
                    symbol,
                    "SELL",
                    qty,
                    fill_price,
                    proceeds,
                    realized_pnl,
                    basis,
                    "stop_loss",
                )
            )
            trades += 1
    return cash, trades


def _target_weights(
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: BacktestConfig,
) -> dict[str, float] | None:
    risk_on = _risk_on(histories, indexes, config)
    if config.use_regime_filter and not risk_on:
        return {}

    ranked: list[tuple[float, float, str]] = []
    for symbol in _tradable_symbols(histories, config):
        bars = histories[symbol]
        score_and_vol = _score_symbol(bars, indexes.get(symbol, -1), config)
        if score_and_vol is None:
            continue
        score, volatility = score_and_vol
        ranked.append((score, volatility, symbol))
    ranked.sort(reverse=True)
    selected = ranked[: config.top_n]
    if not selected:
        return {}

    inverse_volatility = [(1 / max(volatility, 0.05), symbol) for _, volatility, symbol in selected]
    total_inverse = sum(item for item, _ in inverse_volatility)
    if not total_inverse:
        return {}

    weights: dict[str, float] = {}
    for inv_vol, symbol in inverse_volatility:
        weights[symbol] = min(inv_vol / total_inverse, config.max_weight)

    total_weight = sum(weights.values())
    if total_weight > 1:
        weights = {symbol: weight / total_weight for symbol, weight in weights.items()}
    return weights


def _risk_on(histories: dict[str, list[Bar]], indexes: dict[str, int], config: BacktestConfig) -> bool:
    if not config.use_regime_filter:
        return True
    market_bars = histories.get(config.market_symbol)
    market_index = indexes.get(config.market_symbol, -1)
    if not market_bars or market_index < config.long_window:
        return False
    market_closes = [bar.close for bar in market_bars[: market_index + 1]]
    if market_closes[-1] <= sma(market_closes, config.long_window):
        return False

    vix_bars = histories.get(config.volatility_symbol)
    vix_index = indexes.get(config.volatility_symbol, -1)
    if vix_bars and vix_index >= 0 and vix_bars[vix_index].close >= config.vix_threshold:
        return False
    return True


def _tradable_symbols(histories: dict[str, list[Bar]], config: BacktestConfig) -> list[str]:
    if config.tradable_symbols:
        return [symbol for symbol in config.tradable_symbols if symbol in histories]
    return [symbol for symbol in histories if not symbol.startswith("^")]


def _score_symbol(bars: list[Bar], index: int, config: BacktestConfig) -> tuple[float, float] | None:
    required = max(config.long_window, config.medium_window, config.short_window, config.min_history_days) + 1
    if index < required:
        return None

    history = bars[: index + 1]
    closes = [bar.close for bar in history]
    price = closes[-1]
    if price <= 0 or price <= sma(closes, config.long_window):
        return None
    if _average_dollar_volume(history) < config.min_dollar_volume:
        return None

    momentum_short = price / closes[-config.short_window] - 1
    momentum_medium = price / closes[-config.medium_window] - 1
    momentum_long = price / closes[-min(252, len(closes) - 1)] - 1
    if momentum_medium < config.min_momentum:
        return None

    recent_returns = pct_change(closes[-64:])
    volatility = annualized_volatility(recent_returns)
    if volatility > config.max_volatility:
        return None

    drawdown = max_drawdown(closes[-252:])
    score = (
        momentum_short * 1.2
        + momentum_medium * 1.8
        + momentum_long * 0.8
        - volatility * 0.7
        + drawdown * 0.5
    )
    return score, volatility


def _average_dollar_volume(bars: list[Bar], window: int = 20) -> float:
    recent = bars[-window:]
    if not recent:
        return 0.0
    return mean(bar.close * bar.volume for bar in recent)


def _rebalance(
    cash: float,
    positions: dict[str, float],
    entry_prices: dict[str, float],
    cost_basis_prices: dict[str, float],
    prices: dict[str, float],
    weights: dict[str, float],
    config: BacktestConfig,
    trade_log: list[dict],
    current_date: date,
) -> tuple[float, int]:
    trades = 0
    cost = _cost_rate(config)
    equity = _portfolio_equity(cash, positions, prices)

    for symbol in list(positions):
        price = prices.get(symbol)
        if not price:
            continue
        target_value = weights.get(symbol, 0.0) * equity
        current_value = positions[symbol] * price
        if current_value > target_value * 1.01:
            sell_value = current_value - target_value
            sell_qty = min(positions[symbol], sell_value / price)
            proceeds = sell_qty * price * (1 - cost)
            basis = cost_basis_prices.get(symbol, entry_prices.get(symbol, price))
            realized_pnl = proceeds - sell_qty * basis
            cash += proceeds
            positions[symbol] -= sell_qty
            trade_log.append(
                _trade_record(
                    current_date,
                    symbol,
                    "SELL",
                    sell_qty,
                    price,
                    proceeds,
                    realized_pnl,
                    basis,
                    "rebalance",
                )
            )
            trades += 1
            if positions[symbol] <= 0.000001:
                positions.pop(symbol, None)
                entry_prices.pop(symbol, None)
                cost_basis_prices.pop(symbol, None)

    for symbol, weight in weights.items():
        price = prices.get(symbol)
        if not price or price <= 0:
            continue
        target_value = weight * equity
        current_value = positions.get(symbol, 0.0) * price
        buy_value = max(target_value - current_value, 0.0)
        if buy_value <= 0:
            continue
        buy_value = min(buy_value, cash / (1 + cost))
        if buy_value <= 0:
            continue
        qty = buy_value / price
        if qty <= 0.000001 or buy_value < 1.0:
            continue
        gross_cost = buy_value * (1 + cost)
        cash -= gross_cost
        existing_qty = positions.get(symbol, 0.0)
        new_basis = gross_cost / qty
        if existing_qty:
            entry_prices[symbol] = ((existing_qty * entry_prices.get(symbol, price)) + buy_value) / (existing_qty + qty)
            cost_basis_prices[symbol] = (
                (existing_qty * cost_basis_prices.get(symbol, new_basis)) + gross_cost
            ) / (existing_qty + qty)
        else:
            entry_prices[symbol] = price
            cost_basis_prices[symbol] = new_basis
        positions[symbol] = existing_qty + qty
        trade_log.append(
            _trade_record(
                current_date,
                symbol,
                "BUY",
                qty,
                price,
                gross_cost,
                None,
                None,
                "rebalance",
            )
        )
        trades += 1

    return cash, trades


def _cost_rate(config: BacktestConfig) -> float:
    return (config.cost_bps + config.slippage_bps) / 10_000


def _trade_record(
    trade_date: date,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    notional: float,
    realized_pnl: float | None,
    basis: float | None,
    reason: str,
) -> dict:
    pnl_pct: float | str = ""
    if realized_pnl is not None and basis and qty:
        pnl_pct = realized_pnl / (basis * qty)
    return {
        "date": trade_date.isoformat(),
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "notional": notional,
        "realized_pnl": realized_pnl if realized_pnl is not None else "",
        "pnl_pct": pnl_pct,
        "reason": reason,
    }
