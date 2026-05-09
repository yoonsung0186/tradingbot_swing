from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import mean
from time import sleep


@dataclass(frozen=True)
class PriceTick:
    timestamp: datetime
    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    volume: int | None = None
    source: str = "unknown"


@dataclass(frozen=True)
class RealtimeConfig:
    lookback_ticks: int = 6
    entry_momentum_pct: float = 0.003
    exit_momentum_pct: float = -0.002
    stop_loss_pct: float = 0.004
    take_profit_pct: float = 0.008
    trailing_stop_pct: float = 0.004
    max_spread_pct: float = 0.0025
    max_position_pct: float = 0.10
    cooldown_ticks: int = 12
    min_price: float = 5.0
    partial_take_profit_pct: float | None = None
    partial_take_profit_fraction: float = 0.0
    breakeven_after_pct: float | None = None
    breakeven_offset_pct: float = 0.0
    execution_slippage_bps: float = 2.0
    min_tick_volume: int = 0
    max_tick_age_seconds: int | None = None
    max_trades_per_day: int = 999


@dataclass(frozen=True)
class RealtimeDecision:
    timestamp: datetime
    symbol: str
    action: str
    price: float
    qty: float
    notional: float
    reason: str
    source: str


@dataclass
class RealtimePosition:
    qty: float
    entry_price: float
    high_watermark: float
    initial_qty: float
    partial_taken: bool = False
    breakeven_armed: bool = False


class RealtimeReactiveModel:
    def __init__(self, cash: float, config: RealtimeConfig | None = None) -> None:
        self.cash = cash
        self.config = config or RealtimeConfig()
        self.windows: dict[str, deque[PriceTick]] = defaultdict(lambda: deque(maxlen=self.config.lookback_ticks))
        self.positions: dict[str, RealtimePosition] = {}
        self.cooldowns: dict[str, int] = defaultdict(int)
        self.decisions: list[RealtimeDecision] = []
        self.daily_entry_counts: dict[date, int] = defaultdict(int)

    def update_config(self, config: RealtimeConfig) -> None:
        old_lookback = self.config.lookback_ticks
        self.config = config
        if config.lookback_ticks != old_lookback:
            self.windows = defaultdict(
                lambda: deque(maxlen=self.config.lookback_ticks),
                {
                    symbol: deque(list(window)[-self.config.lookback_ticks :], maxlen=self.config.lookback_ticks)
                    for symbol, window in self.windows.items()
                },
            )

    def mark_to_market(self, latest_prices: dict[str, float]) -> tuple[float, float]:
        market_value = 0.0
        for symbol, position in self.positions.items():
            price = latest_prices.get(symbol, position.entry_price)
            market_value += position.qty * price
        return self.cash + market_value, market_value

    def position_rows(self, latest_prices: dict[str, float]) -> list[dict[str, float | str]]:
        rows: list[dict[str, float | str]] = []
        for symbol, position in sorted(self.positions.items()):
            price = latest_prices.get(symbol, position.entry_price)
            market_value = position.qty * price
            rows.append(
                {
                    "symbol": symbol,
                    "qty": position.qty,
                    "entry_price": position.entry_price,
                    "latest_price": price,
                    "market_value": market_value,
                    "unrealized_pnl": market_value - position.qty * position.entry_price,
                    "unrealized_return": price / position.entry_price - 1 if position.entry_price else 0.0,
                }
            )
        return rows

    def on_tick(self, tick: PriceTick, allow_entry: bool = True) -> list[RealtimeDecision]:
        symbol = tick.symbol.upper()
        if tick.price < self.config.min_price:
            return []

        window = self.windows[symbol]
        window.append(tick)
        if self.cooldowns[symbol] > 0:
            self.cooldowns[symbol] -= 1

        decisions: list[RealtimeDecision] = []
        position = self.positions.get(symbol)
        if position:
            decisions.extend(self._manage_position(tick, position))
        elif allow_entry and self._entry_signal(symbol, tick):
            decisions.append(self._open_position(tick))

        self.decisions.extend(decisions)
        return decisions

    def _entry_signal(self, symbol: str, tick: PriceTick) -> bool:
        if self.cash <= 0 or self.config.max_position_pct <= 0:
            return False
        if self.cooldowns[symbol] > 0:
            return False
        if self._entry_count_for_day(tick) >= self.config.max_trades_per_day:
            return False
        if self._spread_pct(tick) > self.config.max_spread_pct:
            return False
        if self.config.min_tick_volume > 0 and tick.volume is not None and tick.volume < self.config.min_tick_volume:
            return False
        if self.config.max_tick_age_seconds is not None and self._tick_age_seconds(tick) > self.config.max_tick_age_seconds:
            return False

        window = self.windows[symbol]
        if len(window) < self.config.lookback_ticks:
            return False
        first = window[0].price
        if first <= 0:
            return False
        momentum = tick.price / first - 1
        average_price = mean(item.price for item in window)
        return momentum >= self.config.entry_momentum_pct and tick.price >= average_price

    def _open_position(self, tick: PriceTick) -> RealtimeDecision:
        fill_price = self._buy_fill_price(tick)
        notional = max(self.cash * self.config.max_position_pct, 0.0)
        qty = notional / fill_price if fill_price else 0.0
        self.cash -= notional
        self.positions[tick.symbol] = RealtimePosition(
            qty=qty,
            entry_price=fill_price,
            high_watermark=tick.price,
            initial_qty=qty,
        )
        self.daily_entry_counts[self._trade_day(tick)] += 1
        self.cooldowns[tick.symbol] = self.config.cooldown_ticks
        return RealtimeDecision(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            action="BUY_PLAN",
            price=fill_price,
            qty=qty,
            notional=notional,
            reason="momentum breakout with conservative ask-side fill",
            source=tick.source,
        )

    def _manage_position(self, tick: PriceTick, position: RealtimePosition) -> list[RealtimeDecision]:
        position.high_watermark = max(position.high_watermark, tick.price)
        sell_price = self._sell_fill_price(tick)
        entry_return = sell_price / position.entry_price - 1 if position.entry_price else 0.0
        trailing_return = tick.price / position.high_watermark - 1 if position.high_watermark else 0.0

        if self.config.breakeven_after_pct is not None and entry_return >= self.config.breakeven_after_pct:
            position.breakeven_armed = True

        partial = self._partial_take_profit(tick, position, entry_return)
        if partial:
            return [partial]

        reason = ""
        if entry_return <= -self.config.stop_loss_pct:
            reason = "stop loss"
        elif position.breakeven_armed and entry_return <= self.config.breakeven_offset_pct:
            reason = "breakeven stop"
        elif entry_return >= self.config.take_profit_pct:
            reason = "take profit"
        elif trailing_return <= -self.config.trailing_stop_pct:
            reason = "trailing stop"
        elif self._short_momentum(tick.symbol) <= self.config.exit_momentum_pct:
            reason = "momentum faded"

        if not reason:
            return []

        notional = position.qty * sell_price
        self.cash += notional
        self.positions.pop(tick.symbol, None)
        self.cooldowns[tick.symbol] = self.config.cooldown_ticks
        return [
            RealtimeDecision(
                timestamp=tick.timestamp,
                symbol=tick.symbol,
                action="SELL_PLAN",
                price=sell_price,
                qty=position.qty,
                notional=notional,
                reason=reason,
                source=tick.source,
            )
        ]

    def _partial_take_profit(
        self,
        tick: PriceTick,
        position: RealtimePosition,
        entry_return: float,
    ) -> RealtimeDecision | None:
        threshold = self.config.partial_take_profit_pct
        fraction = self.config.partial_take_profit_fraction
        if threshold is None or threshold <= 0:
            return None
        if position.partial_taken or not (0 < fraction < 1):
            return None
        if entry_return < threshold or position.qty <= 0:
            return None

        sell_qty = position.qty * fraction
        position.qty -= sell_qty
        position.partial_taken = True
        sell_price = self._sell_fill_price(tick)
        notional = sell_qty * sell_price
        self.cash += notional
        return RealtimeDecision(
            timestamp=tick.timestamp,
            symbol=tick.symbol,
            action="SELL_PARTIAL_PLAN",
            price=sell_price,
            qty=sell_qty,
            notional=notional,
            reason="partial take profit",
            source=tick.source,
        )

    def _short_momentum(self, symbol: str) -> float:
        window = self.windows[symbol]
        if len(window) < 2 or window[0].price <= 0:
            return 0.0
        return window[-1].price / window[0].price - 1

    def _buy_fill_price(self, tick: PriceTick) -> float:
        price = tick.ask if tick.ask and tick.ask > 0 else tick.price
        return price * (1 + self.config.execution_slippage_bps / 10_000)

    def _sell_fill_price(self, tick: PriceTick) -> float:
        price = tick.bid if tick.bid and tick.bid > 0 else tick.price
        return price * (1 - self.config.execution_slippage_bps / 10_000)

    @staticmethod
    def _tick_age_seconds(tick: PriceTick) -> float:
        timestamp = tick.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - timestamp.astimezone(timezone.utc)).total_seconds(), 0.0)

    def _entry_count_for_day(self, tick: PriceTick) -> int:
        return self.daily_entry_counts[self._trade_day(tick)]

    @staticmethod
    def _trade_day(tick: PriceTick) -> date:
        timestamp = tick.timestamp
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).date()

    @staticmethod
    def _spread_pct(tick: PriceTick) -> float:
        if tick.bid is None or tick.ask is None or tick.price <= 0:
            return 0.0
        return max(tick.ask - tick.bid, 0.0) / tick.price


class JsonlTickFeed:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.offset = 0

    def read_new(self) -> list[PriceTick]:
        if not self.path.exists():
            return []
        ticks: list[PriceTick] = []
        with self.path.open("r", encoding="utf-8") as handle:
            handle.seek(self.offset)
            for line in handle:
                line = line.strip()
                if line:
                    ticks.append(parse_tick_json(line))
            self.offset = handle.tell()
        return ticks


def parse_tick_json(raw: str) -> PriceTick:
    item = json.loads(raw)
    timestamp = item.get("timestamp") or item.get("ts")
    if isinstance(timestamp, str):
        parsed_timestamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    else:
        parsed_timestamp = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
    return PriceTick(
        timestamp=parsed_timestamp,
        symbol=str(item["symbol"]).upper(),
        price=float(item["price"]),
        bid=float(item["bid"]) if item.get("bid") is not None else None,
        ask=float(item["ask"]) if item.get("ask") is not None else None,
        volume=int(item["volume"]) if item.get("volume") is not None else None,
        source=str(item.get("source", "jsonl")),
    )


def run_tick_file_dry_run(
    path: Path,
    symbols: list[str],
    cash: float,
    config: RealtimeConfig | None = None,
    watch_seconds: int = 0,
    poll_seconds: int = 10,
) -> tuple[RealtimeReactiveModel, list[RealtimeDecision]]:
    allowed = {symbol.upper() for symbol in symbols}
    feed = JsonlTickFeed(path)
    model = RealtimeReactiveModel(cash=cash, config=config)
    decisions: list[RealtimeDecision] = []

    elapsed = 0
    while True:
        ticks = [tick for tick in feed.read_new() if not allowed or tick.symbol in allowed]
        ticks.sort(key=lambda item: item.timestamp)
        for tick in ticks:
            decisions.extend(model.on_tick(tick))
        if watch_seconds <= 0 or elapsed >= watch_seconds:
            break
        sleep(max(poll_seconds, 1))
        elapsed += max(poll_seconds, 1)
    return model, decisions
