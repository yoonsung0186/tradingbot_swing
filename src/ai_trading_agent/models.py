from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime


@dataclass(frozen=True)
class Bar:
    symbol: str
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    raw_close: float | None = None


@dataclass(frozen=True)
class IntradayBar:
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass(frozen=True)
class Signal:
    symbol: str
    action: str
    score: float
    price: float
    stop_loss: float
    take_profit: float
    reason: str


@dataclass(frozen=True)
class OrderIntent:
    symbol: str
    side: str
    qty: int
    estimated_price: float
    notional: float
    reason: str


@dataclass(frozen=True)
class RiskDecision:
    allowed: bool
    reason: str
    order: OrderIntent | None = None


@dataclass
class Position:
    symbol: str
    qty: int
    avg_price: float


@dataclass(frozen=True)
class PortfolioSnapshot:
    cash: float
    equity: float
    positions: dict[str, Position]
    prices: dict[str, float]
