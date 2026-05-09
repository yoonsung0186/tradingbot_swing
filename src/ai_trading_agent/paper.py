from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .models import OrderIntent, PortfolioSnapshot, Position


class PaperPortfolio:
    def __init__(self, state_file: Path, starting_cash: float) -> None:
        self.state_file = state_file
        self.starting_cash = starting_cash
        self.cash = starting_cash
        self.positions: dict[str, Position] = {}
        self.trades: list[dict] = []
        self.load()

    def load(self) -> None:
        if not self.state_file.exists():
            return
        data = json.loads(self.state_file.read_text(encoding="utf-8"))
        self.cash = float(data.get("cash", self.starting_cash))
        self.positions = {
            symbol: Position(symbol=symbol, qty=int(raw["qty"]), avg_price=float(raw["avg_price"]))
            for symbol, raw in data.get("positions", {}).items()
        }
        self.trades = list(data.get("trades", []))

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cash": self.cash,
            "positions": {symbol: asdict(position) for symbol, position in self.positions.items()},
            "trades": self.trades,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def snapshot(self, prices: dict[str, float]) -> PortfolioSnapshot:
        equity = self.cash
        for symbol, position in self.positions.items():
            price = prices.get(symbol, position.avg_price)
            equity += position.qty * price
        return PortfolioSnapshot(
            cash=self.cash,
            equity=equity,
            positions=dict(self.positions),
            prices=prices,
        )

    def execute(self, order: OrderIntent) -> dict:
        if order.side.lower() != "buy":
            raise ValueError("local paper portfolio currently supports buy orders only")
        cost = order.qty * order.estimated_price
        if cost > self.cash:
            raise ValueError("not enough cash for paper order")
        position = self.positions.get(order.symbol)
        if position:
            total_qty = position.qty + order.qty
            avg_price = ((position.qty * position.avg_price) + cost) / total_qty
            self.positions[order.symbol] = Position(order.symbol, total_qty, avg_price)
        else:
            self.positions[order.symbol] = Position(order.symbol, order.qty, order.estimated_price)
        self.cash -= cost
        trade = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "price": order.estimated_price,
            "notional": cost,
            "reason": order.reason,
        }
        self.trades.append(trade)
        self.save()
        return trade
