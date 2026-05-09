from __future__ import annotations

from .config import RiskConfig
from .models import OrderIntent, PortfolioSnapshot, RiskDecision, Signal


class RiskManager:
    def __init__(self, config: RiskConfig | None = None) -> None:
        self.config = config or RiskConfig()

    def review_buy(self, signal: Signal, portfolio: PortfolioSnapshot) -> RiskDecision:
        if signal.action != "BUY":
            return RiskDecision(False, f"{signal.symbol}: signal action is {signal.action}")
        if signal.price <= 0:
            return RiskDecision(False, f"{signal.symbol}: invalid price")
        if portfolio.cash < signal.price:
            return RiskDecision(False, f"{signal.symbol}: not enough cash")

        existing_position = portfolio.positions.get(signal.symbol)
        if existing_position and existing_position.qty > 0:
            return RiskDecision(False, f"{signal.symbol}: already held")

        if len(portfolio.positions) >= self.config.max_new_positions:
            return RiskDecision(False, "max open positions reached")

        risk_budget = portfolio.equity * self.config.risk_per_trade_pct
        stop_distance = max(signal.price - signal.stop_loss, signal.price * 0.01)
        risk_sized_notional = risk_budget / stop_distance * signal.price
        max_weight_notional = portfolio.equity * self.config.max_symbol_weight
        notional = min(
            risk_sized_notional,
            max_weight_notional,
            self.config.max_order_value,
            portfolio.cash,
        )
        qty = int(notional // signal.price)
        if qty < 1:
            return RiskDecision(False, f"{signal.symbol}: position size rounds to zero")

        final_notional = qty * signal.price
        order = OrderIntent(
            symbol=signal.symbol,
            side="buy",
            qty=qty,
            estimated_price=signal.price,
            notional=final_notional,
            reason=signal.reason,
        )
        return RiskDecision(True, "approved", order)
