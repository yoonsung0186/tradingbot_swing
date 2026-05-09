from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import load_dotenv


@dataclass(frozen=True)
class ExecutionOrder:
    symbol: str
    side: str
    qty: float
    order_type: str = "market"
    time_in_force: str = "day"
    limit_price: float | None = None
    reason: str = ""


@dataclass(frozen=True)
class ExecutionPlan:
    created_at: str
    broker: str
    mode: str
    orders: list[ExecutionOrder]
    notes: list[str]


class ExecutionError(RuntimeError):
    pass


class DryRunBroker:
    name = "dry-run"

    def submit_orders(self, orders: list[ExecutionOrder]) -> list[dict]:
        return [
            {
                "status": "dry_run",
                "symbol": order.symbol,
                "side": order.side,
                "qty": order.qty,
                "order_type": order.order_type,
                "reason": order.reason,
            }
            for order in orders
        ]


class AlpacaBroker:
    name = "alpaca"

    def __init__(self, paper: bool = True) -> None:
        load_dotenv()
        self.paper = paper
        self.key_id = os.getenv("ALPACA_KEY_ID", "")
        self.secret_key = os.getenv("ALPACA_SECRET_KEY", "")
        if not self.key_id or not self.secret_key:
            raise ExecutionError("Missing ALPACA_KEY_ID or ALPACA_SECRET_KEY")
        if paper:
            self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        else:
            if os.getenv("ALPACA_LIVE_TRADING_ENABLED", "").lower() != "true":
                raise ExecutionError("Live trading is locked. Set ALPACA_LIVE_TRADING_ENABLED=true to enable it.")
            self.base_url = os.getenv("ALPACA_LIVE_BASE_URL", "https://api.alpaca.markets").rstrip("/")

    def submit_orders(self, orders: list[ExecutionOrder]) -> list[dict]:
        return [self.submit_order(order) for order in orders]

    def submit_order(self, order: ExecutionOrder) -> dict:
        payload = {
            "symbol": order.symbol.upper(),
            "qty": f"{order.qty:.6f}".rstrip("0").rstrip("."),
            "side": order.side.lower(),
            "type": order.order_type,
            "time_in_force": order.time_in_force,
        }
        if order.limit_price is not None:
            payload["limit_price"] = str(order.limit_price)
        return self._request("POST", "/v2/orders", payload)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        body = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            method=method,
            headers={
                "APCA-API-KEY-ID": self.key_id,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "us-ai-trading-agent/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise ExecutionError(f"Alpaca HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise ExecutionError(f"Alpaca request failed: {exc}") from exc
        return json.loads(text) if text else {}


def build_execution_plan(
    orders: list[ExecutionOrder],
    broker: str = "alpaca",
    mode: str = "dry-run",
) -> ExecutionPlan:
    return ExecutionPlan(
        created_at=datetime.now(timezone.utc).isoformat(),
        broker=broker,
        mode=mode,
        orders=orders,
        notes=[
            "Dry-run is the default. Paper/live submission requires explicit CLI flags.",
            "Live trading should only be enabled after paper trading and broker-side risk controls are verified.",
        ],
    )


def save_execution_plan(plan: ExecutionPlan, output_dir: Path = Path("reports")) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"execution_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    payload = asdict(plan)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def format_execution_plan(plan: ExecutionPlan) -> str:
    rows = [
        f"broker={plan.broker} mode={plan.mode} created_at={plan.created_at}",
        "symbol side qty order_type tif reason",
        "------ ---- --- ---------- --- ------",
    ]
    for order in plan.orders:
        rows.append(
            f"{order.symbol:6} {order.side:4} {order.qty:.6f} "
            f"{order.order_type:10} {order.time_in_force:3} {order.reason}"
        )
    return "\n".join(rows)
