from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ETF_UNIVERSE = ["SPY", "QQQ", "DIA", "IWM", "TLT", "GLD", "SHY"]
SECTOR_ETF_UNIVERSE = [
    "XLK",
    "XLF",
    "XLV",
    "XLY",
    "XLP",
    "XLE",
    "XLU",
    "XLI",
    "XLB",
    "XLRE",
    "SMH",
    "IGV",
    "VNQ",
]
MEGA_CAP_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AVGO",
    "JPM",
    "UNH",
]
STOCK_UNIVERSE = [
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "META",
    "GOOGL",
    "TSLA",
    "AVGO",
    "AMD",
    "NFLX",
    "ORCL",
    "CRM",
    "ADBE",
    "INTC",
    "QCOM",
    "COST",
    "WMT",
    "HD",
    "NKE",
    "MCD",
    "JPM",
    "BAC",
    "GS",
    "MS",
    "V",
    "MA",
    "AXP",
    "LLY",
    "UNH",
    "JNJ",
    "ABBV",
    "MRK",
    "XOM",
    "CVX",
    "COP",
    "GE",
    "CAT",
    "BA",
]
LEVERAGED_ETF_UNIVERSE = [
    "TQQQ",
    "SOXL",
    "TECL",
    "UPRO",
    "SPXL",
    "FNGU",
    "BULZ",
    "TNA",
    "LABU",
]
DEFAULT_UNIVERSE = ETF_UNIVERSE + SECTOR_ETF_UNIVERSE + STOCK_UNIVERSE + LEVERAGED_ETF_UNIVERSE
DEFENSIVE_SYMBOLS = {"TLT", "GLD", "SHY"}
MARKET_CONTEXT_SYMBOLS = ["SPY", "QQQ", "DIA", "IWM", "TLT", "GLD", "SHY", "^VIX"]


@dataclass(frozen=True)
class StrategyConfig:
    short_window: int = 20
    long_window: int = 50
    volume_window: int = 20
    min_volume_ratio: float = 1.15
    near_high_pct: float = 0.98
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10
    max_candidates: int = 5


@dataclass(frozen=True)
class RiskConfig:
    risk_per_trade_pct: float = 0.005
    max_symbol_weight: float = 0.10
    max_order_value: float = 2500.0
    max_new_positions: int = 3
    daily_loss_stop_pct: float = 0.02


@dataclass(frozen=True)
class AppConfig:
    starting_cash: float = 10000.0
    state_file: Path = Path("data/paper_state.json")


def load_dotenv(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def app_config() -> AppConfig:
    load_dotenv()
    cash = float(os.getenv("TRADING_AGENT_STARTING_CASH", "10000"))
    state_file = Path(os.getenv("TRADING_AGENT_STATE_FILE", "data/paper_state.json"))
    return AppConfig(starting_cash=cash, state_file=state_file)


def resolve_universe(name: str, extra_symbols: list[str] | None = None) -> list[str]:
    normalized = name.lower()
    if normalized == "etf":
        symbols = ETF_UNIVERSE
    elif normalized in {"sector", "sectors"}:
        symbols = SECTOR_ETF_UNIVERSE
    elif normalized in {"mega", "megacap", "large"}:
        symbols = MEGA_CAP_UNIVERSE
    elif normalized in {"stock", "stocks", "equity", "equities"}:
        symbols = STOCK_UNIVERSE + LEVERAGED_ETF_UNIVERSE
    elif normalized in {"leveraged", "leveraged-etf", "leveraged_etf", "leverage"}:
        symbols = LEVERAGED_ETF_UNIVERSE
    elif normalized == "all":
        symbols = DEFAULT_UNIVERSE
    else:
        raise ValueError("universe must be one of: etf, sector, mega, stock, all")

    seen: set[str] = set()
    result: list[str] = []
    for symbol in [*symbols, *(extra_symbols or [])]:
        upper = symbol.upper()
        if upper not in seen:
            seen.add(upper)
            result.append(upper)
    return result
