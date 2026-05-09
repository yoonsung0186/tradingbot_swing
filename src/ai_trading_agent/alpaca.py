from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode
import urllib.error
import urllib.request

from .config import load_dotenv
from .models import IntradayBar, OrderIntent


class AlpacaError(RuntimeError):
    pass


@dataclass(frozen=True)
class AlpacaNewsItem:
    id: int | str
    headline: str
    summary: str
    url: str
    source: str
    created_at: datetime
    updated_at: datetime
    symbols: tuple[str, ...]


def _env_any(*names: str) -> str:
    for name in names:
        value = os.getenv(name, "")
        if value:
            return value
    return ""


class AlpacaPaperClient:
    def __init__(self) -> None:
        load_dotenv()
        self.key_id = _env_any("ALPACA_KEY_ID", "APCA_API_KEY_ID")
        self.secret_key = _env_any("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY")
        self.base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
        if not self.key_id or not self.secret_key:
            raise AlpacaError("Missing ALPACA_KEY_ID or ALPACA_SECRET_KEY")

    def account(self) -> dict:
        return self._request("GET", "/v2/account")

    def clock(self) -> dict:
        payload = self._request("GET", "/v2/clock")
        return payload if isinstance(payload, dict) else {}

    def calendar(self, start: str, end: str) -> list[dict[str, Any]]:
        query = urlencode({"start": start, "end": end})
        payload = self._request("GET", f"/v2/calendar?{query}")
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            items = payload.get("calendar", [])
            if isinstance(items, list):
                return [item for item in items if isinstance(item, dict)]
        return []

    def submit_order(self, order: OrderIntent) -> dict:
        payload = {
            "symbol": order.symbol,
            "qty": str(order.qty),
            "side": order.side.lower(),
            "type": "market",
            "time_in_force": "day",
        }
        return self._request("POST", "/v2/orders", payload)

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict | list:
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
            raise AlpacaError(f"Alpaca HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AlpacaError(f"Alpaca request failed: {exc}") from exc
        return json.loads(text) if text else {}


class AlpacaMarketDataClient:
    def __init__(self, base_url: str | None = None) -> None:
        load_dotenv()
        self.key_id = _env_any("ALPACA_KEY_ID", "APCA_API_KEY_ID")
        self.secret_key = _env_any("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY")
        self.base_url = (base_url or os.getenv("ALPACA_DATA_BASE_URL", "https://data.alpaca.markets")).rstrip("/")
        if not self.key_id or not self.secret_key:
            raise AlpacaError("Missing ALPACA_KEY_ID or ALPACA_SECRET_KEY")

    def latest_quotes(self, symbols: list[str], feed: str = "iex") -> dict[str, dict[str, Any]]:
        cleaned_symbols = [symbol.upper() for symbol in symbols if symbol.strip()]
        if not cleaned_symbols:
            return {}
        query = urlencode({"symbols": ",".join(cleaned_symbols), "feed": feed})
        payload = self._request("GET", f"/v2/stocks/quotes/latest?{query}")
        quotes = payload.get("quotes", {})
        if not isinstance(quotes, dict):
            return {}
        return {symbol.upper(): quote for symbol, quote in quotes.items() if isinstance(quote, dict)}

    def latest_quote_ticks(self, symbols: list[str], feed: str = "iex") -> list[dict[str, Any]]:
        ticks: list[dict[str, Any]] = []
        for symbol, quote in self.latest_quotes(symbols, feed=feed).items():
            tick = quote_to_tick(symbol, quote, source=f"alpaca:{feed}")
            if tick:
                ticks.append(tick)
        return ticks

    def intraday_bars(
        self,
        symbols: list[str],
        feed: str = "iex",
        minutes: int = 90,
        timeframe: str = "1Min",
    ) -> dict[str, list[IntradayBar]]:
        cleaned_symbols = [symbol.upper() for symbol in symbols if symbol.strip()]
        if not cleaned_symbols:
            return {}
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=minutes)
        query = urlencode(
            {
                "symbols": ",".join(cleaned_symbols),
                "timeframe": timeframe,
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "feed": feed,
                "limit": 1000,
                "sort": "asc",
            }
        )
        payload = self._request("GET", f"/v2/stocks/bars?{query}")
        bars_by_symbol = payload.get("bars", {})
        if not isinstance(bars_by_symbol, dict):
            return {}
        result: dict[str, list[IntradayBar]] = {}
        for symbol, bars in bars_by_symbol.items():
            if not isinstance(bars, list):
                continue
            parsed: list[IntradayBar] = []
            for item in bars:
                if not isinstance(item, dict):
                    continue
                try:
                    parsed.append(
                        IntradayBar(
                            symbol=symbol.upper(),
                            timestamp=_parse_alpaca_timestamp(item.get("t")),
                            open=float(item["o"]),
                            high=float(item["h"]),
                            low=float(item["l"]),
                            close=float(item["c"]),
                            volume=int(item.get("v") or 0),
                        )
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            if parsed:
                result[symbol.upper()] = parsed
        return result

    def latest_news(
        self,
        symbols: list[str],
        lookback_minutes: int = 90,
        limit: int = 50,
        include_content: bool = False,
    ) -> list[AlpacaNewsItem]:
        cleaned_symbols = [symbol.upper() for symbol in symbols if symbol.strip()]
        if not cleaned_symbols:
            return []
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=max(lookback_minutes, 1))
        query = urlencode(
            {
                "symbols": ",".join(cleaned_symbols),
                "start": start.isoformat().replace("+00:00", "Z"),
                "end": end.isoformat().replace("+00:00", "Z"),
                "sort": "desc",
                "limit": max(1, min(limit, 50)),
                "include_content": str(include_content).lower(),
            }
        )
        payload = self._request("GET", f"/v1beta1/news?{query}")
        articles = payload.get("news", [])
        if not isinstance(articles, list):
            return []
        result: list[AlpacaNewsItem] = []
        for item in articles:
            if not isinstance(item, dict):
                continue
            headline = str(item.get("headline") or "").strip()
            if not headline:
                continue
            raw_symbols = item.get("symbols") or []
            if not isinstance(raw_symbols, list):
                raw_symbols = []
            result.append(
                AlpacaNewsItem(
                    id=item.get("id", ""),
                    headline=headline,
                    summary=str(item.get("summary") or "").strip(),
                    url=str(item.get("url") or "").strip(),
                    source=str(item.get("source") or "").strip(),
                    created_at=_parse_alpaca_timestamp(item.get("created_at")),
                    updated_at=_parse_alpaca_timestamp(item.get("updated_at")),
                    symbols=tuple(str(symbol).upper() for symbol in raw_symbols if str(symbol).strip()),
                )
            )
        return result

    def historical_news(
        self,
        symbols: list[str],
        start: datetime,
        end: datetime,
        limit: int = 50,
        include_content: bool = False,
        max_pages: int = 20,
    ) -> list[AlpacaNewsItem]:
        cleaned_symbols = [symbol.upper() for symbol in symbols if symbol.strip()]
        if not cleaned_symbols:
            return []
        result: list[AlpacaNewsItem] = []
        page_token = ""
        pages = 0
        while pages < max_pages:
            query_payload = {
                "symbols": ",".join(cleaned_symbols),
                "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "end": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                "sort": "asc",
                "limit": max(1, min(limit, 50)),
                "include_content": str(include_content).lower(),
            }
            if page_token:
                query_payload["page_token"] = page_token
            query = urlencode(query_payload)
            payload = self._request("GET", f"/v1beta1/news?{query}")
            articles = payload.get("news", [])
            if not isinstance(articles, list):
                break
            result.extend(_parse_news_items(articles))
            pages += 1
            page_token = str(payload.get("next_page_token") or "").strip()
            if not page_token:
                break
        return result

    def _request(self, method: str, path: str) -> dict:
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            method=method,
            headers={
                "APCA-API-KEY-ID": self.key_id,
                "APCA-API-SECRET-KEY": self.secret_key,
                "Accept": "application/json",
                "User-Agent": "us-ai-trading-agent/0.1",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=20) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise AlpacaError(f"Alpaca market data HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise AlpacaError(f"Alpaca market data request failed: {exc}") from exc
        return json.loads(text) if text else {}


def quote_to_tick(symbol: str, quote: dict[str, Any], source: str = "alpaca") -> dict[str, Any] | None:
    bid = _optional_float(quote.get("bp"))
    ask = _optional_float(quote.get("ap"))
    if bid and ask and bid > 0 and ask > 0:
        price = (bid + ask) / 2
    else:
        price = ask or bid
    if not price or price <= 0:
        return None

    timestamp = _parse_alpaca_timestamp(quote.get("t"))
    bid_size = _optional_int(quote.get("bs"))
    ask_size = _optional_int(quote.get("as"))
    volume = None
    if bid_size is not None or ask_size is not None:
        volume = (bid_size or 0) + (ask_size or 0)
    return {
        "timestamp": timestamp,
        "symbol": symbol.upper(),
        "price": float(price),
        "bid": bid,
        "ask": ask,
        "volume": volume,
        "source": source,
    }


def alpaca_credentials_available() -> bool:
    load_dotenv()
    return bool(_env_any("ALPACA_KEY_ID", "APCA_API_KEY_ID") and _env_any("ALPACA_SECRET_KEY", "APCA_API_SECRET_KEY"))


def _parse_news_items(articles: list) -> list[AlpacaNewsItem]:
    result: list[AlpacaNewsItem] = []
    for item in articles:
        if not isinstance(item, dict):
            continue
        headline = str(item.get("headline") or "").strip()
        if not headline:
            continue
        raw_symbols = item.get("symbols") or []
        if not isinstance(raw_symbols, list):
            raw_symbols = []
        result.append(
            AlpacaNewsItem(
                id=item.get("id", ""),
                headline=headline,
                summary=str(item.get("summary") or "").strip(),
                url=str(item.get("url") or "").strip(),
                source=str(item.get("source") or "").strip(),
                created_at=_parse_alpaca_timestamp(item.get("created_at")),
                updated_at=_parse_alpaca_timestamp(item.get("updated_at")),
                symbols=tuple(str(symbol).upper() for symbol in raw_symbols if str(symbol).strip()),
            )
        )
    return result


def _parse_alpaca_timestamp(value: Any) -> datetime:
    if isinstance(value, str):
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    return datetime.now(timezone.utc)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
