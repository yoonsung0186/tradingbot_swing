from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

from .models import Bar, IntradayBar


class DataError(RuntimeError):
    pass


class YahooChartClient:
    def __init__(
        self,
        timeout: int = 15,
        cache_dir: Path = Path("data/cache"),
        cache_ttl_hours: int = 12,
    ) -> None:
        self.timeout = timeout
        self.cache_dir = cache_dir
        self.cache_ttl_hours = cache_ttl_hours

    def history(
        self,
        symbol: str,
        start: date | None = None,
        end: date | None = None,
    ) -> list[Bar]:
        end = end or date.today()
        start = start or (end - timedelta(days=420))
        yahoo_symbol = self._symbol(symbol)
        period1 = self._timestamp(start)
        period2 = self._timestamp(end + timedelta(days=1))
        query = urllib.parse.urlencode(
            {
                "period1": period1,
                "period2": period2,
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": "true",
            }
        )
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_symbol)}?{query}"
        cache_path = self._cache_path(symbol, start, end)
        cached = self._read_cache(cache_path)
        if cached:
            bars = self._parse_chart(symbol.upper(), cached)
            if bars:
                return bars

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 us-ai-trading-agent/0.1",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise DataError(f"Could not fetch {symbol} from Yahoo Finance: {exc}") from exc

        self._write_cache(cache_path, text)
        bars = self._parse_chart(symbol.upper(), text)
        if not bars:
            raise DataError(f"No price data returned for {symbol}")
        return bars

    def latest_prices(self, symbols: list[str]) -> dict[str, float]:
        prices: dict[str, float] = {}
        for symbol in symbols:
            bars = self.history(symbol)
            prices[symbol.upper()] = bars[-1].close
        return prices

    def intraday_snapshot(self, symbol: str, interval: str = "5m", range_: str = "1d") -> dict:
        yahoo_symbol = self._symbol(symbol)
        query = urllib.parse.urlencode({"range": range_, "interval": interval, "includePrePost": "false"})
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_symbol)}?{query}"
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 us-ai-trading-agent/0.1",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            text = response.read().decode("utf-8")
        payload = json.loads(text)
        result = (payload.get("chart", {}).get("result") or [{}])[0]
        meta = result.get("meta", {})
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        closes = [item for item in quote.get("close", []) if item is not None]
        volumes = [item for item in quote.get("volume", []) if item is not None]
        price = float(meta.get("regularMarketPrice") or (closes[-1] if closes else 0))
        previous_close = float(meta.get("chartPreviousClose") or meta.get("previousClose") or 0)
        day_return = price / previous_close - 1 if previous_close else 0.0
        short_return = closes[-1] / closes[-6] - 1 if len(closes) >= 6 and closes[-6] else 0.0
        volume_sum = int(sum(volumes[-12:])) if volumes else 0
        return {
            "symbol": symbol.upper(),
            "price": price,
            "previous_close": previous_close,
            "day_return": day_return,
            "short_return": short_return,
            "recent_volume": volume_sum,
            "market_time": meta.get("regularMarketTime"),
            "source": "Yahoo Finance chart",
        }

    def intraday_history(self, symbol: str, interval: str = "5m", range_: str = "30d") -> list[IntradayBar]:
        yahoo_symbol = self._symbol(symbol)
        query = urllib.parse.urlencode(
            {
                "range": range_,
                "interval": interval,
                "includePrePost": "false",
            }
        )
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(yahoo_symbol)}?{query}"
        cache_path = self._intraday_cache_path(symbol, interval, range_)
        cached = self._read_cache(cache_path)
        if cached:
            bars = self._parse_intraday_chart(symbol.upper(), cached)
            if bars:
                return bars

        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 us-ai-trading-agent/0.1",
                "Accept": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise DataError(f"Could not fetch intraday {symbol} from Yahoo Finance: {exc}") from exc

        self._write_cache(cache_path, text)
        bars = self._parse_intraday_chart(symbol.upper(), text)
        if not bars:
            raise DataError(f"No intraday price data returned for {symbol}")
        return bars

    @staticmethod
    def _symbol(symbol: str) -> str:
        return symbol.upper().replace(".", "-")

    def _cache_path(self, symbol: str, start: date, end: date) -> Path:
        safe_symbol = symbol.upper().replace("^", "INDEX_").replace(".", "-")
        filename = f"{safe_symbol}_{start:%Y%m%d}_{end:%Y%m%d}.json"
        return self.cache_dir / filename

    def _intraday_cache_path(self, symbol: str, interval: str, range_: str) -> Path:
        safe_symbol = symbol.upper().replace("^", "INDEX_").replace(".", "-")
        safe_range = range_.replace("/", "-")
        safe_interval = interval.replace("/", "-")
        filename = f"{safe_symbol}_{safe_range}_{safe_interval}_intraday.json"
        return self.cache_dir / filename

    def _read_cache(self, path: Path) -> str | None:
        if not path.exists():
            return None
        age = datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        if age > timedelta(hours=self.cache_ttl_hours):
            return None
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def _write_cache(self, path: Path, text: str) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        except OSError:
            return

    @staticmethod
    def _timestamp(value: date) -> int:
        return int(datetime.combine(value, time.min, tzinfo=timezone.utc).timestamp())

    @staticmethod
    def _parse_chart(symbol: str, text: str) -> list[Bar]:
        payload = json.loads(text)
        chart = payload.get("chart", {})
        error = chart.get("error")
        if error:
            raise DataError(f"Yahoo Finance error for {symbol}: {error}")
        results = chart.get("result") or []
        if not results:
            return []
        result = results[0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adjclose = ((result.get("indicators") or {}).get("adjclose") or [{}])[0].get("adjclose") or []
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        bars: list[Bar] = []
        for idx, raw_timestamp in enumerate(timestamps):
            try:
                close = closes[idx]
                open_price = opens[idx]
                high = highs[idx]
                low = lows[idx]
            except IndexError:
                continue
            if None in {close, open_price, high, low}:
                continue
            adjusted_close = adjclose[idx] if idx < len(adjclose) and adjclose[idx] is not None else close
            ratio = adjusted_close / close if close else 1.0
            volume = volumes[idx] if idx < len(volumes) and volumes[idx] is not None else 0
            bars.append(
                Bar(
                    symbol=symbol,
                    date=datetime.fromtimestamp(raw_timestamp, tz=timezone.utc).date(),
                    open=float(open_price) * ratio,
                    high=float(high) * ratio,
                    low=float(low) * ratio,
                    close=float(adjusted_close),
                    volume=int(volume),
                    raw_close=float(close),
                )
            )
        return sorted(bars, key=lambda bar: bar.date)

    @staticmethod
    def _parse_intraday_chart(symbol: str, text: str) -> list[IntradayBar]:
        payload = json.loads(text)
        chart = payload.get("chart", {})
        error = chart.get("error")
        if error:
            raise DataError(f"Yahoo Finance intraday error for {symbol}: {error}")
        results = chart.get("result") or []
        if not results:
            return []
        result = results[0]
        timestamps = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        bars: list[IntradayBar] = []
        for idx, raw_timestamp in enumerate(timestamps):
            try:
                open_price = opens[idx]
                high = highs[idx]
                low = lows[idx]
                close = closes[idx]
            except IndexError:
                continue
            if None in {open_price, high, low, close}:
                continue
            volume = volumes[idx] if idx < len(volumes) and volumes[idx] is not None else 0
            bars.append(
                IntradayBar(
                    symbol=symbol,
                    timestamp=datetime.fromtimestamp(raw_timestamp, tz=timezone.utc),
                    open=float(open_price),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=int(volume),
                )
            )
        return sorted(bars, key=lambda bar: bar.timestamp)
