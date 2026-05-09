from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

from .config import load_dotenv


DEFAULT_FRED_SERIES = ("DGS10", "DGS2", "BAMLH0A0HYM2", "VIXCLS")


@dataclass(frozen=True)
class FredObservation:
    series_id: str
    observation_date: date
    realtime_start: date
    realtime_end: date
    value: float


@dataclass(frozen=True)
class MacroScore:
    symbol: str
    as_of: date
    allow_entry: bool
    risk_level: str
    risk_score: int
    positive_score: int
    score_adjustment: float
    position_multiplier: float
    reasons: tuple[str, ...]
    data_points: int


class FredMacroProvider:
    def __init__(
        self,
        start: date,
        end: date,
        series_ids: tuple[str, ...] = DEFAULT_FRED_SERIES,
        lookback_days: int = 30,
        block_risk_score: int = 6,
        caution_risk_score: int = 3,
        cache_dir: Path = Path("data/fred_cache"),
        timeout: int = 20,
    ) -> None:
        load_dotenv()
        self.start = start
        self.end = end
        self.series_ids = series_ids
        self.lookback_days = lookback_days
        self.block_risk_score = block_risk_score
        self.caution_risk_score = caution_risk_score
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.api_key = os.getenv("FRED_API_KEY", "")
        self.base_url = os.getenv("FRED_API_BASE_URL", "https://api.stlouisfed.org/fred").rstrip("/")
        self.observations: dict[str, list[FredObservation]] = {series_id: [] for series_id in series_ids}
        self.errors: list[str] = []
        self._score_cache: dict[tuple[str, date], MacroScore] = {}
        self._checked = 0
        self._blocked = 0
        self._caution = 0
        self._positive = 0
        self._load_series()

    def score(self, symbol: str, as_of: date) -> MacroScore:
        key = (symbol.upper(), as_of)
        if key in self._score_cache:
            return self._score_cache[key]

        risk_score = 0
        positive_score = 0
        reasons: list[str] = []
        data_points = 0

        ten_year = self._latest_value("DGS10", as_of)
        two_year = self._latest_value("DGS2", as_of)
        high_yield_spread = self._latest_value("BAMLH0A0HYM2", as_of)
        vix = self._latest_value("VIXCLS", as_of)
        prior_date = as_of - timedelta(days=self.lookback_days)
        ten_year_prior = self._latest_value("DGS10", as_of, target_date=prior_date)
        high_yield_prior = self._latest_value("BAMLH0A0HYM2", as_of, target_date=prior_date)
        vix_prior = self._latest_value("VIXCLS", as_of, target_date=prior_date)

        if ten_year is not None:
            data_points += 1
            if ten_year_prior is not None:
                ten_year_change = ten_year - ten_year_prior
                reasons.append(f"10y={ten_year:.2f}, 30d={ten_year_change:+.2f}p")
                if ten_year_change >= 0.45:
                    risk_score += 2
                elif ten_year_change >= 0.25:
                    risk_score += 1
                elif ten_year_change <= -0.25:
                    positive_score += 1
        if ten_year is not None and two_year is not None:
            data_points += 1
            curve = ten_year - two_year
            reasons.append(f"10y2y_curve={curve:+.2f}p")
            if curve <= -1.00:
                risk_score += 1
            elif curve >= 0.00:
                positive_score += 1

        if high_yield_spread is not None:
            data_points += 1
            if high_yield_prior is not None:
                spread_change = high_yield_spread - high_yield_prior
                reasons.append(f"hy_spread={high_yield_spread:.2f}, 30d={spread_change:+.2f}p")
                if high_yield_spread >= 6.0:
                    risk_score += 4
                elif high_yield_spread >= 5.0:
                    risk_score += 2
                elif high_yield_spread <= 3.5:
                    positive_score += 1
                if spread_change >= 0.75:
                    risk_score += 2
                elif spread_change >= 0.40:
                    risk_score += 1
                elif spread_change <= -0.40:
                    positive_score += 1

        if vix is not None:
            data_points += 1
            if vix_prior is not None:
                vix_change = vix - vix_prior
                reasons.append(f"vix={vix:.1f}, 30d={vix_change:+.1f}")
                if vix >= 35:
                    risk_score += 4
                elif vix >= 28:
                    risk_score += 2
                elif vix <= 18:
                    positive_score += 1
                if vix_change >= 8:
                    risk_score += 2
                elif vix_change >= 5:
                    risk_score += 1
                elif vix_change <= -5:
                    positive_score += 1

        if data_points == 0:
            result = MacroScore(
                symbol=symbol.upper(),
                as_of=as_of,
                allow_entry=True,
                risk_level="unavailable",
                risk_score=0,
                positive_score=0,
                score_adjustment=0.0,
                position_multiplier=1.0,
                reasons=("FRED macro data unavailable",),
                data_points=0,
            )
            self._score_cache[key] = result
            self._checked += 1
            return result

        if risk_score >= self.block_risk_score:
            risk_level = "blocked"
            allow_entry = False
            position_multiplier = 0.0
        elif risk_score >= self.caution_risk_score:
            risk_level = "caution"
            allow_entry = True
            position_multiplier = max(0.45, 0.78 - (risk_score - self.caution_risk_score) * 0.08)
        else:
            risk_level = "ok"
            allow_entry = True
            position_multiplier = min(1.12, max(0.85, 1.0 + positive_score * 0.04 - risk_score * 0.05))

        score_adjustment = max(min(positive_score * 0.025 - risk_score * 0.045, 0.15), -0.30)
        if not reasons:
            reasons.append("macro neutral")
        result = MacroScore(
            symbol=symbol.upper(),
            as_of=as_of,
            allow_entry=allow_entry,
            risk_level=risk_level,
            risk_score=risk_score,
            positive_score=positive_score,
            score_adjustment=score_adjustment,
            position_multiplier=position_multiplier,
            reasons=tuple(reasons[:6]),
            data_points=data_points,
        )
        self._score_cache[key] = result
        self._checked += 1
        if risk_level == "blocked":
            self._blocked += 1
        elif risk_level == "caution":
            self._caution += 1
        if positive_score > 0:
            self._positive += 1
        return result

    def summary(self) -> dict:
        return {
            "source": "fred",
            "series": list(self.series_ids),
            "checked_signals": self._checked,
            "blocked_signals": self._blocked,
            "caution_signals": self._caution,
            "positive_signals": self._positive,
            "errors": self.errors[:10],
            "lookback_days": self.lookback_days,
            "asof_rule": "observation_date <= signal_date and realtime_start <= signal_date",
            "observations": {series_id: len(items) for series_id, items in self.observations.items()},
        }

    def _latest_value(self, series_id: str, as_of: date, target_date: date | None = None) -> float | None:
        observation = _latest_realtime_value(self.observations.get(series_id, []), as_of, target_date=target_date)
        return observation.value if observation else None

    def _load_series(self) -> None:
        if not self.api_key:
            self.errors.append("FRED_API_KEY unavailable; macro filter is neutral")
            return
        observation_start = self.start - timedelta(days=max(260, self.lookback_days * 4))
        for series_id in self.series_ids:
            try:
                payload = self._series_json(series_id, observation_start, self.end)
                self.observations[series_id] = _parse_observations(series_id, payload)
            except Exception as exc:
                self.errors.append(f"{series_id}: FRED unavailable: {exc}")

    def _series_json(self, series_id: str, observation_start: date, observation_end: date) -> dict:
        cache_path = self.cache_dir / f"{series_id}_{observation_start.isoformat()}_{observation_end.isoformat()}.json"
        if cache_path.exists():
            return json.loads(cache_path.read_text(encoding="utf-8"))
        params = urllib.parse.urlencode(
            {
                "series_id": series_id,
                "api_key": self.api_key,
                "file_type": "json",
                "observation_start": observation_start.isoformat(),
                "observation_end": observation_end.isoformat(),
                "realtime_start": observation_start.isoformat(),
                "realtime_end": observation_end.isoformat(),
                "sort_order": "asc",
            }
        )
        url = f"{self.base_url}/series/observations?{params}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError:
            cached = self._nearest_cached_series(series_id, observation_start, observation_end)
            if cached:
                return json.loads(cached)
            raise
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(text, encoding="utf-8")
        return json.loads(text)

    def _nearest_cached_series(self, series_id: str, observation_start: date, observation_end: date) -> str | None:
        if not self.cache_dir.exists():
            return None
        prefix = f"{series_id}_{observation_start.isoformat()}_"
        candidates: list[tuple[date, Path]] = []
        for path in self.cache_dir.glob(f"{prefix}*.json"):
            raw_end = path.stem.removeprefix(prefix)
            try:
                cached_end = date.fromisoformat(raw_end)
            except ValueError:
                continue
            if cached_end <= observation_end:
                candidates.append((cached_end, path))
        if not candidates:
            return None
        _, path = max(candidates, key=lambda item: item[0])
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None


def _latest_realtime_value(
    observations: list[FredObservation],
    as_of: date,
    target_date: date | None = None,
) -> FredObservation | None:
    cutoff = target_date or as_of
    candidates = [
        item
        for item in observations
        if item.observation_date <= cutoff and item.realtime_start <= as_of
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda item: (item.observation_date, item.realtime_start))


def _parse_observations(series_id: str, payload: dict) -> list[FredObservation]:
    observations: list[FredObservation] = []
    for item in payload.get("observations", []):
        value = item.get("value")
        if value in {None, "."}:
            continue
        try:
            observations.append(
                FredObservation(
                    series_id=series_id,
                    observation_date=date.fromisoformat(item["date"]),
                    realtime_start=date.fromisoformat(item["realtime_start"]),
                    realtime_end=date.fromisoformat(item["realtime_end"]),
                    value=float(value),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    observations.sort(key=lambda item: (item.observation_date, item.realtime_start))
    return observations
