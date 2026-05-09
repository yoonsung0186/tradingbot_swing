from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import date
from statistics import mean, pstdev

from .metrics import performance_stats
from .models import Bar


@dataclass(frozen=True)
class SwingConfig:
    model_name: str = "swing_stable"
    ema_fast_window: int = 20
    ema_slow_window: int = 50
    exit_ema_window: int = 20
    bollinger_window: int = 20
    bollinger_std: float = 2.0
    atr_window: int = 14
    rsi_window: int = 14
    momentum_window: int = 20
    short_momentum_window: int = 5
    relative_strength_window: int = 20
    sector_relative_strength_window: int = 20
    breakout_window: int = 20
    near_high_window: int = 60
    volume_window: int = 20
    squeeze_lookback_window: int = 120
    market_filter_window: int = 50
    min_volume_ratio: float = 1.25
    min_dollar_volume: float = 20_000_000.0
    min_momentum: float = 0.02
    min_short_momentum: float = -1.0
    min_relative_strength: float = 0.0
    min_sector_relative_strength: float = -1.0
    min_near_high_pct: float = 0.0
    max_squeeze_rank: float = 1.0
    min_rsi: float = 45.0
    max_rsi: float = 74.0
    max_bollinger_extension: float = 0.04
    pullback_tolerance_pct: float = 0.025
    risk_per_trade_pct: float = 0.01
    max_position_pct: float = 0.18
    max_positions: int = 5
    max_new_entries_per_day: int = 3
    stop_atr_multiple: float = 1.6
    max_stop_pct: float = 0.07
    partial_take_profit_pct: float = 0.05
    partial_take_profit_fraction: float = 0.50
    breakeven_offset_pct: float = 0.002
    trailing_stop_pct: float = 0.03
    overheat_rsi_threshold: float = 86.0
    overheat_trailing_stop_pct: float = 0.0
    overheat_trailing_atr_multiple: float = 0.0
    trail_after_partial_only: bool = True
    max_hold_days: int = 15
    time_exit_mode: str = "always"
    time_exit_sideways_return_pct: float = 0.03
    min_extension_relative_strength: float = 0.0
    max_extended_hold_days: int = 60
    leveraged_max_extended_hold_days: int = 45
    min_extension_unrealized_return: float = 0.03
    min_extension_industry_score: float = 0.20
    early_weak_exit_days: int = 0
    early_weak_exit_max_return: float = 0.0
    early_weak_exit_max_relative_strength: float = 0.0
    early_weak_exit_max_recent_return: float = 1.0
    early_weak_exit_require_below_exit_ema: bool = False
    use_macd_filter: bool = False
    macd_fast_window: int = 12
    macd_slow_window: int = 26
    macd_signal_window: int = 9
    macd_min_histogram_pct: float = 0.0
    macd_require_histogram_rising: bool = False
    use_strong_market_sizing: bool = False
    strong_market_position_pct: float = 0.55
    strong_market_leveraged_position_pct: float = 0.55
    strong_market_risk_multiplier: float = 1.15
    strong_market_min_spy_momentum: float = 0.03
    strong_market_min_qqq_relative_strength: float = 0.01
    strong_market_vix_threshold: float = 22.0
    require_macro_ok_for_strong_sizing: bool = True
    use_four_stage_regime: bool = False
    regime_neutral_position_pct: float = 0.35
    regime_bull_position_pct: float = 0.45
    regime_strong_bull_position_pct: float = 0.55
    regime_neutral_risk_multiplier: float = 0.70
    regime_bull_risk_multiplier: float = 1.00
    regime_strong_bull_risk_multiplier: float = 1.12
    regime_risk_off_vix_jump: float = 3.0
    regime_leveraged_entry_min: str = ""
    regime_pyramid_entry_min: str = ""
    regime_risk_off_trailing_stop_pct: float = 0.0
    regime_use_ema200_filter: bool = False
    regime_long_trend_window: int = 200
    allow_pyramiding: bool = False
    max_pyramid_adds: int = 0
    pyramid_trigger_pct: float = 0.12
    pyramid_add_fraction: float = 0.35
    pyramid_max_position_pct: float = 0.55
    pyramid_min_relative_strength: float = 0.02
    pyramid_min_bars_between_adds: int = 5
    max_pyramid_adds_per_day: int = 2
    pyramid_min_recent_return_pct: float = 0.0
    pyramid_max_recent_return_pct: float = 0.18
    pyramid_vix_threshold: float = 22.0
    pyramid_max_bollinger_extension: float = 0.03
    pyramid_require_macro_ok: bool = True
    second_pyramid_trigger_pct: float = 0.0
    second_pyramid_add_fraction: float = 0.0
    second_pyramid_min_relative_strength: float = 0.0
    second_pyramid_min_recent_return_pct: float = 0.0
    second_pyramid_max_bollinger_extension: float = -1.0
    second_pyramid_max_position_pct: float = 0.0
    second_pyramid_leveraged_allowed: bool = False
    second_pyramid_after_partial_allowed: bool = False
    leveraged_allow_pyramiding: bool = True
    leveraged_fast_exit_days: int = 0
    leveraged_fast_exit_min_return: float = 0.0
    leveraged_fast_exit_min_relative_strength: float = 0.0
    leveraged_allowed_symbols: tuple[str, ...] = ()
    leveraged_blocked_symbols: tuple[str, ...] = ()
    leveraged_strict_symbols: tuple[str, ...] = ()
    leveraged_strict_underlying_min_momentum: float = 0.035
    leveraged_strict_underlying_min_short_momentum: float = 0.010
    leveraged_strict_underlying_min_relative_strength: float = 0.005
    leveraged_strict_underlying_min_volume_ratio: float = 1.10
    leveraged_strict_underlying_max_rsi: float = 86.0
    leveraged_strict_underlying_max_bollinger_extension: float = 0.16
    leveraged_strict_require_underlying_breakout: bool = True
    research_position_multiplier_cap: float = 1.25
    event_positive_score_bonus: float = 0.0
    event_industry_score_bonus: float = 0.0
    leveraged_underlying_min_short_momentum: float = -1.0
    leveraged_underlying_min_volume_ratio: float = 0.0
    leveraged_require_underlying_breakout: bool = False
    leveraged_underlying_max_bollinger_extension: float = 0.20
    cost_bps: float = 5.0
    slippage_bps: float = 15.0
    leveraged_max_position_pct: float = 0.16
    leveraged_risk_multiplier: float = 0.60
    leveraged_underlying_min_momentum: float = 0.03
    leveraged_underlying_min_relative_strength: float = 0.0
    leveraged_underlying_max_rsi: float = 84.0
    use_macro_filter: bool = True
    macro_filter_leveraged_only: bool = True
    use_market_filter: bool = True
    use_sector_filter: bool = False
    require_catalyst_breakout: bool = False
    market_symbol: str = "SPY"
    benchmark_symbol: str = "SPY"
    volatility_symbol: str = "^VIX"
    vix_threshold: float = 32.0
    tradable_symbols: tuple[str, ...] = ()


LEVERAGED_SYMBOLS = {"TQQQ", "SOXL", "TECL", "UPRO", "SPXL", "FNGU", "BULZ", "TNA", "LABU"}


SECTOR_PROXY_BY_SYMBOL = {
    "AAPL": "XLK",
    "MSFT": "XLK",
    "NVDA": "SMH",
    "AVGO": "SMH",
    "AMD": "SMH",
    "INTC": "SMH",
    "QCOM": "SMH",
    "GOOGL": "IGV",
    "META": "IGV",
    "NFLX": "IGV",
    "AMZN": "XLY",
    "TSLA": "XLY",
    "COST": "XLP",
    "WMT": "XLP",
    "HD": "XLY",
    "NKE": "XLY",
    "MCD": "XLP",
    "ORCL": "IGV",
    "CRM": "IGV",
    "ADBE": "IGV",
    "JPM": "XLF",
    "BAC": "XLF",
    "GS": "XLF",
    "MS": "XLF",
    "V": "XLF",
    "MA": "XLF",
    "AXP": "XLF",
    "LLY": "XLV",
    "UNH": "XLV",
    "JNJ": "XLV",
    "ABBV": "XLV",
    "MRK": "XLV",
    "XOM": "XLE",
    "CVX": "XLE",
    "COP": "XLE",
    "GE": "XLI",
    "CAT": "XLI",
    "BA": "XLI",
    "NOW": "IGV",
    "PANW": "IGV",
    "CRWD": "IGV",
    "PLTR": "IGV",
    "SNOW": "IGV",
    "ARM": "SMH",
    "MU": "SMH",
    "KLAC": "SMH",
    "LRCX": "SMH",
    "AMAT": "SMH",
    "ASML": "SMH",
    "TSM": "SMH",
    "MRVL": "SMH",
    "APP": "IGV",
    "UBER": "XLY",
    "ABNB": "XLY",
    "SHOP": "XLY",
    "COIN": "XLF",
    "MSTR": "IGV",
    "HOOD": "XLF",
    "SMCI": "SMH",
    "DELL": "XLK",
    "ANET": "XLK",
    "CEG": "XLU",
    "VRT": "XLI",
    "ETN": "XLI",
    "PWR": "XLI",
    "NVO": "XLV",
    "ISRG": "XLV",
    "REGN": "XLV",
    "TMO": "XLV",
    "LIN": "XLB",
    "LMT": "XLI",
    "RTX": "XLI",
    "HON": "XLI",
    "URI": "XLI",
    "DE": "XLI",
    "FSLR": "XLE",
    "ENPH": "XLE",
    "RCL": "XLY",
    "BKNG": "XLY",
    "MAR": "XLY",
    "TQQQ": "QQQ",
    "SOXL": "SMH",
    "TECL": "XLK",
    "UPRO": "SPY",
    "SPXL": "SPY",
    "FNGU": "QQQ",
    "BULZ": "QQQ",
    "TNA": "IWM",
    "LABU": "XLV",
}


@dataclass(frozen=True)
class SwingSignal:
    symbol: str
    score: float
    atr: float
    reason: str
    position_multiplier: float = 1.0
    research_risk_level: str = ""
    signal_date: date | None = None


@dataclass
class SwingPosition:
    symbol: str
    qty: float
    entry_date: date
    entry_price: float
    stop_price: float
    initial_stop_price: float
    highest_price: float
    partial_taken: bool
    overheat_armed: bool
    bars_held: int
    entry_reason: str
    add_count: int = 0
    last_add_bars_held: int = 0


def stable_swing_config(**overrides) -> SwingConfig:
    values = {
        "model_name": "swing_stable",
        "min_volume_ratio": 1.25,
        "min_momentum": 0.015,
        "min_relative_strength": 0.0,
        "max_position_pct": 0.18,
        "max_positions": 5,
        "risk_per_trade_pct": 0.009,
        "stop_atr_multiple": 1.6,
        "max_stop_pct": 0.07,
        "partial_take_profit_pct": 0.05,
        "partial_take_profit_fraction": 0.50,
        "trailing_stop_pct": 0.03,
        "max_hold_days": 15,
        "market_filter_window": 50,
        "max_rsi": 74.0,
    }
    values.update(overrides)
    return SwingConfig(**values)


def aggressive_swing_config(**overrides) -> SwingConfig:
    values = {
        "model_name": "swing_aggressive",
        "min_volume_ratio": 1.45,
        "min_momentum": 0.025,
        "min_relative_strength": -0.005,
        "max_position_pct": 0.32,
        "max_positions": 3,
        "risk_per_trade_pct": 0.018,
        "stop_atr_multiple": 2.2,
        "max_stop_pct": 0.10,
        "partial_take_profit_pct": 0.08,
        "partial_take_profit_fraction": 0.30,
        "trailing_stop_pct": 0.055,
        "max_hold_days": 22,
        "market_filter_window": 100,
        "max_rsi": 82.0,
        "max_bollinger_extension": 0.08,
    }
    values.update(overrides)
    return SwingConfig(**values)


def catalyst_momentum_swing_config(**overrides) -> SwingConfig:
    values = {
        "model_name": "catalyst_momentum_swing",
        "ema_fast_window": 20,
        "ema_slow_window": 50,
        "exit_ema_window": 20,
        "momentum_window": 20,
        "short_momentum_window": 5,
        "relative_strength_window": 20,
        "sector_relative_strength_window": 20,
        "breakout_window": 50,
        "near_high_window": 60,
        "squeeze_lookback_window": 120,
        "volume_window": 20,
        "min_volume_ratio": 1.70,
        "min_momentum": 0.045,
        "min_short_momentum": 0.018,
        "min_relative_strength": 0.010,
        "min_sector_relative_strength": -0.010,
        "min_near_high_pct": 0.96,
        "max_squeeze_rank": 0.45,
        "max_position_pct": 0.38,
        "max_positions": 3,
        "max_new_entries_per_day": 2,
        "risk_per_trade_pct": 0.018,
        "stop_atr_multiple": 2.5,
        "max_stop_pct": 0.12,
        "partial_take_profit_pct": 0.12,
        "partial_take_profit_fraction": 0.25,
        "trailing_stop_pct": 0.09,
        "overheat_trailing_stop_pct": 0.06,
        "trail_after_partial_only": False,
        "max_hold_days": 35,
        "market_filter_window": 50,
        "max_rsi": 88.0,
        "max_bollinger_extension": 0.14,
        "use_sector_filter": True,
        "require_catalyst_breakout": True,
    }
    values.update(overrides)
    return SwingConfig(**values)


def catalyst_rsi_exit_config(**overrides) -> SwingConfig:
    values = {
        **_catalyst_base_values(),
        "model_name": "catalyst_rsi_exit",
        "overheat_trailing_stop_pct": 0.0,
        "overheat_trailing_atr_multiple": 0.0,
    }
    values.update(overrides)
    return SwingConfig(**values)


def catalyst_pullback4_config(**overrides) -> SwingConfig:
    values = {
        **_catalyst_base_values(),
        "model_name": "catalyst_pullback4_exit",
        "overheat_trailing_stop_pct": 0.04,
        "overheat_trailing_atr_multiple": 0.0,
    }
    values.update(overrides)
    return SwingConfig(**values)


def catalyst_atr_exit_config(**overrides) -> SwingConfig:
    values = {
        **_catalyst_base_values(),
        "model_name": "catalyst_atr_exit",
        "overheat_trailing_stop_pct": 0.0,
        "overheat_trailing_atr_multiple": 2.0,
    }
    values.update(overrides)
    return SwingConfig(**values)


def catalyst_atr_strength_extend_config(**overrides) -> SwingConfig:
    values = {
        **_catalyst_base_values(),
        "model_name": "catalyst_atr_strength_extend",
        "overheat_trailing_stop_pct": 0.0,
        "overheat_trailing_atr_multiple": 2.0,
        "time_exit_mode": "strength_extend",
    }
    values.update(overrides)
    return SwingConfig(**values)


def catalyst_atr_weak_time_exit_config(**overrides) -> SwingConfig:
    values = {
        **_catalyst_base_values(),
        "model_name": "catalyst_atr_weak_time_exit",
        "overheat_trailing_stop_pct": 0.0,
        "overheat_trailing_atr_multiple": 2.0,
        "time_exit_mode": "weak_only",
        "time_exit_sideways_return_pct": 0.03,
    }
    values.update(overrides)
    return SwingConfig(**values)


def leveraged_overlay_aggressive_config(**overrides) -> SwingConfig:
    values = {
        **_catalyst_base_values(),
        "model_name": "leveraged_overlay_aggressive",
        "min_volume_ratio": 1.35,
        "min_momentum": 0.030,
        "min_short_momentum": 0.008,
        "min_relative_strength": -0.005,
        "min_sector_relative_strength": -0.015,
        "max_position_pct": 0.45,
        "max_positions": 2,
        "max_new_entries_per_day": 2,
        "risk_per_trade_pct": 0.026,
        "stop_atr_multiple": 3.0,
        "max_stop_pct": 0.16,
        "partial_take_profit_pct": 0.18,
        "partial_take_profit_fraction": 0.20,
        "trailing_stop_pct": 0.12,
        "overheat_trailing_stop_pct": 0.0,
        "overheat_trailing_atr_multiple": 2.8,
        "max_hold_days": 25,
        "max_rsi": 92.0,
        "max_bollinger_extension": 0.20,
        "leveraged_max_position_pct": 0.45,
        "leveraged_risk_multiplier": 1.20,
        "leveraged_underlying_min_momentum": 0.025,
        "leveraged_underlying_min_relative_strength": -0.005,
        "leveraged_underlying_max_rsi": 88.0,
    }
    values.update(overrides)
    return SwingConfig(**values)


def leveraged_overlay_improved_config(**overrides) -> SwingConfig:
    values = {
        **leveraged_overlay_aggressive_config().__dict__,
        "model_name": "leveraged_overlay_improved",
        "use_strong_market_sizing": True,
        "strong_market_position_pct": 0.55,
        "strong_market_leveraged_position_pct": 0.55,
        "strong_market_risk_multiplier": 1.12,
        "strong_market_min_spy_momentum": 0.03,
        "strong_market_min_qqq_relative_strength": 0.01,
        "strong_market_vix_threshold": 22.0,
        "allow_pyramiding": True,
        "max_pyramid_adds": 1,
        "pyramid_trigger_pct": 0.15,
        "pyramid_add_fraction": 0.35,
        "pyramid_max_position_pct": 0.55,
        "pyramid_min_relative_strength": 0.02,
        "pyramid_min_bars_between_adds": 5,
        "max_pyramid_adds_per_day": 2,
        "pyramid_min_recent_return_pct": 0.0,
        "pyramid_max_recent_return_pct": 0.18,
        "pyramid_vix_threshold": 22.0,
        "pyramid_max_bollinger_extension": 0.03,
        "pyramid_require_macro_ok": True,
        "time_exit_mode": "evidence_extend",
        "min_extension_unrealized_return": 0.08,
        "min_extension_relative_strength": 0.03,
        "max_extended_hold_days": 60,
        "leveraged_max_extended_hold_days": 45,
    }
    values.update(overrides)
    return SwingConfig(**values)


def leveraged_overlay_regime_4stage_config(**overrides) -> SwingConfig:
    values = {
        **leveraged_overlay_improved_config().__dict__,
        "model_name": "leveraged_overlay_regime_4stage",
        "use_strong_market_sizing": False,
        "use_four_stage_regime": True,
        "regime_neutral_position_pct": 0.40,
        "regime_bull_position_pct": 0.45,
        "regime_strong_bull_position_pct": 0.55,
        "regime_neutral_risk_multiplier": 0.85,
        "regime_bull_risk_multiplier": 1.00,
        "regime_strong_bull_risk_multiplier": 1.12,
        "regime_risk_off_vix_jump": 99.0,
        "regime_use_ema200_filter": True,
        "leveraged_blocked_symbols": ("BULZ",),
    }
    values.update(overrides)
    return SwingConfig(**values)


def _catalyst_base_values() -> dict:
    return {
        "ema_fast_window": 20,
        "ema_slow_window": 50,
        "exit_ema_window": 20,
        "momentum_window": 20,
        "short_momentum_window": 5,
        "relative_strength_window": 20,
        "sector_relative_strength_window": 20,
        "breakout_window": 50,
        "near_high_window": 60,
        "squeeze_lookback_window": 120,
        "volume_window": 20,
        "min_volume_ratio": 1.70,
        "min_momentum": 0.045,
        "min_short_momentum": 0.018,
        "min_relative_strength": 0.010,
        "min_sector_relative_strength": -0.010,
        "min_near_high_pct": 0.96,
        "max_squeeze_rank": 0.45,
        "max_position_pct": 0.38,
        "max_positions": 3,
        "max_new_entries_per_day": 2,
        "risk_per_trade_pct": 0.018,
        "stop_atr_multiple": 2.5,
        "max_stop_pct": 0.12,
        "partial_take_profit_pct": 0.12,
        "partial_take_profit_fraction": 0.25,
        "trailing_stop_pct": 0.09,
        "trail_after_partial_only": False,
        "max_hold_days": 35,
        "market_filter_window": 50,
        "max_rsi": 88.0,
        "max_bollinger_extension": 0.14,
        "use_sector_filter": True,
        "require_catalyst_breakout": True,
        "leveraged_max_position_pct": 0.38,
        "leveraged_risk_multiplier": 1.0,
    }


def default_swing_configs(**overrides) -> dict[str, SwingConfig]:
    return {
        "stable": stable_swing_config(**overrides),
        "aggressive": aggressive_swing_config(**overrides),
        "catalyst": catalyst_momentum_swing_config(**overrides),
        "catalyst_rsi": catalyst_rsi_exit_config(**overrides),
        "catalyst_pullback4": catalyst_pullback4_config(**overrides),
        "catalyst_atr": catalyst_atr_exit_config(**overrides),
        "catalyst_atr_strength_extend": catalyst_atr_strength_extend_config(**overrides),
        "catalyst_atr_weak_time": catalyst_atr_weak_time_exit_config(**overrides),
        "leveraged_overlay_aggressive": leveraged_overlay_aggressive_config(**overrides),
        "leveraged_overlay_improved": leveraged_overlay_improved_config(**overrides),
        "leveraged_overlay_regime_4stage": leveraged_overlay_regime_4stage_config(**overrides),
    }


def run_swing_backtest(
    histories: dict[str, list[Bar]],
    cash: float,
    start: date | None = None,
    end: date | None = None,
    config: SwingConfig | None = None,
    research_provider: object | None = None,
    macro_provider: object | None = None,
) -> dict:
    config = config or stable_swing_config()
    tradable_symbols = _tradable_symbols(histories, config)
    all_dates = sorted({bar.date for symbol in tradable_symbols for bar in histories.get(symbol, [])})
    if start:
        all_dates = [item for item in all_dates if item >= start]
    if end:
        all_dates = [item for item in all_dates if item <= end]
    if not all_dates:
        return _empty_result(cash, config)

    indexes = {symbol: -1 for symbol in histories}
    portfolio_cash = cash
    positions: dict[str, SwingPosition] = {}
    pending_entries: list[SwingSignal] = []
    pending_pyramids: list[SwingSignal] = []
    pending_exits: dict[str, str] = {}
    equity_curve: list[tuple[date, float]] = []
    trade_log: list[dict] = []
    closed_trades: list[dict] = []
    cost_rate = _cost_rate(config)

    for current_date in all_dates:
        _advance_indexes(histories, indexes, current_date)
        open_prices = _prices_for_indexes(histories, indexes, "open")
        close_prices = _prices_for_indexes(histories, indexes, "close")

        portfolio_cash = _execute_pending_exits(
            current_date,
            portfolio_cash,
            positions,
            histories,
            indexes,
            pending_exits,
            cost_rate,
            trade_log,
            closed_trades,
        )
        pending_exits = {}

        portfolio_cash = _execute_pending_pyramids(
            current_date,
            portfolio_cash,
            positions,
            histories,
            indexes,
            pending_pyramids,
            open_prices,
            close_prices,
            config,
            cost_rate,
            trade_log,
            macro_provider,
        )
        pending_pyramids = []

        portfolio_cash = _execute_pending_entries(
            current_date,
            portfolio_cash,
            positions,
            histories,
            indexes,
            pending_entries,
            open_prices,
            close_prices,
            config,
            cost_rate,
            trade_log,
            macro_provider,
        )
        pending_entries = []

        portfolio_cash = _manage_open_positions(
            current_date,
            portfolio_cash,
            positions,
            histories,
            indexes,
            config,
            cost_rate,
            trade_log,
            closed_trades,
        )

        equity = _portfolio_equity(portfolio_cash, positions, close_prices)
        equity_curve.append((current_date, equity))

        pending_exits = _close_based_exit_signals(
            current_date,
            positions,
            histories,
            indexes,
            config,
            research_provider,
            macro_provider,
        )
        pending_pyramids = _pyramid_signals(
            current_date,
            positions,
            histories,
            indexes,
            pending_exits,
            config,
            research_provider,
            macro_provider,
        )
        pending_entries = _entry_signals(
            current_date,
            histories,
            indexes,
            tradable_symbols,
            positions,
            pending_exits,
            config,
            research_provider,
            macro_provider,
        )

    if positions:
        final_date = all_dates[-1]
        _liquidate_at_end(
            final_date,
            portfolio_cash,
            positions,
            histories,
            indexes,
            cost_rate,
            trade_log,
            closed_trades,
        )
        portfolio_cash = trade_log[-1]["portfolio_cash"] if trade_log else portfolio_cash
        close_prices = _prices_for_indexes(histories, indexes, "close")
        if equity_curve:
            equity_curve[-1] = (final_date, _portfolio_equity(portfolio_cash, positions, close_prices))

    stats = performance_stats(equity_curve)
    wins = [trade for trade in closed_trades if float(trade["realized_pnl"]) > 0]
    losses = [trade for trade in closed_trades if float(trade["realized_pnl"]) < 0]
    gross_profit = sum(float(trade["realized_pnl"]) for trade in wins)
    gross_loss = abs(sum(float(trade["realized_pnl"]) for trade in losses))
    profit_factor = float("inf") if gross_profit > 0 and gross_loss == 0 else (gross_profit / gross_loss if gross_loss else 0.0)
    trade_returns = [float(trade["pnl_pct"]) for trade in closed_trades if trade.get("pnl_pct") != ""]

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
        "trades": len(closed_trades),
        "transaction_count": len(trade_log),
        "win_rate": len(wins) / len(closed_trades) if closed_trades else 0.0,
        "profit_factor": profit_factor,
        "average_trade_return": mean(trade_returns) if trade_returns else 0.0,
        "best_trade_return": max(trade_returns) if trade_returns else 0.0,
        "worst_trade_return": min(trade_returns) if trade_returns else 0.0,
        "trade_log": trade_log,
        "closed_trades": closed_trades,
        "open_positions": [
            {
                "symbol": position.symbol,
                "qty": position.qty,
                "entry_date": position.entry_date.isoformat(),
                "entry_price": position.entry_price,
                "stop_price": position.stop_price,
                "bars_held": position.bars_held,
            }
            for position in positions.values()
        ],
        "config": asdict(config),
        "research_summary": research_provider.summary() if research_provider and hasattr(research_provider, "summary") else {},
        "macro_summary": macro_provider.summary() if macro_provider and hasattr(macro_provider, "summary") else {},
        "equity_curve": equity_curve,
    }


def _empty_result(cash: float, config: SwingConfig) -> dict:
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
        "transaction_count": 0,
        "win_rate": 0.0,
        "profit_factor": 0.0,
        "average_trade_return": 0.0,
        "best_trade_return": 0.0,
        "worst_trade_return": 0.0,
        "trade_log": [],
        "closed_trades": [],
        "open_positions": [],
        "config": asdict(config),
        "research_summary": {},
        "macro_summary": {},
        "equity_curve": [],
    }


def _tradable_symbols(histories: dict[str, list[Bar]], config: SwingConfig) -> list[str]:
    if config.tradable_symbols:
        return [symbol for symbol in config.tradable_symbols if symbol in histories]
    excluded = {config.market_symbol, config.benchmark_symbol, config.volatility_symbol}
    return [symbol for symbol in histories if symbol not in excluded and not symbol.startswith("^")]


def _advance_indexes(histories: dict[str, list[Bar]], indexes: dict[str, int], current_date: date) -> None:
    for symbol, bars in histories.items():
        while indexes[symbol] + 1 < len(bars) and bars[indexes[symbol] + 1].date <= current_date:
            indexes[symbol] += 1


def _prices_for_indexes(histories: dict[str, list[Bar]], indexes: dict[str, int], field: str) -> dict[str, float]:
    prices: dict[str, float] = {}
    for symbol, idx in indexes.items():
        if idx >= 0:
            prices[symbol] = float(getattr(histories[symbol][idx], field))
    return prices


def _exact_bar(histories: dict[str, list[Bar]], indexes: dict[str, int], symbol: str, current_date: date) -> Bar | None:
    idx = indexes.get(symbol, -1)
    if idx < 0:
        return None
    bar = histories[symbol][idx]
    return bar if bar.date == current_date else None


def _portfolio_equity(cash: float, positions: dict[str, SwingPosition], prices: dict[str, float]) -> float:
    return cash + sum(position.qty * prices.get(symbol, 0.0) for symbol, position in positions.items())


def _position_cap_pct(
    symbol: str,
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: SwingConfig,
    current_date: date,
    macro_provider: object | None,
) -> float:
    cap = config.max_position_pct
    if symbol in LEVERAGED_SYMBOLS:
        cap = min(cap, config.leveraged_max_position_pct)
    asof_indexes = _indexes_as_of(histories, indexes, current_date)
    if config.use_four_stage_regime:
        regime = _market_regime(histories, asof_indexes, config, current_date, macro_provider)
        if regime == "risk_off":
            cap = 0.0
        elif regime == "neutral":
            cap = min(cap, config.regime_neutral_position_pct)
        elif regime == "bull":
            cap = min(max(cap, config.regime_bull_position_pct), config.regime_bull_position_pct)
        elif regime == "strong_bull":
            cap = max(cap, config.regime_strong_bull_position_pct)
    elif config.use_strong_market_sizing and _strong_market_context(histories, asof_indexes, config, current_date, macro_provider):
        if symbol in LEVERAGED_SYMBOLS:
            cap = max(cap, config.strong_market_leveraged_position_pct)
        else:
            cap = max(cap, config.strong_market_position_pct)
    return min(max(cap, 0.0), 1.0)


def _risk_pct(
    symbol: str,
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: SwingConfig,
    current_date: date,
    macro_provider: object | None,
) -> float:
    risk_pct = config.risk_per_trade_pct
    if symbol in LEVERAGED_SYMBOLS:
        risk_pct *= config.leveraged_risk_multiplier
    asof_indexes = _indexes_as_of(histories, indexes, current_date)
    if config.use_four_stage_regime:
        regime = _market_regime(histories, asof_indexes, config, current_date, macro_provider)
        if regime == "risk_off":
            risk_pct = 0.0
        elif regime == "neutral":
            risk_pct *= config.regime_neutral_risk_multiplier
        elif regime == "bull":
            risk_pct *= config.regime_bull_risk_multiplier
        elif regime == "strong_bull":
            risk_pct *= config.regime_strong_bull_risk_multiplier
    elif config.use_strong_market_sizing and _strong_market_context(histories, asof_indexes, config, current_date, macro_provider):
        risk_pct *= config.strong_market_risk_multiplier
    return max(risk_pct, 0.0)


def _indexes_as_of(histories: dict[str, list[Bar]], indexes: dict[str, int], as_of: date) -> dict[str, int]:
    asof_indexes: dict[str, int] = {}
    for symbol, bars in histories.items():
        idx = indexes.get(symbol, -1)
        while idx >= 0 and idx < len(bars) and bars[idx].date > as_of:
            idx -= 1
        asof_indexes[symbol] = idx
    return asof_indexes


def _market_regime(
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: SwingConfig,
    current_date: date,
    macro_provider: object | None,
) -> str:
    spy_bars = histories.get(config.market_symbol, [])
    spy_idx = indexes.get(config.market_symbol, -1)
    qqq_bars = histories.get("QQQ", [])
    qqq_idx = indexes.get("QQQ", -1)
    required = max(config.market_filter_window, config.relative_strength_window + 1)
    if config.regime_use_ema200_filter:
        required = max(required, config.regime_long_trend_window)
    if spy_idx < required or qqq_idx < required or spy_idx >= len(spy_bars) or qqq_idx >= len(qqq_bars):
        return "risk_off"

    spy_closes = [bar.close for bar in spy_bars[: spy_idx + 1]]
    qqq_closes = [bar.close for bar in qqq_bars[: qqq_idx + 1]]
    spy_above_trend = spy_closes[-1] > _ema(spy_closes, config.market_filter_window)
    qqq_above_trend = qqq_closes[-1] > _ema(qqq_closes, config.market_filter_window)
    spy_fast = _ema(spy_closes, config.ema_fast_window)
    qqq_fast = _ema(qqq_closes, config.ema_fast_window)
    spy_slow = _ema(spy_closes, config.ema_slow_window)
    qqq_slow = _ema(qqq_closes, config.ema_slow_window)
    spy_long = _ema(spy_closes, config.regime_long_trend_window)
    qqq_long = _ema(qqq_closes, config.regime_long_trend_window)

    vix_bars = histories.get(config.volatility_symbol, [])
    vix_idx = indexes.get(config.volatility_symbol, -1)
    vix_ok_for_strong = True
    if vix_bars and vix_idx >= 0 and vix_idx < len(vix_bars):
        current_vix = vix_bars[vix_idx].close
        if current_vix >= config.vix_threshold:
            return "risk_off"
        if current_vix >= config.strong_market_vix_threshold:
            vix_ok_for_strong = False
        if vix_idx >= 5 and current_vix - vix_bars[vix_idx - 5].close >= config.regime_risk_off_vix_jump:
            return "risk_off"

    if not spy_above_trend:
        return "risk_off"
    if config.regime_use_ema200_filter and spy_closes[-1] <= spy_long:
        return "risk_off"
    if not qqq_above_trend:
        return "neutral"
    if config.regime_use_ema200_filter and qqq_closes[-1] <= qqq_long:
        return "neutral"

    spy_return = _window_return(histories, indexes, config.market_symbol, config.relative_strength_window)
    qqq_return = _window_return(histories, indexes, "QQQ", config.relative_strength_window)
    macro_ok = True
    if config.require_macro_ok_for_strong_sizing and macro_provider and hasattr(macro_provider, "score"):
        try:
            macro = macro_provider.score(config.market_symbol, current_date)
        except Exception:
            macro_ok = False
        else:
            macro_ok = getattr(macro, "allow_entry", True) and int(getattr(macro, "risk_score", 0) or 0) == 0

    strong_bull = (
        spy_return >= config.strong_market_min_spy_momentum
        and qqq_return - spy_return >= config.strong_market_min_qqq_relative_strength
        and vix_ok_for_strong
        and macro_ok
    )
    if config.regime_use_ema200_filter:
        strong_bull = strong_bull and spy_closes[-1] > spy_fast > spy_slow > spy_long and qqq_closes[-1] > qqq_fast > qqq_slow > qqq_long
    return "strong_bull" if strong_bull else "bull"


def _regime_rank(regime: str) -> int:
    return {"risk_off": 0, "neutral": 1, "bull": 2, "strong_bull": 3}.get(regime, 0)


def _regime_meets(regime: str, minimum: str) -> bool:
    return not minimum or _regime_rank(regime) >= _regime_rank(minimum)


def _strong_market_context(
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: SwingConfig,
    current_date: date,
    macro_provider: object | None,
) -> bool:
    spy_bars = histories.get(config.market_symbol, [])
    spy_idx = indexes.get(config.market_symbol, -1)
    qqq_bars = histories.get("QQQ", [])
    qqq_idx = indexes.get("QQQ", -1)
    required = max(config.market_filter_window, config.relative_strength_window + 1)
    if spy_idx < required or qqq_idx < required or spy_idx >= len(spy_bars) or qqq_idx >= len(qqq_bars):
        return False

    spy_closes = [bar.close for bar in spy_bars[: spy_idx + 1]]
    qqq_closes = [bar.close for bar in qqq_bars[: qqq_idx + 1]]
    if not (spy_closes[-1] > _ema(spy_closes, config.market_filter_window)):
        return False
    if not (qqq_closes[-1] > _ema(qqq_closes, config.market_filter_window)):
        return False

    spy_return = _window_return(histories, indexes, config.market_symbol, config.relative_strength_window)
    qqq_return = _window_return(histories, indexes, "QQQ", config.relative_strength_window)
    if spy_return < config.strong_market_min_spy_momentum:
        return False
    if qqq_return - spy_return < config.strong_market_min_qqq_relative_strength:
        return False

    vix_bars = histories.get(config.volatility_symbol, [])
    vix_idx = indexes.get(config.volatility_symbol, -1)
    if vix_bars and vix_idx >= 5 and vix_idx < len(vix_bars):
        current_vix = vix_bars[vix_idx].close
        prior_vix = vix_bars[vix_idx - 5].close
        if current_vix >= config.strong_market_vix_threshold:
            return False
        if current_vix - prior_vix >= 3.0:
            return False

    if config.require_macro_ok_for_strong_sizing and macro_provider and hasattr(macro_provider, "score"):
        try:
            macro = macro_provider.score(config.market_symbol, current_date)
        except Exception:
            return False
        if not getattr(macro, "allow_entry", True):
            return False
        if int(getattr(macro, "risk_score", 0) or 0) > 0:
            return False
    return True


def _pyramid_add_allowed(
    symbol: str,
    position: SwingPosition,
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: SwingConfig,
) -> bool:
    if not config.allow_pyramiding or config.max_pyramid_adds <= 0:
        return False
    if position.add_count >= config.max_pyramid_adds:
        return False
    if position.partial_taken and not (config.second_pyramid_after_partial_allowed and position.add_count >= 1):
        return False
    if position.bars_held - position.last_add_bars_held < config.pyramid_min_bars_between_adds:
        return False

    bars = histories.get(symbol, [])
    idx = indexes.get(symbol, -1)
    required = max(config.exit_ema_window, config.ema_slow_window, config.relative_strength_window + 1)
    if idx < required or idx >= len(bars):
        return False
    history = bars[: idx + 1]
    closes = [bar.close for bar in history]
    close = closes[-1]
    unrealized_return = close / position.entry_price - 1 if position.entry_price else 0.0
    if unrealized_return < _pyramid_trigger_pct(position, config):
        return False
    exit_ema = _ema(closes, config.exit_ema_window)
    slow_ema = _ema(closes, config.ema_slow_window)
    if not (close >= exit_ema >= slow_ema):
        return False
    benchmark_return = _window_return(histories, indexes, config.benchmark_symbol, config.relative_strength_window)
    symbol_return = _window_return(histories, indexes, symbol, config.relative_strength_window)
    if symbol_return - benchmark_return < _pyramid_min_relative_strength(position, config):
        return False
    return True


def _pyramid_trigger_pct(position: SwingPosition, config: SwingConfig) -> float:
    if position.add_count >= 1 and config.second_pyramid_trigger_pct > 0:
        return config.second_pyramid_trigger_pct
    return config.pyramid_trigger_pct


def _pyramid_add_fraction(position: SwingPosition, config: SwingConfig) -> float:
    if position.add_count >= 1 and config.second_pyramid_add_fraction > 0:
        return config.second_pyramid_add_fraction
    return config.pyramid_add_fraction


def _pyramid_min_relative_strength(position: SwingPosition, config: SwingConfig) -> float:
    if position.add_count >= 1:
        return max(config.pyramid_min_relative_strength, config.second_pyramid_min_relative_strength)
    return config.pyramid_min_relative_strength


def _pyramid_min_recent_return(position: SwingPosition, config: SwingConfig) -> float:
    if position.add_count >= 1:
        return max(config.pyramid_min_recent_return_pct, config.second_pyramid_min_recent_return_pct)
    return config.pyramid_min_recent_return_pct


def _pyramid_max_bollinger_extension(position: SwingPosition, config: SwingConfig) -> float:
    if position.add_count >= 1 and config.second_pyramid_max_bollinger_extension >= 0:
        return config.second_pyramid_max_bollinger_extension
    return config.pyramid_max_bollinger_extension


def _pyramid_max_position_pct(position: SwingPosition, config: SwingConfig) -> float:
    if position.add_count >= 1 and config.second_pyramid_max_position_pct > 0:
        return config.second_pyramid_max_position_pct
    return config.pyramid_max_position_pct


def _pyramid_signals(
    current_date: date,
    positions: dict[str, SwingPosition],
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    pending_exits: dict[str, str],
    config: SwingConfig,
    research_provider: object | None,
    macro_provider: object | None,
) -> list[SwingSignal]:
    if not config.allow_pyramiding or config.max_pyramid_adds <= 0:
        return []
    if config.use_four_stage_regime and config.regime_pyramid_entry_min:
        regime = _market_regime(histories, indexes, config, current_date, macro_provider)
        if not _regime_meets(regime, config.regime_pyramid_entry_min):
            return []
    signals: list[SwingSignal] = []
    benchmark_return = _window_return(histories, indexes, config.benchmark_symbol, config.relative_strength_window)
    for symbol, position in positions.items():
        if symbol in pending_exits:
            continue
        if symbol in LEVERAGED_SYMBOLS and not config.leveraged_allow_pyramiding:
            continue
        if symbol in LEVERAGED_SYMBOLS and position.add_count >= 1 and not config.second_pyramid_leveraged_allowed:
            continue
        signal = _pyramid_signal_for_position(
            current_date,
            symbol,
            position,
            histories,
            indexes,
            benchmark_return,
            config,
            research_provider,
            macro_provider,
        )
        if signal:
            signals.append(signal)
    signals.sort(key=lambda item: item.score, reverse=True)
    return signals[: config.max_pyramid_adds_per_day]


def _pyramid_signal_for_position(
    current_date: date,
    symbol: str,
    position: SwingPosition,
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    benchmark_return: float,
    config: SwingConfig,
    research_provider: object | None,
    macro_provider: object | None,
) -> SwingSignal | None:
    if not _pyramid_add_allowed(symbol, position, histories, indexes, config):
        return None
    bars = histories.get(symbol, [])
    idx = indexes.get(symbol, -1)
    required = max(
        config.atr_window,
        config.bollinger_window,
        config.exit_ema_window,
        config.ema_slow_window,
        config.relative_strength_window + 1,
        config.short_momentum_window + 1,
    )
    if idx < required or idx >= len(bars):
        return None
    history = bars[: idx + 1]
    closes = [bar.close for bar in history]
    close = closes[-1]
    if close <= 0:
        return None

    recent_return = close / closes[-config.short_momentum_window - 1] - 1
    if recent_return < _pyramid_min_recent_return(position, config):
        return None
    if recent_return > config.pyramid_max_recent_return_pct:
        return None

    middle, upper, lower = _bollinger(closes, config.bollinger_window, config.bollinger_std)
    max_bollinger_extension = _pyramid_max_bollinger_extension(position, config)
    if upper > 0 and close > upper * (1 + max_bollinger_extension):
        return None

    vix_bars = histories.get(config.volatility_symbol, [])
    vix_idx = indexes.get(config.volatility_symbol, -1)
    if vix_bars and 0 <= vix_idx < len(vix_bars) and vix_bars[vix_idx].close >= config.pyramid_vix_threshold:
        return None

    if config.pyramid_require_macro_ok and macro_provider and hasattr(macro_provider, "score"):
        try:
            macro = macro_provider.score(config.market_symbol, current_date)
        except Exception:
            return None
        if not getattr(macro, "allow_entry", True):
            return None
        if int(getattr(macro, "risk_score", 0) or 0) > 0:
            return None

    research_multiplier = 1.0
    research_reason = "research clear"
    if research_provider and hasattr(research_provider, "score"):
        research = research_provider.score(symbol, current_date)
        if not research.allow_entry or getattr(research, "risk_level", "") == "blocked":
            return None
        research_multiplier = research.position_multiplier
        research_reason = "; ".join(research.reasons[:2]) or research_reason

    atr = _atr(history, config.atr_window)
    if atr <= 0:
        return None
    symbol_return = close / closes[-config.relative_strength_window - 1] - 1
    relative_strength = symbol_return - benchmark_return
    unrealized_return = close / position.entry_price - 1 if position.entry_price else 0.0
    exit_ema = _ema(closes, config.exit_ema_window)
    band_position = (close - lower) / (upper - lower) if upper > lower else 0.5
    score = (
        unrealized_return * 2.0
        + relative_strength * 2.2
        + recent_return * 1.2
        + max(close / exit_ema - 1, 0.0) * 1.0
        + min(max(band_position, 0.0), 1.2) * 0.03
    )
    reason = (
        f"independent_pyramid; unrealized={unrealized_return:.2%}; "
        f"recent={recent_return:.2%}; rs={relative_strength:.2%}; "
        f"adds={position.add_count}/{config.max_pyramid_adds}; {research_reason}"
    )
    return SwingSignal(
        symbol=symbol,
        score=score,
        atr=atr,
        reason=reason,
        position_multiplier=research_multiplier,
        research_risk_level="",
        signal_date=current_date,
    )


def _execute_pending_exits(
    current_date: date,
    cash: float,
    positions: dict[str, SwingPosition],
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    pending_exits: dict[str, str],
    cost_rate: float,
    trade_log: list[dict],
    closed_trades: list[dict],
) -> float:
    for symbol, reason in list(pending_exits.items()):
        if symbol not in positions:
            continue
        bar = _exact_bar(histories, indexes, symbol, current_date)
        if not bar:
            continue
        cash = _sell_position(
            current_date,
            cash,
            positions,
            symbol,
            bar.open,
            1.0,
            reason,
            cost_rate,
            trade_log,
            closed_trades,
        )
    return cash


def _execute_pending_pyramids(
    current_date: date,
    cash: float,
    positions: dict[str, SwingPosition],
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    pending_pyramids: list[SwingSignal],
    open_prices: dict[str, float],
    close_prices: dict[str, float],
    config: SwingConfig,
    cost_rate: float,
    trade_log: list[dict],
    macro_provider: object | None = None,
    signal_date: date | None = None,
) -> float:
    if not pending_pyramids:
        return cash

    equity = _portfolio_equity(cash, positions, open_prices or close_prices)
    for signal in sorted(pending_pyramids, key=lambda item: item.score, reverse=True)[: config.max_pyramid_adds_per_day]:
        position = positions.get(signal.symbol)
        if position is None:
            continue
        if position.add_count >= config.max_pyramid_adds:
            continue
        if position.partial_taken and not (config.second_pyramid_after_partial_allowed and position.add_count >= 1):
            continue
        if position.bars_held - position.last_add_bars_held < config.pyramid_min_bars_between_adds:
            continue
        bar = _exact_bar(histories, indexes, signal.symbol, current_date)
        if not bar or bar.open <= 0:
            continue

        entry_price = bar.open * (1 + cost_rate)
        if entry_price <= position.stop_price:
            continue
        risk_per_share = entry_price - position.stop_price
        if risk_per_share <= 0:
            continue

        research_multiplier = min(max(signal.position_multiplier, 0.0), config.research_position_multiplier_cap)
        sizing_date = signal.signal_date or signal_date or current_date
        position_cap_pct = _position_cap_pct(
            signal.symbol,
            histories,
            indexes,
            config,
            sizing_date,
            macro_provider,
        )
        risk_pct = _risk_pct(signal.symbol, histories, indexes, config, sizing_date, macro_provider)
        pyramid_cap_pct = min(max(position_cap_pct, config.pyramid_max_position_pct), _pyramid_max_position_pct(position, config))
        current_value = position.qty * entry_price
        remaining_cap_value = max(equity * pyramid_cap_pct * research_multiplier - current_value, 0.0)
        max_value_qty = remaining_cap_value / entry_price
        add_fraction = _pyramid_add_fraction(position, config)
        risk_qty = (equity * risk_pct * add_fraction * research_multiplier) / risk_per_share
        existing_fraction_qty = position.qty * add_fraction
        cash_qty = cash / entry_price
        qty = min(max_value_qty, risk_qty, existing_fraction_qty, cash_qty)
        if qty <= 0 or qty * entry_price < 50:
            continue

        previous_qty = position.qty
        cash -= qty * entry_price
        position.qty += qty
        position.entry_price = ((position.entry_price * previous_qty) + (entry_price * qty)) / position.qty
        position.highest_price = max(position.highest_price, bar.high, entry_price)
        position.stop_price = max(position.stop_price, position.entry_price * (1 - config.max_stop_pct))
        position.add_count += 1
        position.last_add_bars_held = position.bars_held
        trade_log.append(
            _transaction_record(
                current_date,
                signal.symbol,
                "PYRAMID_BUY",
                qty,
                entry_price,
                qty * entry_price,
                "",
                "",
                signal.reason,
                cash,
            )
        )
    return cash


def _execute_pending_entries(
    current_date: date,
    cash: float,
    positions: dict[str, SwingPosition],
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    pending_entries: list[SwingSignal],
    open_prices: dict[str, float],
    close_prices: dict[str, float],
    config: SwingConfig,
    cost_rate: float,
    trade_log: list[dict],
    macro_provider: object | None = None,
    signal_date: date | None = None,
) -> float:
    if not pending_entries:
        return cash

    slots = max(config.max_positions - len(positions), 0)
    if slots <= 0:
        return cash

    equity = _portfolio_equity(cash, positions, open_prices or close_prices)
    entries = sorted(pending_entries, key=lambda item: item.score, reverse=True)[: config.max_new_entries_per_day]
    for signal in entries:
        if slots <= 0 or signal.symbol in positions:
            continue
        bar = _exact_bar(histories, indexes, signal.symbol, current_date)
        if not bar or bar.open <= 0:
            continue
        entry_price = bar.open * (1 + cost_rate)
        stop_price = max(
            entry_price - config.stop_atr_multiple * signal.atr,
            entry_price * (1 - config.max_stop_pct),
        )
        risk_per_share = max(entry_price - stop_price, 0.0)
        if risk_per_share <= 0:
            continue

        research_multiplier = min(max(signal.position_multiplier, 0.0), config.research_position_multiplier_cap)
        sizing_date = signal.signal_date or signal_date or current_date
        position_cap_pct = _position_cap_pct(
            signal.symbol,
            histories,
            indexes,
            config,
            sizing_date,
            macro_provider,
        )
        risk_pct = _risk_pct(signal.symbol, histories, indexes, config, sizing_date, macro_provider)

        max_value_qty = (equity * position_cap_pct * research_multiplier) / entry_price
        risk_qty = (equity * risk_pct * research_multiplier) / risk_per_share
        cash_qty = cash / entry_price
        qty = min(max_value_qty, risk_qty, cash_qty)
        if qty <= 0 or qty * entry_price < 50:
            continue

        cash -= qty * entry_price
        positions[signal.symbol] = SwingPosition(
            symbol=signal.symbol,
            qty=qty,
            entry_date=current_date,
            entry_price=entry_price,
            stop_price=stop_price,
            initial_stop_price=stop_price,
            highest_price=max(bar.high, entry_price),
            partial_taken=False,
            overheat_armed=False,
            bars_held=0,
            entry_reason=signal.reason,
        )
        trade_log.append(
            _transaction_record(
                current_date,
                signal.symbol,
                "BUY",
                qty,
                entry_price,
                qty * entry_price,
                "",
                "",
                signal.reason,
                cash,
            )
        )
        slots -= 1
    return cash


def _manage_open_positions(
    current_date: date,
    cash: float,
    positions: dict[str, SwingPosition],
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: SwingConfig,
    cost_rate: float,
    trade_log: list[dict],
    closed_trades: list[dict],
) -> float:
    for symbol in list(positions):
        position = positions.get(symbol)
        if position is None:
            continue
        bar = _exact_bar(histories, indexes, symbol, current_date)
        if not bar:
            continue

        active_stop = position.stop_price
        if bar.open <= active_stop:
            reason = _stop_reason(position, config)
            cash = _sell_position(current_date, cash, positions, symbol, bar.open, 1.0, reason, cost_rate, trade_log, closed_trades)
            continue
        if bar.low <= active_stop:
            reason = _stop_reason(position, config)
            cash = _sell_position(current_date, cash, positions, symbol, active_stop, 1.0, reason, cost_rate, trade_log, closed_trades)
            continue

        position = positions.get(symbol)
        if position is None:
            continue
        partial_target = position.entry_price * (1 + config.partial_take_profit_pct)
        if not position.partial_taken and bar.high >= partial_target:
            fill_price = max(partial_target, bar.open)
            cash = _sell_position(
                current_date,
                cash,
                positions,
                symbol,
                fill_price,
                config.partial_take_profit_fraction,
                "partial_take_profit",
                cost_rate,
                trade_log,
                closed_trades,
            )
            position = positions.get(symbol)
            if position is not None:
                position.partial_taken = True
                position.stop_price = max(position.stop_price, position.entry_price * (1 + config.breakeven_offset_pct))

        position = positions.get(symbol)
        if position is None:
            continue
        position.highest_price = max(position.highest_price, bar.high)
        if position.partial_taken or not config.trail_after_partial_only:
            trailing_stop = position.highest_price * (1 - config.trailing_stop_pct)
            position.stop_price = max(position.stop_price, trailing_stop)
        if position.partial_taken and (
            config.overheat_trailing_stop_pct > 0 or config.overheat_trailing_atr_multiple > 0
        ):
            bars = histories.get(symbol, [])
            idx = indexes.get(symbol, -1)
            if idx >= config.rsi_window and idx < len(bars):
                closes = [item.close for item in bars[: idx + 1]]
                if _rsi(closes, config.rsi_window) >= config.overheat_rsi_threshold:
                    position.overheat_armed = True
            if position.overheat_armed:
                overheat_stops: list[float] = []
                if config.overheat_trailing_stop_pct > 0:
                    overheat_stops.append(position.highest_price * (1 - config.overheat_trailing_stop_pct))
                if config.overheat_trailing_atr_multiple > 0 and idx >= config.atr_window and idx < len(bars):
                    atr = _atr(bars[: idx + 1], config.atr_window)
                    if atr > 0:
                        overheat_stops.append(position.highest_price - config.overheat_trailing_atr_multiple * atr)
                if overheat_stops:
                    position.stop_price = max(position.stop_price, max(overheat_stops))
        position.bars_held += 1
    return cash


def _stop_reason(position: SwingPosition, config: SwingConfig) -> str:
    if position.overheat_armed:
        if config.overheat_trailing_atr_multiple > 0:
            return "overheat_atr_exit"
        return "overheat_pullback_exit"
    return "trailing_stop" if position.stop_price > position.initial_stop_price else "stop_loss"


def _close_based_exit_signals(
    current_date: date,
    positions: dict[str, SwingPosition],
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: SwingConfig,
    research_provider: object | None = None,
    macro_provider: object | None = None,
) -> dict[str, str]:
    exits: dict[str, str] = {}
    benchmark_return = _window_return(histories, indexes, config.benchmark_symbol, config.relative_strength_window)
    regime = ""
    if config.use_four_stage_regime and config.regime_risk_off_trailing_stop_pct > 0:
        regime = _market_regime(histories, indexes, config, current_date, macro_provider)
    for symbol, position in positions.items():
        idx = indexes.get(symbol, -1)
        bars = histories.get(symbol, [])
        if idx < max(config.exit_ema_window, config.ema_slow_window, config.rsi_window) or idx >= len(bars):
            continue
        closes = [bar.close for bar in bars[: idx + 1]]
        close = closes[-1]
        exit_ema = _ema(closes, config.exit_ema_window)
        slow_ema = _ema(closes, config.ema_slow_window)
        rsi_value = _rsi(closes, config.rsi_window)
        symbol_return = _window_return(histories, indexes, symbol, config.relative_strength_window)
        relative_strength = symbol_return - benchmark_return
        unrealized_return = close / position.entry_price - 1 if position.entry_price else 0.0
        if (
            symbol in LEVERAGED_SYMBOLS
            and config.leveraged_fast_exit_days > 0
            and position.bars_held >= config.leveraged_fast_exit_days
            and unrealized_return < config.leveraged_fast_exit_min_return
            and relative_strength < config.leveraged_fast_exit_min_relative_strength
        ):
            exits[symbol] = "leveraged_fast_exit"
        elif _should_early_weak_exit(
            symbol,
            position,
            close,
            exit_ema,
            relative_strength,
            unrealized_return,
            histories,
            indexes,
            config,
        ):
            exits[symbol] = "early_weak_exit"
        elif _should_time_exit(
            current_date,
            symbol,
            position,
            close,
            exit_ema,
            slow_ema,
            relative_strength,
            unrealized_return,
            histories,
            indexes,
            config,
            research_provider,
            macro_provider,
        ):
            exits[symbol] = "time_exit"
        elif (
            regime == "risk_off"
            and position.highest_price > 0
            and close <= position.highest_price * (1 - config.regime_risk_off_trailing_stop_pct)
        ):
            exits[symbol] = "regime_risk_off_trailing_exit"
        elif close < exit_ema:
            exits[symbol] = "ema_exit"
        elif (
            position.partial_taken
            and config.overheat_trailing_stop_pct <= 0
            and config.overheat_trailing_atr_multiple <= 0
            and rsi_value >= config.overheat_rsi_threshold
        ):
            exits[symbol] = "overheat_exit"
    return exits


def _should_early_weak_exit(
    symbol: str,
    position: SwingPosition,
    close: float,
    exit_ema: float,
    relative_strength: float,
    unrealized_return: float,
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: SwingConfig,
) -> bool:
    if config.early_weak_exit_days <= 0:
        return False
    if position.bars_held < config.early_weak_exit_days:
        return False
    if unrealized_return >= config.early_weak_exit_max_return:
        return False
    if relative_strength >= config.early_weak_exit_max_relative_strength:
        return False
    if config.early_weak_exit_require_below_exit_ema and close >= exit_ema:
        return False
    bars = histories.get(symbol, [])
    idx = indexes.get(symbol, -1)
    if idx >= config.short_momentum_window and idx < len(bars):
        prior_close = bars[idx - config.short_momentum_window].close
        recent_return = close / prior_close - 1 if prior_close > 0 else 0.0
        if recent_return > config.early_weak_exit_max_recent_return:
            return False
    return True


def _should_time_exit(
    current_date: date,
    symbol: str,
    position: SwingPosition,
    close: float,
    exit_ema: float,
    slow_ema: float,
    relative_strength: float,
    unrealized_return: float,
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: SwingConfig,
    research_provider: object | None,
    macro_provider: object | None,
) -> bool:
    if position.bars_held < config.max_hold_days:
        return False
    if config.time_exit_mode == "evidence_extend":
        return not _evidence_extension_ok(
            current_date,
            symbol,
            position,
            close,
            exit_ema,
            slow_ema,
            relative_strength,
            unrealized_return,
            histories,
            indexes,
            config,
            research_provider,
            macro_provider,
        )
    if config.time_exit_mode == "strength_extend":
        trend_intact = close >= exit_ema
        relative_intact = relative_strength >= config.min_extension_relative_strength
        return not (trend_intact and relative_intact)
    if config.time_exit_mode == "weak_only":
        return unrealized_return <= config.time_exit_sideways_return_pct
    return True


def _evidence_extension_ok(
    current_date: date,
    symbol: str,
    position: SwingPosition,
    close: float,
    exit_ema: float,
    slow_ema: float,
    relative_strength: float,
    unrealized_return: float,
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    config: SwingConfig,
    research_provider: object | None,
    macro_provider: object | None,
) -> bool:
    is_leveraged = symbol in LEVERAGED_SYMBOLS
    max_hold = config.leveraged_max_extended_hold_days if is_leveraged else config.max_extended_hold_days
    if position.bars_held >= max_hold:
        return False
    if unrealized_return < config.min_extension_unrealized_return:
        return False
    if not (close >= exit_ema >= slow_ema):
        return False
    if relative_strength < config.min_extension_relative_strength:
        return False

    research = _provider_score(research_provider, symbol, current_date)
    if research is not None:
        if not getattr(research, "allow_entry", True) or getattr(research, "risk_level", "") == "blocked":
            return False
        if not is_leveraged:
            positive_score = int(getattr(research, "positive_score", 0) or 0)
            industry_score = float(getattr(research, "industry_score", 0.0) or 0.0)
            if positive_score <= 0 and industry_score < config.min_extension_industry_score:
                return False
    elif not is_leveraged:
        return False

    if is_leveraged:
        macro = _provider_score(macro_provider, symbol, current_date)
        if macro is not None:
            if not getattr(macro, "allow_entry", True):
                return False
            if int(getattr(macro, "risk_score", 0) or 0) > 0:
                return False
        if not _underlying_extension_ok(histories, indexes, symbol, config):
            return False
    return True


def _provider_score(provider: object | None, symbol: str, as_of: date) -> object | None:
    if not provider or not hasattr(provider, "score"):
        return None
    try:
        return provider.score(symbol, as_of)
    except Exception:
        return None


def _underlying_extension_ok(
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    symbol: str,
    config: SwingConfig,
) -> bool:
    underlying = SECTOR_PROXY_BY_SYMBOL.get(symbol)
    if not underlying:
        return False
    bars = histories.get(underlying, [])
    idx = indexes.get(underlying, -1)
    required = max(config.ema_slow_window, config.relative_strength_window + 1, config.rsi_window + 1)
    if idx < required or idx >= len(bars):
        return False
    history = bars[: idx + 1]
    closes = [bar.close for bar in history]
    close = closes[-1]
    fast = _ema(closes, config.exit_ema_window)
    slow = _ema(closes, config.ema_slow_window)
    if not (close >= fast >= slow):
        return False
    if _rsi(closes, config.rsi_window) > config.leveraged_underlying_max_rsi + 4:
        return False
    return True


def _entry_signals(
    current_date: date,
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    tradable_symbols: list[str],
    positions: dict[str, SwingPosition],
    pending_exits: dict[str, str],
    config: SwingConfig,
    research_provider: object | None,
    macro_provider: object | None,
) -> list[SwingSignal]:
    regime = ""
    if config.use_four_stage_regime:
        regime = _market_regime(histories, indexes, config, current_date, macro_provider)
        if regime == "risk_off":
            return []
    elif config.use_market_filter and not _risk_on(histories, indexes, config):
        return []

    benchmark_return = _window_return(histories, indexes, config.benchmark_symbol, config.relative_strength_window)
    signals: list[SwingSignal] = []
    for symbol in tradable_symbols:
        if symbol in pending_exits or symbol in positions:
            continue
        if (
            regime
            and symbol in LEVERAGED_SYMBOLS
            and config.regime_leveraged_entry_min
            and not _regime_meets(regime, config.regime_leveraged_entry_min)
        ):
            continue
        signal = _signal_for_symbol(histories, indexes, symbol, benchmark_return, config)
        if not signal:
            continue
        if research_provider and hasattr(research_provider, "score"):
            research = research_provider.score(symbol, current_date)
            if not research.allow_entry:
                continue
            research_reason = "; ".join(research.reasons[:3])
            event_bonus = (
                min(int(getattr(research, "positive_score", 0) or 0), 4) * config.event_positive_score_bonus
                + max(float(getattr(research, "industry_score", 0.0) or 0.0), 0.0) * config.event_industry_score_bonus
            )
            signal = SwingSignal(
                symbol=signal.symbol,
                score=signal.score + research.score_adjustment + event_bonus,
                atr=signal.atr,
                reason=(
                    f"{signal.reason}; research={research.risk_level}"
                    f"/risk{research.risk_score}/pos{research.positive_score}"
                    f"/industry{research.industry_score:.2f}; {research_reason}"
                ),
                position_multiplier=research.position_multiplier,
                research_risk_level=research.risk_level,
            )
        if (
            config.use_macro_filter
            and macro_provider
            and hasattr(macro_provider, "score")
            and (not config.macro_filter_leveraged_only or signal.symbol in LEVERAGED_SYMBOLS)
        ):
            macro = macro_provider.score(symbol, current_date)
            if not macro.allow_entry:
                continue
            macro_reason = "; ".join(macro.reasons[:4])
            signal = SwingSignal(
                symbol=signal.symbol,
                score=signal.score + macro.score_adjustment,
                atr=signal.atr,
                reason=(
                    f"{signal.reason}; macro={macro.risk_level}"
                    f"/risk{macro.risk_score}/pos{macro.positive_score}; {macro_reason}"
                ),
                position_multiplier=signal.position_multiplier * macro.position_multiplier,
                research_risk_level=signal.research_risk_level,
            )
        if regime:
            signal = replace(signal, reason=f"{signal.reason}; regime={regime}", signal_date=current_date)
        else:
            signal = replace(signal, signal_date=current_date)
        signals.append(signal)
    signals.sort(key=lambda item: item.score, reverse=True)
    return signals


def _signal_for_symbol(
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    symbol: str,
    benchmark_return: float,
    config: SwingConfig,
) -> SwingSignal | None:
    if config.model_name.startswith("leveraged_overlay") and symbol in LEVERAGED_SYMBOLS:
        return _leveraged_overlay_signal_for_symbol(histories, indexes, symbol, benchmark_return, config)
    if config.model_name.startswith("leveraged_overlay"):
        return _catalyst_signal_for_symbol(histories, indexes, symbol, benchmark_return, config)
    if config.model_name.startswith("catalyst_"):
        return _catalyst_signal_for_symbol(histories, indexes, symbol, benchmark_return, config)

    bars = histories.get(symbol, [])
    idx = indexes.get(symbol, -1)
    required = max(
        config.ema_slow_window,
        config.bollinger_window,
        config.atr_window + 1,
        config.rsi_window + 1,
        config.momentum_window + 1,
        config.relative_strength_window + 1,
        config.breakout_window + 1,
        config.volume_window + 1,
    )
    if idx < required or idx >= len(bars):
        return None

    history = bars[: idx + 1]
    closes = [bar.close for bar in history]
    latest = history[-1]
    if latest.close <= 0:
        return None

    avg_dollar_volume = _average_dollar_volume(history, config.volume_window)
    if avg_dollar_volume < config.min_dollar_volume:
        return None

    previous_volumes = [bar.volume for bar in history[-config.volume_window - 1 : -1]]
    volume_average = mean(previous_volumes) if previous_volumes else 0.0
    volume_ratio = latest.volume / volume_average if volume_average > 0 else 0.0
    if volume_ratio < config.min_volume_ratio:
        return None

    ema_fast = _ema(closes, config.ema_fast_window)
    ema_slow = _ema(closes, config.ema_slow_window)
    ema_fast_prev = _ema(closes[:-5], config.ema_fast_window) if len(closes) > config.ema_fast_window + 5 else ema_fast
    trend_ok = latest.close > ema_fast > ema_slow and ema_fast >= ema_fast_prev
    if not trend_ok:
        return None

    middle, upper, lower = _bollinger(closes, config.bollinger_window, config.bollinger_std)
    if latest.close < middle:
        return None
    if upper > 0 and latest.close > upper * (1 + config.max_bollinger_extension):
        return None

    rsi_value = _rsi(closes, config.rsi_window)
    if rsi_value < config.min_rsi or rsi_value > config.max_rsi:
        return None

    momentum = latest.close / closes[-config.momentum_window - 1] - 1
    if momentum < config.min_momentum:
        return None

    symbol_relative_return = latest.close / closes[-config.relative_strength_window - 1] - 1
    relative_strength = symbol_relative_return - benchmark_return
    if relative_strength < config.min_relative_strength:
        return None

    prior_high = max(bar.high for bar in history[-config.breakout_window - 1 : -1])
    breakout = latest.close > prior_high
    pullback_reclaim = (
        latest.low <= ema_fast * (1 + config.pullback_tolerance_pct)
        and latest.close > ema_fast
        and latest.close > closes[-2]
    )
    bollinger_push = upper > 0 and latest.close >= upper * 0.98
    if not (breakout or pullback_reclaim or bollinger_push):
        return None

    atr = _atr(history, config.atr_window)
    if atr <= 0:
        return None

    band_position = (latest.close - lower) / (upper - lower) if upper > lower else 0.5
    score = (
        momentum * 2.4
        + relative_strength * 1.8
        + min(volume_ratio, 3.0) * 0.035
        + max(latest.close / ema_fast - 1, 0.0) * 1.2
        + band_position * 0.04
        + (0.08 if breakout else 0.0)
    )
    reason_parts = [
        f"momentum={momentum:.2%}",
        f"rs={relative_strength:.2%}",
        f"volume={volume_ratio:.2f}x",
        f"rsi={rsi_value:.1f}",
    ]
    if breakout:
        reason_parts.append("breakout")
    elif bollinger_push:
        reason_parts.append("bollinger_push")
    else:
        reason_parts.append("ema_reclaim")
    return SwingSignal(symbol=symbol, score=score, atr=atr, reason="; ".join(reason_parts))


def _leveraged_overlay_signal_for_symbol(
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    symbol: str,
    benchmark_return: float,
    config: SwingConfig,
) -> SwingSignal | None:
    if config.leveraged_allowed_symbols and symbol not in config.leveraged_allowed_symbols:
        return None
    if symbol in config.leveraged_blocked_symbols:
        return None

    signal = _catalyst_signal_for_symbol(histories, indexes, symbol, benchmark_return, config)
    if not signal:
        return None

    underlying = SECTOR_PROXY_BY_SYMBOL.get(symbol)
    if not underlying:
        return None
    underlying_score = _underlying_overlay_score(histories, indexes, symbol, underlying, benchmark_return, config)
    if underlying_score is None:
        return None

    reason = (
        f"{signal.reason}; leveraged_overlay={underlying}; "
        f"underlying_momentum={underlying_score['momentum']:.2%}; "
        f"underlying_short={underlying_score['short_momentum']:.2%}; "
        f"underlying_rs={underlying_score['relative_strength']:.2%}; "
        f"underlying_rsi={underlying_score['rsi']:.1f}; "
        f"underlying_volume={underlying_score['volume_ratio']:.2f}x"
    )
    score = signal.score + underlying_score["score"] + 0.18
    return SwingSignal(
        symbol=symbol,
        score=score,
        atr=signal.atr,
        reason=reason,
        position_multiplier=signal.position_multiplier,
        research_risk_level=signal.research_risk_level,
    )


def _underlying_overlay_score(
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    leveraged_symbol: str,
    symbol: str,
    benchmark_return: float,
    config: SwingConfig,
) -> dict | None:
    bars = histories.get(symbol, [])
    idx = indexes.get(symbol, -1)
    required = max(
        config.ema_slow_window,
        config.bollinger_window,
        config.volume_window + 1,
        config.momentum_window + 1,
        config.short_momentum_window + 1,
        config.relative_strength_window + 1,
        config.breakout_window + 1,
    )
    if idx < required or idx >= len(bars):
        return None

    history = bars[: idx + 1]
    closes = [bar.close for bar in history]
    latest = history[-1]
    if latest.close <= 0:
        return None

    ema_fast = _ema(closes, config.ema_fast_window)
    ema_slow = _ema(closes, config.ema_slow_window)
    if not (latest.close > ema_fast > ema_slow):
        return None

    strict = leveraged_symbol in config.leveraged_strict_symbols
    min_momentum = config.leveraged_strict_underlying_min_momentum if strict else config.leveraged_underlying_min_momentum
    min_short_momentum = (
        config.leveraged_strict_underlying_min_short_momentum if strict else config.leveraged_underlying_min_short_momentum
    )
    min_relative_strength = (
        config.leveraged_strict_underlying_min_relative_strength if strict else config.leveraged_underlying_min_relative_strength
    )
    min_volume_ratio = config.leveraged_strict_underlying_min_volume_ratio if strict else config.leveraged_underlying_min_volume_ratio
    max_rsi = config.leveraged_strict_underlying_max_rsi if strict else config.leveraged_underlying_max_rsi
    max_bollinger_extension = (
        config.leveraged_strict_underlying_max_bollinger_extension
        if strict
        else config.leveraged_underlying_max_bollinger_extension
    )
    require_breakout = config.leveraged_strict_require_underlying_breakout if strict else config.leveraged_require_underlying_breakout

    middle, upper, lower = _bollinger(closes, config.bollinger_window, config.bollinger_std)
    if upper > 0 and latest.close > upper * (1 + max_bollinger_extension):
        return None

    previous_volumes = [bar.volume for bar in history[-config.volume_window - 1 : -1]]
    volume_average = mean(previous_volumes) if previous_volumes else 0.0
    volume_ratio = latest.volume / volume_average if volume_average > 0 else 0.0
    if volume_ratio < min_volume_ratio:
        return None

    momentum = latest.close / closes[-config.momentum_window - 1] - 1
    short_momentum = latest.close / closes[-config.short_momentum_window - 1] - 1
    symbol_return = latest.close / closes[-config.relative_strength_window - 1] - 1
    relative_strength = symbol_return - benchmark_return
    rsi_value = _rsi(closes, config.rsi_window)
    prior_high = max(bar.high for bar in history[-config.breakout_window - 1 : -1])
    breakout = latest.close > prior_high

    if momentum < min_momentum:
        return None
    if short_momentum < min_short_momentum:
        return None
    if relative_strength < min_relative_strength:
        return None
    if rsi_value > max_rsi:
        return None
    if require_breakout and not breakout:
        return None

    score = (
        momentum * 3.0
        + short_momentum * 2.2
        + relative_strength * 2.4
        + max(latest.close / ema_fast - 1, 0.0) * 1.4
        + min(volume_ratio, 3.0) * 0.025
        + (0.14 if breakout else 0.0)
    )
    return {
        "score": score,
        "momentum": momentum,
        "short_momentum": short_momentum,
        "relative_strength": relative_strength,
        "rsi": rsi_value,
        "volume_ratio": volume_ratio,
        "breakout": breakout,
    }


def _catalyst_signal_for_symbol(
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    symbol: str,
    benchmark_return: float,
    config: SwingConfig,
) -> SwingSignal | None:
    bars = histories.get(symbol, [])
    idx = indexes.get(symbol, -1)
    required = max(
        config.ema_slow_window,
        config.bollinger_window + 1,
        config.atr_window + 1,
        config.rsi_window + 1,
        config.momentum_window + 1,
        config.short_momentum_window + 1,
        config.relative_strength_window + 1,
        config.breakout_window + 1,
        config.near_high_window + 1,
        config.volume_window + 1,
    )
    if config.use_macd_filter:
        required = max(required, config.macd_slow_window + config.macd_signal_window + 2)
    if idx < required or idx >= len(bars):
        return None

    history = bars[: idx + 1]
    closes = [bar.close for bar in history]
    latest = history[-1]
    previous = history[-2]
    if latest.close <= 0 or previous.close <= 0:
        return None

    avg_dollar_volume = _average_dollar_volume(history, config.volume_window)
    if avg_dollar_volume < config.min_dollar_volume:
        return None

    previous_volumes = [bar.volume for bar in history[-config.volume_window - 1 : -1]]
    volume_average = mean(previous_volumes) if previous_volumes else 0.0
    volume_ratio = latest.volume / volume_average if volume_average > 0 else 0.0
    if volume_ratio < config.min_volume_ratio:
        return None

    ema_fast = _ema(closes, config.ema_fast_window)
    ema_slow = _ema(closes, config.ema_slow_window)
    ema_fast_prev = _ema(closes[:-5], config.ema_fast_window) if len(closes) > config.ema_fast_window + 5 else ema_fast
    if not (latest.close > ema_fast > ema_slow and ema_fast >= ema_fast_prev):
        return None

    middle, upper, lower = _bollinger(closes, config.bollinger_window, config.bollinger_std)
    if upper > 0 and latest.close > upper * (1 + config.max_bollinger_extension):
        return None

    rsi_value = _rsi(closes, config.rsi_window)
    if rsi_value < config.min_rsi or rsi_value > config.max_rsi:
        return None

    momentum = latest.close / closes[-config.momentum_window - 1] - 1
    short_momentum = latest.close / closes[-config.short_momentum_window - 1] - 1
    if momentum < config.min_momentum or short_momentum < config.min_short_momentum:
        return None

    symbol_relative_return = latest.close / closes[-config.relative_strength_window - 1] - 1
    relative_strength = symbol_relative_return - benchmark_return
    if relative_strength < config.min_relative_strength:
        return None

    sector_symbol = SECTOR_PROXY_BY_SYMBOL.get(symbol)
    sector_strength = 0.0
    if config.use_sector_filter and sector_symbol:
        sector_return = _window_return(histories, indexes, sector_symbol, config.sector_relative_strength_window)
        sector_strength = sector_return - benchmark_return
        if sector_strength < config.min_sector_relative_strength:
            return None

    prior_breakout_high = max(bar.high for bar in history[-config.breakout_window - 1 : -1])
    prior_near_high = max(bar.high for bar in history[-config.near_high_window - 1 : -1])
    breakout = latest.close > prior_breakout_high
    near_high = latest.close >= prior_near_high * config.min_near_high_pct
    bollinger_breakout = upper > 0 and latest.close >= upper
    gap_up = latest.open / previous.close - 1 >= 0.02
    squeeze_rank = _recent_bollinger_bandwidth_rank(closes, config.bollinger_window, config.bollinger_std, config.squeeze_lookback_window)
    squeeze_breakout = bollinger_breakout and squeeze_rank <= config.max_squeeze_rank

    if not near_high:
        return None
    if config.require_catalyst_breakout and not (breakout or bollinger_breakout or gap_up):
        return None

    macd_state = None
    if config.use_macd_filter:
        macd_state = _macd(
            closes,
            config.macd_fast_window,
            config.macd_slow_window,
            config.macd_signal_window,
        )
        if macd_state is None:
            return None
        histogram_threshold = latest.close * config.macd_min_histogram_pct
        if macd_state["histogram"] <= histogram_threshold:
            return None
        if config.macd_require_histogram_rising and macd_state["histogram"] <= macd_state["previous_histogram"]:
            return None
        if macd_state["macd"] <= macd_state["signal"]:
            return None

    atr = _atr(history, config.atr_window)
    if atr <= 0:
        return None

    band_position = (latest.close - lower) / (upper - lower) if upper > lower else 0.5
    macd_score = 0.0
    if config.use_macd_filter and macd_state is not None:
        macd_score = max(min(macd_state["histogram_pct"], 0.03), -0.03) * 2.0
    score = (
        momentum * 2.8
        + short_momentum * 3.0
        + relative_strength * 2.2
        + sector_strength * 1.3
        + min(volume_ratio, 4.5) * 0.050
        + band_position * 0.035
        + (0.14 if breakout else 0.0)
        + (0.08 if squeeze_breakout else 0.0)
        + (0.07 if gap_up else 0.0)
        + macd_score
    )

    reason_parts = [
        f"catalyst_momentum={momentum:.2%}",
        f"short_momentum={short_momentum:.2%}",
        f"rs={relative_strength:.2%}",
        f"sector_rs={sector_strength:.2%}",
        f"volume={volume_ratio:.2f}x",
        f"rsi={rsi_value:.1f}",
    ]
    if config.use_macd_filter and macd_state is not None:
        reason_parts.append(f"macd_hist={macd_state['histogram_pct']:.2%}")
    if breakout:
        reason_parts.append("50d_breakout")
    if squeeze_breakout:
        reason_parts.append(f"squeeze_breakout_rank={squeeze_rank:.2f}")
    if bollinger_breakout and not squeeze_breakout:
        reason_parts.append("bollinger_breakout")
    if gap_up:
        reason_parts.append("gap_catalyst")
    return SwingSignal(symbol=symbol, score=score, atr=atr, reason="; ".join(reason_parts))


def _risk_on(histories: dict[str, list[Bar]], indexes: dict[str, int], config: SwingConfig) -> bool:
    market_bars = histories.get(config.market_symbol, [])
    market_idx = indexes.get(config.market_symbol, -1)
    if market_idx < config.market_filter_window or market_idx >= len(market_bars):
        return False
    closes = [bar.close for bar in market_bars[: market_idx + 1]]
    if closes[-1] <= _ema(closes, config.market_filter_window):
        return False

    vix_bars = histories.get(config.volatility_symbol, [])
    vix_idx = indexes.get(config.volatility_symbol, -1)
    if vix_bars and 0 <= vix_idx < len(vix_bars) and vix_bars[vix_idx].close >= config.vix_threshold:
        return False
    return True


def _window_return(histories: dict[str, list[Bar]], indexes: dict[str, int], symbol: str, window: int) -> float:
    bars = histories.get(symbol, [])
    idx = indexes.get(symbol, -1)
    if idx < window or idx >= len(bars):
        return 0.0
    previous = bars[idx - window].close
    return bars[idx].close / previous - 1 if previous else 0.0


def _average_dollar_volume(bars: list[Bar], window: int) -> float:
    recent = bars[-window:]
    if not recent:
        return 0.0
    return mean(bar.close * bar.volume for bar in recent)


def _atr(bars: list[Bar], window: int) -> float:
    if len(bars) <= window:
        return 0.0
    ranges: list[float] = []
    for idx in range(len(bars) - window, len(bars)):
        bar = bars[idx]
        previous_close = bars[idx - 1].close
        ranges.append(max(bar.high - bar.low, abs(bar.high - previous_close), abs(bar.low - previous_close)))
    return mean(ranges) if ranges else 0.0


def _ema(values: list[float], window: int) -> float:
    if len(values) < window:
        return values[-1] if values else 0.0
    multiplier = 2 / (window + 1)
    value = mean(values[:window])
    for item in values[window:]:
        value = item * multiplier + value * (1 - multiplier)
    return value


def _ema_series(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    multiplier = 2 / (window + 1)
    ema_values: list[float] = []
    current = values[0]
    for value in values:
        current = value if not ema_values else value * multiplier + current * (1 - multiplier)
        ema_values.append(current)
    return ema_values


def _macd(
    values: list[float],
    fast_window: int = 12,
    slow_window: int = 26,
    signal_window: int = 9,
) -> dict[str, float] | None:
    required = max(fast_window, slow_window) + signal_window + 1
    if len(values) < required or not values[-1]:
        return None
    fast = _ema_series(values, fast_window)
    slow = _ema_series(values, slow_window)
    macd_values = [fast_value - slow_value for fast_value, slow_value in zip(fast, slow)]
    signal_values = _ema_series(macd_values, signal_window)
    if len(signal_values) < 2:
        return None
    histogram = macd_values[-1] - signal_values[-1]
    previous_histogram = macd_values[-2] - signal_values[-2]
    return {
        "macd": macd_values[-1],
        "signal": signal_values[-1],
        "histogram": histogram,
        "previous_histogram": previous_histogram,
        "histogram_pct": histogram / values[-1],
    }


def _bollinger(values: list[float], window: int, std_multiple: float) -> tuple[float, float, float]:
    recent = values[-window:]
    middle = mean(recent)
    deviation = pstdev(recent) if len(recent) > 1 else 0.0
    return middle, middle + deviation * std_multiple, middle - deviation * std_multiple


def _recent_bollinger_bandwidth_rank(values: list[float], window: int, std_multiple: float, lookback: int) -> float:
    if len(values) < window + 2:
        return 1.0
    widths: list[float] = []
    for end in range(window, len(values) + 1):
        middle, upper, lower = _bollinger(values[:end], window, std_multiple)
        if middle > 0:
            widths.append((upper - lower) / middle)
    if len(widths) < 2:
        return 1.0
    current = widths[-2]
    sample = widths[-lookback - 1 : -1] if len(widths) > lookback else widths[:-1]
    if not sample:
        return 1.0
    return sum(1 for width in sample if width <= current) / len(sample)


def _rsi(values: list[float], window: int) -> float:
    if len(values) <= window:
        return 50.0
    changes = [current - previous for previous, current in zip(values[-window - 1 : -1], values[-window:])]
    gains = [max(change, 0.0) for change in changes]
    losses = [abs(min(change, 0.0)) for change in changes]
    average_gain = mean(gains) if gains else 0.0
    average_loss = mean(losses) if losses else 0.0
    if average_loss == 0:
        return 100.0 if average_gain > 0 else 50.0
    relative_strength = average_gain / average_loss
    return 100 - (100 / (1 + relative_strength))


def _sell_position(
    current_date: date,
    cash: float,
    positions: dict[str, SwingPosition],
    symbol: str,
    gross_price: float,
    fraction: float,
    reason: str,
    cost_rate: float,
    trade_log: list[dict],
    closed_trades: list[dict],
) -> float:
    position = positions.get(symbol)
    if position is None:
        return cash
    qty = position.qty * min(max(fraction, 0.0), 1.0)
    if qty <= 0:
        return cash
    net_price = gross_price * (1 - cost_rate)
    proceeds = qty * net_price
    realized_pnl = proceeds - qty * position.entry_price
    pnl_pct = realized_pnl / (qty * position.entry_price) if position.entry_price and qty else 0.0
    cash += proceeds
    position.qty -= qty

    trade_log.append(
        _transaction_record(
            current_date,
            symbol,
            "SELL",
            qty,
            net_price,
            proceeds,
            realized_pnl,
            pnl_pct,
            reason,
            cash,
        )
    )
    closed_trades.append(
        {
            "entry_date": position.entry_date.isoformat(),
            "exit_date": current_date.isoformat(),
            "symbol": symbol,
            "qty": qty,
            "entry_price": position.entry_price,
            "exit_price": net_price,
            "notional": qty * position.entry_price,
            "realized_pnl": realized_pnl,
            "pnl_pct": pnl_pct,
            "bars_held": position.bars_held,
            "reason": reason,
            "entry_reason": position.entry_reason,
        }
    )
    if position.qty <= 0.000001:
        positions.pop(symbol, None)
    return cash


def _liquidate_at_end(
    current_date: date,
    cash: float,
    positions: dict[str, SwingPosition],
    histories: dict[str, list[Bar]],
    indexes: dict[str, int],
    cost_rate: float,
    trade_log: list[dict],
    closed_trades: list[dict],
) -> None:
    for symbol in list(positions):
        idx = indexes.get(symbol, -1)
        if idx < 0:
            continue
        bar = histories[symbol][idx]
        cash = _sell_position(
            current_date,
            cash,
            positions,
            symbol,
            bar.close,
            1.0,
            "end_of_test",
            cost_rate,
            trade_log,
            closed_trades,
        )


def _transaction_record(
    trade_date: date,
    symbol: str,
    side: str,
    qty: float,
    price: float,
    notional: float,
    realized_pnl: float | str,
    pnl_pct: float | str,
    reason: str,
    portfolio_cash: float,
) -> dict:
    return {
        "date": trade_date.isoformat(),
        "symbol": symbol,
        "side": side,
        "qty": qty,
        "price": price,
        "notional": notional,
        "realized_pnl": realized_pnl,
        "pnl_pct": pnl_pct,
        "reason": reason,
        "portfolio_cash": portfolio_cash,
    }


def _cost_rate(config: SwingConfig) -> float:
    return (config.cost_bps + config.slippage_bps) / 10_000
