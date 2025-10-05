from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional, Tuple
import os
import sys

import numpy as np
import pandas as pd
import pytz

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.append(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

if __package__:
    from . import config
else:  # pragma: no cover - script execution fallback
    import config  # type: ignore


UTC = pytz.UTC


def ensure_utc(ts: pd.Timestamp | datetime) -> datetime:
    if isinstance(ts, pd.Timestamp):
        ts = ts.to_pydatetime()
    if ts.tzinfo is None:
        return UTC.localize(ts)
    return ts.astimezone(UTC)


def compute_atr(df: pd.DataFrame, period: int = config.ATR_PERIOD) -> Optional[float]:
    if df is None or len(df) < period + 2:
        return None
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean()
    atr_val = atr_series.iloc[-2]
    if pd.isna(atr_val):
        return None
    return float(atr_val)


def calculate_adx(df: pd.DataFrame, period: int = 14) -> Optional[float]:
    if df is None or len(df) < period + 1:
        return None

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)

    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    dm_plus = ((high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)).fillna(0)
    dm_minus = ((low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)).fillna(0)

    tr_smooth = tr.rolling(period).mean()
    dm_plus_smooth = dm_plus.rolling(period).mean()
    dm_minus_smooth = dm_minus.rolling(period).mean()

    di_plus = 100 * (dm_plus_smooth / tr_smooth)
    di_minus = 100 * (dm_minus_smooth / tr_smooth)

    dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus))
    adx = dx.rolling(period).mean()

    last_value = adx.iloc[-1]
    return float(last_value) if not pd.isna(last_value) else None


def detect_market_regime(df: pd.DataFrame) -> str:
    if df is None or len(df) < config.REGIME_LOOKBACK_PERIODS:
        return "RANGING"

    adx = calculate_adx(df)
    current_atr = compute_atr(df, period=14)
    historical_atr = compute_atr(df.iloc[:-10], period=14) if len(df) > 10 else None

    if current_atr is None or historical_atr in (None, 0):
        return "RANGING"

    volatility_ratio = current_atr / historical_atr

    if adx is not None and adx > config.TREND_THRESHOLD * 100:
        if volatility_ratio > config.VOLATILITY_THRESHOLD:
            return "VOLATILE"
        return "TRENDING"

    if volatility_ratio > config.VOLATILITY_THRESHOLD:
        return "VOLATILE"

    return "RANGING"


def get_session_info(timestamp: datetime) -> str:
    ts = ensure_utc(timestamp)
    hour = ts.hour

    if 22 <= hour or hour < 7:
        return "Sydney/Tokyo"
    if 8 <= hour < 13:
        return "London"
    if 13 <= hour < 17:
        return "London-NY Overlap"
    if 17 <= hour < 22:
        return "New York"
    return "Dead Zone"


def is_market_session_active(timestamp: datetime) -> bool:
    ts = ensure_utc(timestamp)
    weekday = ts.weekday()
    hour = ts.hour

    if weekday == 6:
        return hour >= 22
    if weekday == 5:
        return False
    if weekday == 4 and hour >= 22:
        return False
    if 7 <= hour < 8:
        return False
    return True


def get_instrument_session_priority(symbol: str, timestamp: datetime) -> int:
    session = get_session_info(timestamp)
    return config.INSTRUMENT_SESSION_PRIORITY.get(symbol, {}).get(session, 3)


def should_trade_instrument(symbol: str, timestamp: datetime) -> bool:
    if not is_market_session_active(timestamp):
        return False
    priority = get_instrument_session_priority(symbol, timestamp)
    return priority >= config.MIN_SESSION_PRIORITY


def get_regime_strategy_weight(strategy_name: str, regime: str) -> float:
    weights = config.REGIME_STRATEGY_WEIGHTS.get(regime, config.REGIME_STRATEGY_WEIGHTS["RANGING"])
    return weights.get(strategy_name, 0.2)


def should_trade_strategy_in_regime(strategy_name: str, regime: str) -> bool:
    return get_regime_strategy_weight(strategy_name, regime) >= 0.15


def calculate_risk_multiplier(session_priority: int, regime_weight: float, performance_weight: float) -> float:
    multiplier = 1.0
    if session_priority >= 5:
        multiplier += 0.15
    elif session_priority == 4:
        multiplier += 0.05

    baseline_regime = 0.2
    multiplier += (regime_weight - baseline_regime) * 0.4
    multiplier += (performance_weight - 1.0) * 0.3

    multiplier = max(config.RISK_MULTIPLIER_MIN, min(config.RISK_MULTIPLIER_MAX, multiplier))
    return round(multiplier, 2)


def confirm_signal_with_mtf(
    signal: str,
    symbol: str,
    timestamp: datetime,
    h1_df: pd.DataFrame,
) -> bool:
    if not config.ENABLE_MTF_CONFIRMATION:
        return True

    ts = ensure_utc(timestamp)
    if h1_df is None or h1_df.empty:
        return True

    subset = h1_df[h1_df["time"] <= ts].tail(12)
    if len(subset) < 10:
        return True

    close = subset["close"].astype(float)
    ma_short = close.rolling(5).mean().iloc[-1]
    ma_long = close.rolling(10).mean().iloc[-1]
    current_price = close.iloc[-1]

    if pd.isna(ma_short) or pd.isna(ma_long):
        return True

    bias = "neutral"
    if ma_short > ma_long and current_price > ma_short:
        bias = "bullish"
    elif ma_short < ma_long and current_price < ma_short:
        bias = "bearish"

    if bias == "neutral":
        return True
    if signal == "buy" and bias == "bullish":
        return True
    if signal == "sell" and bias == "bearish":
        return True
    return False


@dataclass
class MicroMomentumResult:
    passed: bool
    score: float
    threshold: float
    soft_pass: bool


def confirm_with_micro_momentum(
    symbol: str,
    signal: str,
    regime: str,
    atr_value: Optional[float],
    reference_price: Optional[float],
    timestamp: datetime,
    micro_df: pd.DataFrame,
) -> MicroMomentumResult:
    if not config.ENABLE_MICRO_MOMENTUM_CONFIRMATION:
        return MicroMomentumResult(True, 0.0, 0.0, False)

    if micro_df is None or micro_df.empty:
        return MicroMomentumResult(True, 0.0, 0.0, False)

    ts = ensure_utc(timestamp)
    subset = micro_df[micro_df["time"] <= ts].tail(config.MICRO_MOMENTUM_LOOKBACK + 3)
    if len(subset) < config.MICRO_MOMENTUM_LOOKBACK + 1:
        return MicroMomentumResult(True, 0.0, 0.0, False)

    close = subset["close"].astype(float)
    momentum_series = close.pct_change().rolling(config.MICRO_MOMENTUM_LOOKBACK).sum()
    momentum_score = float(momentum_series.iloc[-1])

    if np.isnan(momentum_score):
        return MicroMomentumResult(True, 0.0, 0.0, False)

    threshold = config.MICRO_MOMENTUM_BASE_THRESHOLD
    if atr_value and reference_price:
        atr_pct = max(1e-9, atr_value / reference_price)
        dynamic = atr_pct * config.MICRO_MOMENTUM_DYNAMIC_MULTIPLIER
        threshold = max(
            config.MICRO_MOMENTUM_MIN_THRESHOLD,
            min(config.MICRO_MOMENTUM_MAX_THRESHOLD, dynamic),
        )

    if regime == "TRENDING":
        threshold *= 0.75
    elif regime == "VOLATILE":
        threshold *= 0.85

    threshold = max(config.MICRO_MOMENTUM_MIN_THRESHOLD, threshold)

    soft_pass = False
    if signal == "buy":
        if momentum_score >= threshold:
            return MicroMomentumResult(True, momentum_score, threshold, False)
        if momentum_score >= threshold * config.MICRO_MOMENTUM_SOFT_PASS_RATIO:
            soft_pass = True
            return MicroMomentumResult(True, momentum_score, threshold, True)
    else:
        if momentum_score <= -threshold:
            return MicroMomentumResult(True, momentum_score, threshold, False)
        if momentum_score <= -threshold * config.MICRO_MOMENTUM_SOFT_PASS_RATIO:
            soft_pass = True
            return MicroMomentumResult(True, momentum_score, threshold, True)

    return MicroMomentumResult(False, momentum_score, threshold, soft_pass)


def calculate_position_size(
    symbol_info,
    atr_value: Optional[float],
    sl_mult: float,
    account_balance: float,
    risk_multiplier: float = 1.0,
) -> float:
    if account_balance <= 0:
        account_balance = config.DEFAULT_ACCOUNT_BALANCE

    if not atr_value or atr_value <= 0:
        if symbol_info is not None and getattr(symbol_info, "name", "").upper().startswith("XAUUSD"):
            return 0.1
        return 0.5

    bounded_multiplier = max(config.RISK_MULTIPLIER_MIN, min(config.RISK_MULTIPLIER_MAX, risk_multiplier))
    risk_amount = account_balance * config.ACCOUNT_RISK_PER_TRADE * bounded_multiplier

    tick_size = getattr(symbol_info, "trade_tick_size", None) or getattr(symbol_info, "point", 0.01) or 0.01
    tick_value = getattr(symbol_info, "trade_tick_value", 1.0) or 1.0
    sl_distance = atr_value * sl_mult
    ticks_at_risk = sl_distance / tick_size
    loss_per_unit = ticks_at_risk * tick_value

    if loss_per_unit <= 0:
        return config.MIN_LOT_SIZE

    optimal_size = risk_amount / loss_per_unit
    position_size = max(config.MIN_LOT_SIZE, min(optimal_size, config.MAX_LOT_SIZE))
    return round(position_size, 2)


def normalize_volume(volume: float, symbol_info) -> float:
    if symbol_info is None:
        return round(volume, 2)
    step = getattr(symbol_info, "volume_step", 0.01) or 0.01
    min_volume = getattr(symbol_info, "volume_min", step) or step
    max_volume = getattr(symbol_info, "volume_max", volume) or volume
    normalized = max(min_volume, min(volume, max_volume))
    steps = round(normalized / step)
    normalized = steps * step
    decimals = len(str(step).split(".")[-1]) if "." in str(step) else 0
    return round(normalized, decimals)


def compute_spread_points(symbol_info, bar: pd.Series) -> float:
    if symbol_info is None:
        return float("inf")
    point = getattr(symbol_info, "point", 0.0001) or 0.0001
    if "spread" in bar:
        # MetaTrader returns spread in points already
        return float(bar["spread"])
    if "ask" in bar and "bid" in bar and point:
        return float((bar["ask"] - bar["bid"]) / point)
    return float("inf")


def compute_spread_ratio(spread_points: float, symbol_info, atr_value: Optional[float]) -> float:
    if atr_value in (None, 0) or spread_points in (None, float("inf")):
        return 0.0
    point = getattr(symbol_info, "point", 0.0001) or 0.0001
    spread = spread_points * point
    return spread / atr_value


def should_block_for_correlation(
    symbol: str,
    signal: str,
    open_positions: Dict[str, "OpenPosition"],
) -> Tuple[bool, float]:
    active_groups: Dict[str, int] = {}
    exposure_multiplier = 1.0

    # Identify groups for the candidate symbol
    candidate_groups = [name for name, members in config.CORRELATION_GROUPS.items() if symbol in members]
    if not candidate_groups:
        return False, 1.0

    for pos in open_positions.values():
        if pos.status != "open":
            continue
        for group_name, members in config.CORRELATION_GROUPS.items():
            if pos.symbol in members:
                active_groups[group_name] = active_groups.get(group_name, 0) + 1

    should_block = False
    for group in candidate_groups:
        count = active_groups.get(group, 0)
        if count >= config.MAX_CORRELATED_POSITIONS:
            should_block = True
        elif count > 0:
            exposure_multiplier = min(exposure_multiplier, config.CORRELATION_POSITION_LIMIT)

    return should_block, exposure_multiplier


@dataclass
class StrategyPerformance:
    total_trades: int = 0
    winning_trades: int = 0
    total_pnl: float = 0.0

    def record_trade(self, pnl: float) -> None:
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.0
        return self.winning_trades / self.total_trades

    @property
    def performance_weight(self) -> float:
        if self.total_trades < config.MIN_TRADES_FOR_ADAPTATION:
            return 1.0
        approx_profit_factor = self.total_pnl / max(1.0, self.total_trades)
        if approx_profit_factor > 0.1 and self.win_rate > 0.4:
            return min(1.5, 1.0 + (approx_profit_factor * 0.5) + (self.win_rate * 0.3))
        if approx_profit_factor < -0.05 or self.win_rate < 0.3:
            return max(0.3, 1.0 + approx_profit_factor)
        return 1.0


@dataclass
class OpenPosition:
    symbol: str
    direction: str  # 'buy' or 'sell'
    strategy_key: str
    strategy_label: str
    lot: float
    entry_price: float
    entry_index: int
    sl: Optional[float]
    tp: Optional[float]
    intrabar_priority: str
    risk_multiplier: float
    sl_mult: float
    tp_mult: float
    atr_value: Optional[float] = None
    atr_band: str = "normal"
    atr_period: int = 0
    status: str = "open"
    last_checked_index: int = -1
    open_time: Optional[datetime] = None
    close_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    

    