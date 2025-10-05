from __future__ import annotations

import os
import random
import sys
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from statistics import mean
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import MetaTrader5 as mt5
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
    from .helpers import (
        OpenPosition,
        StrategyPerformance,
        calculate_position_size,
        calculate_risk_multiplier,
        compute_atr,
        compute_spread_points,
        compute_spread_ratio,
        confirm_signal_with_mtf,
        confirm_with_micro_momentum,
        detect_market_regime,
        get_instrument_session_priority,
        get_regime_strategy_weight,
        normalize_volume,
        should_block_for_correlation,
        should_trade_instrument,
        should_trade_strategy_in_regime,
    )
else:  # pragma: no cover - direct script execution fallback
    import config  # type: ignore
    from helpers import (  # type: ignore
        OpenPosition,
        StrategyPerformance,
        calculate_position_size,
        calculate_risk_multiplier,
        compute_atr,
        compute_spread_points,
        compute_spread_ratio,
        confirm_signal_with_mtf,
        confirm_with_micro_momentum,
        detect_market_regime,
        get_instrument_session_priority,
        get_regime_strategy_weight,
        normalize_volume,
        should_block_for_correlation,
        should_trade_instrument,
        should_trade_strategy_in_regime,
    )

from live.agents import (
    BreakoutAgent,
    DonchianChannelAgent,
    MACrossoverAgent,
    MeanReversionAgent,
    MomentumTrendAgent,
)


UTC = pytz.UTC

AGENT_CLASS_MAP = {
    "ma_crossover": MACrossoverAgent,
    "mean_reversion": MeanReversionAgent,
    "momentum_trend": MomentumTrendAgent,
    "breakout": BreakoutAgent,
    "donchian_channel": DonchianChannelAgent,
}


@dataclass
class StrategyDefinition:
    label: str
    name: str
    params: Dict[str, float]
    sl_mult: float
    tp_mult: float
    priority: int
    intrabar_priority: str = config.INTRABAR_PRIORITY
    atr_period: int = config.ATR_PERIOD
    atr_bands: Dict[str, Dict[str, float]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "StrategyDefinition":
        raw_bands = raw.get("atr_bands", {}) or {}
        atr_bands: Dict[str, Dict[str, float]] = {}
        if isinstance(raw_bands, dict):
            for band_key, settings in raw_bands.items():
                if not isinstance(settings, dict):
                    continue
                normalized_key = str(band_key).lower()
                parsed: Dict[str, float] = {}
                period_value = settings.get("period")
                if period_value is not None:
                    try:
                        parsed["period"] = float(int(period_value))
                    except (TypeError, ValueError):
                        pass
                sl_value = settings.get("sl_mult")
                if sl_value is not None:
                    try:
                        parsed["sl_mult"] = float(sl_value)
                    except (TypeError, ValueError):
                        pass
                tp_value = settings.get("tp_mult")
                if tp_value is not None:
                    try:
                        parsed["tp_mult"] = float(tp_value)
                    except (TypeError, ValueError):
                        pass
                atr_bands[normalized_key] = parsed
        return cls(
            label=str(raw.get("label", raw.get("name", "Strategy"))),
            name=str(raw.get("name")),
            params=dict(raw.get("params", {})),
            sl_mult=float(raw.get("sl_mult", config.SL_ATR_MULTIPLIER)),
            tp_mult=float(raw.get("tp_mult", config.TP_ATR_MULTIPLIER)),
            priority=int(raw.get("priority", 999)),
            intrabar_priority=str(raw.get("intrabar_priority", config.INTRABAR_PRIORITY)),
            atr_period=int(raw.get("atr_period", config.ATR_PERIOD)),
            atr_bands=atr_bands,
        )


@dataclass
class StrategyState:
    definition: StrategyDefinition
    agent: object
    performance: StrategyPerformance = field(default_factory=StrategyPerformance)


@dataclass
class PendingEntry:
    symbol: str
    strategy_key: str
    strategy_label: str
    direction: str
    entry_index: int
    atr_value: float
    sl_mult: float
    tp_mult: float
    intrabar_priority: str
    risk_multiplier: float
    lot: float
    regime: str
    adjusted_priority: float
    session_priority: int
    micro_score: float
    micro_threshold: float
    micro_soft_pass: bool
    atr_band: str = "normal"
    atr_period: int = 0


@dataclass
class TradeRecord:
    symbol: str
    strategy_key: str
    strategy_label: str
    direction: str
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    lot: float
    pnl: float
    risk_multiplier: float
    sl_mult: float
    tp_mult: float
    intrabar_priority: str
    atr_band: str = "normal"
    atr_period: int = 0


@dataclass
class OptimizationWeights:
    profit: float = 1.0
    drawdown: float = 0.15
    trade_count: float = 0.02
    consistency: float = 0.3


@dataclass
class WalkForwardConfig:
    segments: int = 3
    min_segment_trades: int = 10
    validation_ratio: Optional[float] = None
    stability_weight: float = 0.35
    dispersion_penalty: float = 0.12
    negative_penalty: float = 0.2


class BacktestEngine:
    def __init__(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        initial_balance: float = config.DEFAULT_INITIAL_BALANCE,
        strategy_definitions: Optional[Sequence[StrategyDefinition]] = None,
    ) -> None:
        self.start_date = start_date or datetime.now() - timedelta(days=config.DEFAULT_LOOKBACK_DAYS)
        self.end_date = end_date or datetime.now()
        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)
        self.symbols = list(config.SYMBOLS)
        self.timeframe = config.TIMEFRAME
        if strategy_definitions is None:
            self.strategy_definitions = [StrategyDefinition.from_dict(item) for item in config.STRATEGY_DEFINITIONS]
        else:
            self.strategy_definitions = [replace(defn) for defn in strategy_definitions]

        self.data: Dict[str, pd.DataFrame] = {}
        self.micro_data: Dict[str, pd.DataFrame] = {}
        self.higher_timeframe_data: Dict[str, pd.DataFrame] = {}
        self.symbol_info: Dict[str, object] = {}
        self.strategy_states: Dict[str, Dict[str, StrategyState]] = {}
        self.open_positions: Dict[int, OpenPosition] = {}
        self.pending_entries: List[PendingEntry] = []
        self.trades: List[TradeRecord] = []
        self.balance_curve: List[float] = [self.balance]
        self.position_counter = 0
        self._data_loaded = False
        self._connection_owned = True
        self._atr_cache = {}  # type: Dict[str, Dict[int, np.ndarray]]
        self._scoring_profile = False

    @classmethod
    def from_existing(
        cls,
        other: "BacktestEngine",
        strategy_definitions: Optional[Sequence[StrategyDefinition]] = None,
    ) -> "BacktestEngine":
        clone = cls(
            start_date=other.start_date,
            end_date=other.end_date,
            initial_balance=other.initial_balance,
            strategy_definitions=strategy_definitions or other.clone_strategy_definitions(),
        )
        clone.data = other.data
        clone.micro_data = other.micro_data
        clone.higher_timeframe_data = other.higher_timeframe_data
        clone.symbol_info = other.symbol_info
        clone._data_loaded = True
        clone._connection_owned = False
        clone._scoring_profile = other._scoring_profile
        return clone

    def clone_strategy_definitions(self) -> List[StrategyDefinition]:
        return [replace(defn) for defn in self.strategy_definitions]

    def set_scoring_profile(self, enabled: bool = True) -> None:
        """Enable a relaxed filter profile used during scoring/optimization runs."""
        self._scoring_profile = enabled

    def _collect_required_atr_periods(self) -> Set[int]:
        periods: Set[int] = set()
        base_period = int(getattr(config, "ATR_PERIOD", 0) or 0)
        if base_period:
            periods.add(base_period)

        lookback_multiplier = float(getattr(config, "ATR_VOL_LOOKBACK_MULTIPLIER", 2.0))
        min_period = int(max(1, getattr(config, "ATR_VOL_MIN_PERIOD", 3)))

        for definition in self.strategy_definitions:
            periods.add(int(definition.atr_period))
            if definition.atr_bands:
                for band_settings in definition.atr_bands.values():
                    candidate = int(band_settings.get("period", definition.atr_period))
                    if candidate:
                        periods.add(candidate)

            if getattr(config, "ENABLE_ADAPTIVE_ATR", False):
                long_period = int(round(definition.atr_period * lookback_multiplier))
                if long_period <= definition.atr_period:
                    long_period = definition.atr_period + max(1, int(definition.atr_period * 0.25))
                long_period = max(min_period, long_period)
                periods.add(long_period)

        return {period for period in periods if period >= 1}

    @staticmethod
    def _compute_true_range(df: pd.DataFrame) -> pd.Series:
        high = df["high"]
        low = df["low"]
        close = df["close"]
        prev_close = close.shift(1)
        return pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ], axis=1).max(axis=1)

    def _precompute_symbol_atr(self, symbol: str, df: pd.DataFrame, periods: Set[int]) -> None:
        if df.empty:
            return

        tr = self._compute_true_range(df)
        cache = self._atr_cache.setdefault(symbol, {})
        for period in periods:
            if period in cache:
                continue
            if period <= 0 or len(tr) < period:
                cache[period] = np.full(len(tr), np.nan, dtype=float)
                continue
            atr_series = tr.rolling(period).mean().shift(1)
            cache[period] = atr_series.to_numpy(dtype=float)

    def _get_precomputed_atr(self, symbol: str, period: int, index: int) -> Optional[float]:
        cache = self._atr_cache.get(symbol)
        if cache is None:
            return None

        period = int(period)
        series = cache.get(period)
        if series is None:
            df = self.data.get(symbol)
            if df is None:
                return None
            self._precompute_symbol_atr(symbol, df, {period})
            series = cache.get(period)
            if series is None:
                return None

        if index >= len(series) or index < 0:
            return None

        value = series[index]
        if value is None or not np.isfinite(value) or value <= 0:
            return None
        return float(value)

    def run(self) -> Dict[str, object]:
        try:
            self._initialize()
            self._simulate()
            return self._build_results()
        finally:
            if self._connection_owned:
                mt5.shutdown()

    def _initialize(self) -> None:
        if self._connection_owned:
            if not mt5.initialize():
                code, message = mt5.last_error()
                raise RuntimeError(f"MetaTrader5 initialize failed: {code} | {message}")

        if not self._data_loaded:
            self._load_market_data()
            self._data_loaded = True

        self._prepare_strategies()

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------
    def _load_market_data(self) -> None:
        required_periods = self._collect_required_atr_periods()
        for symbol in self.symbols:
            rates = mt5.copy_rates_range(symbol, self.timeframe, self.start_date, self.end_date)
            if rates is None or len(rates) == 0:
                print(f"⚠️  No data returned for {symbol} on timeframe {self.timeframe}")
                continue

            df = pd.DataFrame(rates)
            df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
            self.data[symbol] = df

            micro_start = self.start_date - timedelta(days=5)
            micro_rates = mt5.copy_rates_range(symbol, config.MICRO_MOMENTUM_TIMEFRAME, micro_start, self.end_date)
            if micro_rates is not None and len(micro_rates) > 0:
                micro_df = pd.DataFrame(micro_rates)
                micro_df["time"] = pd.to_datetime(micro_df["time"], unit="s", utc=True)
                self.micro_data[symbol] = micro_df
            else:
                self.micro_data[symbol] = pd.DataFrame()

            mtf_start = self.start_date - timedelta(days=5)
            mtf_rates = mt5.copy_rates_range(symbol, config.MTF_TIMEFRAME, mtf_start, self.end_date)
            if mtf_rates is not None and len(mtf_rates) > 0:
                mtf_df = pd.DataFrame(mtf_rates)
                mtf_df["time"] = pd.to_datetime(mtf_df["time"], unit="s", utc=True)
                self.higher_timeframe_data[symbol] = mtf_df
            else:
                self.higher_timeframe_data[symbol] = pd.DataFrame()

            self.symbol_info[symbol] = mt5.symbol_info(symbol)
            self._precompute_symbol_atr(symbol, df, required_periods)

    def _prepare_strategies(self) -> None:
        self.strategy_states.clear()
        for symbol in self.symbols:
            states: Dict[str, StrategyState] = {}
            for definition in self.strategy_definitions:
                agent_cls = AGENT_CLASS_MAP.get(definition.name)
                if agent_cls is None:
                    raise ValueError(f"Unsupported strategy '{definition.name}'")
                agent = agent_cls(**definition.params)
                states[definition.name] = StrategyState(definition=replace(definition), agent=agent)
            self.strategy_states[symbol] = states

    # ------------------------------------------------------------------
    # Simulation core
    # ------------------------------------------------------------------
    def _simulate(self) -> None:
        if not self.data:
            raise RuntimeError("No market data available for simulation")

        min_bars = min(len(df) for df in self.data.values() if not df.empty)
        warmup = self._compute_warmup_bars()
        final_index = min_bars - 1

        if warmup >= final_index:
            raise RuntimeError("Not enough historical data to cover warmup period")

        dots_cycle = ['.', '..', '...']
        status_index = 0

        for idx in range(warmup, final_index + 1):
            timestamp = self._get_timestamp(idx)
            self._open_pending_entries(idx, timestamp)
            self._update_positions(idx)
            if idx < final_index:
                self._evaluate_new_signals(idx)
            self.balance_curve.append(self.balance)

            if idx % 10 == 0 or idx == final_index:
                dots = dots_cycle[status_index % len(dots_cycle)]
                print(f"Simulating{dots} (bar {idx - warmup + 1}/{final_index - warmup + 1})", end='\r', flush=True)
                status_index += 1

        # Force close any remaining positions at final close price
        if self.open_positions:
            self._force_close_remaining_positions(final_index)

        print("Simulating... done!".ljust(60))

    def _get_timestamp(self, index: int) -> datetime:
        reference_symbol = next(iter(self.data))
        df = self.data[reference_symbol]
        ts = df["time"].iloc[index]
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
        return ts

    def _compute_warmup_bars(self) -> int:
        requirements = [
            config.REGIME_LOOKBACK_PERIODS + 5,
            config.MTF_LOOKBACK_BARS + 5,
            config.MICRO_MOMENTUM_LOOKBACK + 5,
        ]
        for definition in self.strategy_definitions:
            requirements.append(definition.atr_period + 5)
            for band_settings in definition.atr_bands.values():
                period_override = int(band_settings.get("period", definition.atr_period))
                requirements.append(period_override + 5)
            if definition.name == "ma_crossover":
                fast = int(definition.params.get("fast_period", 5))
                slow = int(definition.params.get("slow_period", 30))
                requirements.append(max(fast, slow) + 5)
            elif definition.name == "mean_reversion":
                ma_period = int(definition.params.get("ma_period", 10))
                requirements.append(ma_period + 5)
            elif definition.name == "momentum_trend":
                ma_period = int(definition.params.get("ma_period", 30))
                roc_period = int(definition.params.get("roc_period", 5))
                requirements.append(max(ma_period, roc_period + 2) + 5)
            elif definition.name == "breakout":
                lookback = int(definition.params.get("lookback", 20))
                requirements.append(lookback + 5)
            elif definition.name == "donchian_channel":
                chan = int(definition.params.get("channel_length", 20))
                requirements.append(chan + 5)
        return max(requirements)

    def _apply_atr_band_adjustment(
        self,
        definition: StrategyDefinition,
        base_atr_value: float,
        get_atr: Callable[[int], Optional[float]],
    ) -> Tuple[Optional[float], float, float, str, int]:
        if not config.ENABLE_ADAPTIVE_ATR or not definition.atr_bands or base_atr_value <= 0:
            return base_atr_value, definition.sl_mult, definition.tp_mult, "normal", definition.atr_period

        low_threshold = getattr(config, "ATR_VOL_LOW_THRESHOLD", 0.85)
        high_threshold = getattr(config, "ATR_VOL_HIGH_THRESHOLD", 1.2)
        lookback_multiplier = getattr(config, "ATR_VOL_LOOKBACK_MULTIPLIER", 2.0)
        min_period = int(max(1, getattr(config, "ATR_VOL_MIN_PERIOD", 3)))

        long_period = int(round(definition.atr_period * lookback_multiplier))
        if long_period <= definition.atr_period:
            long_period = definition.atr_period + max(1, int(definition.atr_period * 0.25))
        long_period = max(min_period, long_period)

        long_atr_value = get_atr(long_period)
        if long_atr_value is None or long_atr_value <= 0:
            long_atr_value = base_atr_value

        ratio = base_atr_value / long_atr_value if long_atr_value else 1.0

        band_key = "normal"
        if ratio <= low_threshold:
            band_key = "low"
        elif ratio >= high_threshold:
            band_key = "high"

        band_settings = definition.atr_bands.get(band_key) or definition.atr_bands.get("normal") or {}
        period_override = int(band_settings.get("period", definition.atr_period))
        period_override = max(min_period, period_override)

        resolved_atr = base_atr_value
        if period_override != definition.atr_period:
            override_atr = get_atr(period_override)
            if override_atr is not None and override_atr > 0:
                resolved_atr = override_atr

        sl_mult = float(band_settings.get("sl_mult", definition.sl_mult))
        tp_mult = float(band_settings.get("tp_mult", definition.tp_mult))

        return resolved_atr, sl_mult, tp_mult, band_key, period_override

    def _open_pending_entries(self, index: int, timestamp: datetime) -> None:
        if not self.pending_entries:
            return

        remaining: List[PendingEntry] = []
        for entry in self.pending_entries:
            if entry.entry_index != index:
                remaining.append(entry)
                continue

            df_symbol = self.data.get(entry.symbol)
            if df_symbol is None or index >= len(df_symbol):
                continue

            entry_bar = df_symbol.iloc[index]
            entry_price = float(entry_bar["open"])
            sl, tp = self._compute_stop_prices(entry.symbol, entry.direction, entry_price, entry.atr_value, entry.sl_mult, entry.tp_mult)

            position = OpenPosition(
                symbol=entry.symbol,
                direction=entry.direction,
                strategy_key=entry.strategy_key,
                strategy_label=entry.strategy_label,
                lot=entry.lot,
                entry_price=entry_price,
                entry_index=index,
                sl=sl,
                tp=tp,
                intrabar_priority=entry.intrabar_priority,
                risk_multiplier=entry.risk_multiplier,
                sl_mult=entry.sl_mult,
                tp_mult=entry.tp_mult,
                atr_value=entry.atr_value,
                    atr_band=entry.atr_band,
                    atr_period=entry.atr_period,
                open_time=df_symbol["time"].iloc[index].to_pydatetime() if isinstance(df_symbol["time"].iloc[index], pd.Timestamp) else df_symbol["time"].iloc[index],
            )

            position_id = self._next_position_id()
            self.open_positions[position_id] = position

        self.pending_entries = remaining

    def _update_positions(self, index: int) -> None:
        if not self.open_positions:
            return

        to_remove: List[int] = []
        for position_id, position in list(self.open_positions.items()):
            if position.status != "open":
                to_remove.append(position_id)
                continue

            df_symbol = self.data.get(position.symbol)
            if df_symbol is None or index >= len(df_symbol):
                continue

            bar = df_symbol.iloc[index]
            high = float(bar["high"])
            low = float(bar["low"])

            exit_price = self._determine_exit_price(position, high, low)
            is_last_bar = index == len(df_symbol) - 1
            if exit_price is None and is_last_bar:
                exit_price = float(bar["close"])

            if exit_price is None:
                position.last_checked_index = index
                continue

            exit_time = df_symbol["time"].iloc[index]
            if isinstance(exit_time, pd.Timestamp):
                exit_time = exit_time.to_pydatetime()

            profit = self._calculate_profit(position.symbol, position.direction, position.entry_price, exit_price, position.lot)

            position.status = "closed"
            position.close_time = exit_time
            position.exit_price = exit_price
            position.pnl = profit
            position.last_checked_index = index

            self.balance += profit
            state = self.strategy_states[position.symbol][position.strategy_key]
            state.performance.record_trade(profit)

            trade = TradeRecord(
                symbol=position.symbol,
                strategy_key=position.strategy_key,
                strategy_label=position.strategy_label,
                direction=position.direction,
                entry_time=position.open_time or df_symbol["time"].iloc[position.entry_index],
                exit_time=exit_time,
                entry_price=position.entry_price,
                exit_price=exit_price,
                lot=position.lot,
                pnl=profit,
                risk_multiplier=position.risk_multiplier,
                sl_mult=position.sl_mult,
                tp_mult=position.tp_mult,
                intrabar_priority=position.intrabar_priority,
                atr_band=position.atr_band,
                atr_period=position.atr_period,
            )
            self.trades.append(trade)
            to_remove.append(position_id)

        for position_id in to_remove:
            self.open_positions.pop(position_id, None)

    def _evaluate_new_signals(self, index: int) -> None:
        for symbol in self.symbols:
            df_symbol = self.data.get(symbol)
            if df_symbol is None or index >= len(df_symbol):
                continue

            bar_time = df_symbol["time"].iloc[index]
            if isinstance(bar_time, pd.Timestamp):
                bar_time = bar_time.to_pydatetime()

            if not should_trade_instrument(symbol, bar_time) and not self._scoring_profile:
                continue

            session_priority = get_instrument_session_priority(symbol, bar_time)
            regime = detect_market_regime(df_symbol.iloc[: index + 1])

            open_buys = any(pos.symbol == symbol and pos.status == "open" and pos.direction == "buy" for pos in self.open_positions.values())
            open_sells = any(pos.symbol == symbol and pos.status == "open" and pos.direction == "sell" for pos in self.open_positions.values())

            atr_cache: Dict[int, Optional[float]] = {}

            def get_atr(period: int) -> Optional[float]:
                if period not in atr_cache:
                    atr_cache[period] = self._get_precomputed_atr(symbol, period, index)
                return atr_cache[period]

            candidates: List[PendingEntry] = []
            symbol_info = self.symbol_info.get(symbol)
            mtf_df = self.higher_timeframe_data.get(symbol, pd.DataFrame())
            micro_df = self.micro_data.get(symbol, pd.DataFrame())
            reference_price = float(df_symbol["close"].iloc[index])

            for strategy_key, state in self.strategy_states[symbol].items():
                definition = state.definition
                if not should_trade_strategy_in_regime(strategy_key, regime) and not self._scoring_profile:
                    continue

                base_atr_value = get_atr(definition.atr_period)
                if base_atr_value is None or base_atr_value <= 0:
                    continue

                atr_value, sl_mult, tp_mult, atr_band, atr_period_used = self._apply_atr_band_adjustment(
                    definition,
                    base_atr_value,
                    get_atr,
                )
                if atr_value is None or atr_value <= 0:
                    continue

                agent_df = self._build_agent_dataframe(df_symbol, index)
                try:
                    signal = state.agent.get_signal(agent_df)
                except Exception as exc:  # pragma: no cover - guard unexpected agent errors
                    print(f"⚠️  {symbol} {definition.label}: failed to compute signal ({exc})")
                    continue

                if signal not in ("buy", "sell"):
                    continue

                if signal == "buy":
                    if open_buys:
                        continue
                    if open_sells and not config.ALLOW_HEDGING:
                        continue
                else:  # sell
                    if open_sells:
                        continue
                    if open_buys and not config.ALLOW_HEDGING:
                        continue

                if not confirm_signal_with_mtf(signal, symbol, bar_time, mtf_df):
                    continue

                micro_result = confirm_with_micro_momentum(
                    symbol=symbol,
                    signal=signal,
                    regime=regime,
                    atr_value=atr_value,
                    reference_price=reference_price,
                    timestamp=bar_time,
                    micro_df=micro_df,
                )
                if not micro_result.passed and not self._scoring_profile:
                    continue

                spread_points = compute_spread_points(symbol_info, df_symbol.iloc[index])
                if spread_points > config.SPREAD_POINTS_LIMIT:
                    continue

                spread_ratio = compute_spread_ratio(spread_points, symbol_info, atr_value)
                if spread_ratio > config.SPREAD_ATR_RATIO_LIMIT:
                    continue

                should_block, correlation_multiplier = should_block_for_correlation(symbol, signal, self.open_positions)
                if should_block and not self._scoring_profile:
                    continue
                if self._scoring_profile and should_block:
                    correlation_multiplier = max(correlation_multiplier, 0.5)

                regime_weight = get_regime_strategy_weight(strategy_key, regime)
                performance_weight = state.performance.performance_weight
                combined_weight = max(1e-6, regime_weight * performance_weight)
                adjusted_priority = definition.priority / combined_weight
                risk_multiplier = calculate_risk_multiplier(session_priority, regime_weight, performance_weight)

                lot = calculate_position_size(
                    symbol_info,
                    atr_value,
                    sl_mult,
                    self.balance,
                    risk_multiplier,
                )

                lot *= correlation_multiplier
                lot = normalize_volume(lot, symbol_info)

                if lot < config.MIN_LOT_SIZE:
                    continue

                entry_index = index + 1
                if entry_index >= len(df_symbol):
                    continue

                candidate = PendingEntry(
                    symbol=symbol,
                    strategy_key=strategy_key,
                    strategy_label=definition.label,
                    direction=signal,
                    entry_index=entry_index,
                    atr_value=atr_value,
                    sl_mult=sl_mult,
                    tp_mult=tp_mult,
                    intrabar_priority=definition.intrabar_priority,
                    risk_multiplier=risk_multiplier,
                    lot=lot,
                    regime=regime,
                    adjusted_priority=adjusted_priority,
                    session_priority=session_priority,
                    micro_score=micro_result.score,
                    micro_threshold=micro_result.threshold,
                    micro_soft_pass=micro_result.soft_pass,
                    atr_band=atr_band,
                    atr_period=atr_period_used,
                )
                candidates.append(candidate)

            if not candidates:
                continue

            best_candidate = min(candidates, key=lambda item: item.adjusted_priority)
            self.pending_entries.append(best_candidate)

    def _build_agent_dataframe(self, source: pd.DataFrame, index: int) -> pd.DataFrame:
        window = source.iloc[: index + 1].copy()
        if "time" in window.columns:
            window = window.copy()
        return window

    def _force_close_remaining_positions(self, index: int) -> None:
        if not self.open_positions:
            return

        df_lookup = {symbol: self.data[symbol] for symbol in self.symbols if symbol in self.data}
        for position_id, position in list(self.open_positions.items()):
            df_symbol = df_lookup.get(position.symbol)
            if df_symbol is None or index >= len(df_symbol):
                continue

            bar = df_symbol.iloc[index]
            exit_price = float(bar["close"])
            exit_time = df_symbol["time"].iloc[index]
            if isinstance(exit_time, pd.Timestamp):
                exit_time = exit_time.to_pydatetime()

            profit = self._calculate_profit(position.symbol, position.direction, position.entry_price, exit_price, position.lot)

            position.status = "closed"
            position.close_time = exit_time
            position.exit_price = exit_price
            position.pnl = profit

            self.balance += profit

            state = self.strategy_states[position.symbol][position.strategy_key]
            state.performance.record_trade(profit)

            trade = TradeRecord(
                symbol=position.symbol,
                strategy_key=position.strategy_key,
                strategy_label=position.strategy_label,
                direction=position.direction,
                entry_time=position.open_time or df_symbol["time"].iloc[position.entry_index],
                exit_time=exit_time,
                entry_price=position.entry_price,
                exit_price=exit_price,
                lot=position.lot,
                pnl=profit,
                risk_multiplier=position.risk_multiplier,
                sl_mult=position.sl_mult,
                tp_mult=position.tp_mult,
                intrabar_priority=position.intrabar_priority,
                atr_band=position.atr_band,
                atr_period=position.atr_period,
            )
            self.trades.append(trade)
            del self.open_positions[position_id]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _next_position_id(self) -> int:
        self.position_counter += 1
        return self.position_counter

    def _compute_stop_prices(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        atr_value: float,
        sl_mult: float,
        tp_mult: float,
    ) -> tuple[Optional[float], Optional[float]]:
        sl = tp = None
        if atr_value and atr_value > 0:
            sl_dist = atr_value * sl_mult
            tp_dist = atr_value * tp_mult
            if direction == "buy":
                sl = entry_price - sl_dist
                tp = entry_price + tp_dist
            else:
                sl = entry_price + sl_dist
                tp = entry_price - tp_dist

            symbol_info = self.symbol_info.get(symbol)
            digits = getattr(symbol_info, "digits", 5) if symbol_info else 5
            if sl is not None:
                sl = round(sl, digits)
            if tp is not None:
                tp = round(tp, digits)
        return sl, tp

    def _determine_exit_price(self, position: OpenPosition, bar_high: float, bar_low: float) -> Optional[float]:
        hit_sl = hit_tp = False
        if position.direction == "buy":
            if position.tp is not None and bar_high >= position.tp:
                hit_tp = True
            if position.sl is not None and bar_low <= position.sl:
                hit_sl = True
        else:  # sell
            if position.tp is not None and bar_low <= position.tp:
                hit_tp = True
            if position.sl is not None and bar_high >= position.sl:
                hit_sl = True

        if hit_sl and hit_tp:
            if position.intrabar_priority.upper() == "TP":
                return position.tp
            return position.sl
        if hit_sl:
            return position.sl
        if hit_tp:
            return position.tp
        return None

    def _calculate_profit(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        exit_price: float,
        lot: float,
    ) -> float:
        symbol_info = self.symbol_info.get(symbol)
        tick_size = getattr(symbol_info, "trade_tick_size", None)
        if not tick_size:
            tick_size = getattr(symbol_info, "point", 0.01) if symbol_info else 0.01
        tick_value = getattr(symbol_info, "trade_tick_value", 1.0) if symbol_info else 1.0

        if direction == "buy":
            price_diff = exit_price - entry_price
        else:
            price_diff = entry_price - exit_price

        ticks = price_diff / tick_size if tick_size else 0.0
        return ticks * tick_value * lot

    # ------------------------------------------------------------------
    # Results aggregation
    # ------------------------------------------------------------------
    def _build_results(self) -> Dict[str, object]:
        per_symbol: Dict[str, Dict[str, Dict[str, float]]] = {}
        per_strategy: Dict[str, List[TradeRecord]] = {}
        band_usage: Dict[str, Dict[str, int]] = {}

        sorted_trades = sorted(self.trades, key=lambda trade: trade.exit_time)
        total_trades = len(sorted_trades)
        winning_trades = sum(1 for trade in sorted_trades if trade.pnl > 0)
        win_rate = (winning_trades / total_trades * 100.0) if total_trades else 0.0

        summary = {
            "initial_balance": self.initial_balance,
            "final_balance": self.balance,
            "total_profit": self.balance - self.initial_balance,
            "max_drawdown": self._calculate_max_drawdown(self.balance_curve),
            "total_trades": total_trades,
            "win_rate": win_rate,
        }

        for trade in sorted_trades:
            per_symbol.setdefault(trade.symbol, {}).setdefault(trade.strategy_key, []).append(trade)
            per_strategy.setdefault(trade.strategy_key, []).append(trade)
            band_usage.setdefault(trade.strategy_key, {})
            band_key = trade.atr_band or "normal"
            band_usage[trade.strategy_key][band_key] = band_usage[trade.strategy_key].get(band_key, 0) + 1

        def summarize(trades: List[TradeRecord]) -> Dict[str, float]:
            total_profit = sum(t.pnl for t in trades)
            count = len(trades)
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate = (wins / count * 100.0) if count else 0.0
            profit_per_trade = total_profit / count if count else 0.0
            max_dd = self._calculate_max_drawdown_from_trades(trades)
            return {
                "trades": count,
                "win_rate": win_rate,
                "total_profit": total_profit,
                "profit_per_trade": profit_per_trade,
                "max_drawdown": max_dd,
            }

        per_symbol_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for symbol, strategy_trades in per_symbol.items():
            per_symbol_summary[symbol] = {
                strategy_key: summarize(trades) for strategy_key, trades in strategy_trades.items()
            }

        per_strategy_summary = {strategy_key: summarize(trades) for strategy_key, trades in per_strategy.items()}

        priority_order = sorted(
            (
                (self._strategy_label_by_key(key), stats["total_profit"])
                for key, stats in per_strategy_summary.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        return {
            "summary": summary,
            "per_symbol": per_symbol_summary,
            "per_strategy": per_strategy_summary,
            "priority_order": priority_order,
            "trades": [trade.__dict__ for trade in sorted_trades],
            "atr_band_usage": band_usage,
        }

    def _strategy_label_by_key(self, strategy_key: str) -> str:
        for definition in self.strategy_definitions:
            if definition.name == strategy_key:
                return definition.label
        return strategy_key

    @staticmethod
    def _calculate_max_drawdown(curve: List[float]) -> float:
        peak = curve[0] if curve else 0.0
        max_dd = 0.0
        for value in curve:
            if value > peak:
                peak = value
            drawdown = peak - value
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd

    @staticmethod
    def _calculate_max_drawdown_from_trades(trades: List[TradeRecord]) -> float:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for trade in trades:
            equity += trade.pnl
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd

    @staticmethod
    def _calculate_max_drawdown_from_pnls(pnls: List[float]) -> float:
        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        for pnl in pnls:
            equity += pnl
            if equity > peak:
                peak = equity
            drawdown = peak - equity
            if drawdown > max_dd:
                max_dd = drawdown
        return max_dd


def _hydrate_trades(trade_dicts: List[Dict[str, object]]) -> List[TradeRecord]:
    hydrated: List[TradeRecord] = []
    for item in trade_dicts:
        exit_time = item.get("exit_time")
        entry_time = item.get("entry_time")
        if isinstance(exit_time, str):
            exit_time = datetime.fromisoformat(exit_time)
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)
        hydrated.append(
            TradeRecord(
                symbol=item.get("symbol", ""),
                strategy_key=item.get("strategy_key", ""),
                strategy_label=item.get("strategy_label", ""),
                direction=item.get("direction", ""),
                entry_time=entry_time or datetime.min,
                exit_time=exit_time or datetime.min,
                entry_price=float(item.get("entry_price", 0.0)),
                exit_price=float(item.get("exit_price", 0.0)),
                lot=float(item.get("lot", 0.0)),
                pnl=float(item.get("pnl", 0.0)),
                risk_multiplier=float(item.get("risk_multiplier", 1.0)),
                sl_mult=float(item.get("sl_mult", 0.0)),
                tp_mult=float(item.get("tp_mult", 0.0)),
                intrabar_priority=str(item.get("intrabar_priority", "TP")),
                atr_band=str(item.get("atr_band", "normal")),
                atr_period=int(item.get("atr_period", 0)),
            )
        )
    hydrated.sort(key=lambda trade: trade.exit_time)
    return hydrated


def _trade_stats(trades: List[TradeRecord]) -> Dict[str, float]:
    if not trades:
        return {
            "total_profit": 0.0,
            "count": 0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
        }
    total_profit = sum(trade.pnl for trade in trades)
    count = len(trades)
    wins = sum(1 for trade in trades if trade.pnl > 0)
    win_rate = (wins / count * 100.0) if count else 0.0
    max_drawdown = BacktestEngine._calculate_max_drawdown_from_pnls([trade.pnl for trade in trades])
    return {
        "total_profit": total_profit,
        "count": count,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
    }


def _score_trade_slice(
    trades: List[TradeRecord],
    weights: OptimizationWeights,
    validation_split: float,
) -> Dict[str, object]:
    if not trades:
        return {
            "score": 0.0,
            "train": _trade_stats([]),
            "validation": _trade_stats([]),
            "worst_drawdown": 0.0,
            "consistency_penalty": 0.0,
            "combined_profit": 0.0,
            "combined_trades": 0,
            "components": {
                "profit": 0.0,
                "drawdown_penalty": 0.0,
                "trade_bonus": 0.0,
                "consistency_penalty": 0.0,
            },
        }

    split_ratio = max(0.0, min(0.9, float(validation_split)))
    split_index = int(len(trades) * (1.0 - split_ratio))
    split_index = min(max(1, split_index), len(trades))

    train_trades = trades[:split_index]
    validation_trades = trades[split_index:]
    if not validation_trades:
        validation_trades = train_trades

    stats_train = _trade_stats(train_trades)
    stats_validation = _trade_stats(validation_trades)

    combined_profit = stats_train["total_profit"] + stats_validation["total_profit"]
    combined_trades = stats_train["count"] + stats_validation["count"]
    worst_drawdown = max(stats_train["max_drawdown"], stats_validation["max_drawdown"])
    consistency_penalty = abs(stats_train["total_profit"] - stats_validation["total_profit"])

    profit_term = weights.profit * combined_profit
    drawdown_term = weights.drawdown * worst_drawdown
    trade_term = weights.trade_count * combined_trades
    consistency_term = weights.consistency * consistency_penalty

    score_value = profit_term - drawdown_term + trade_term - consistency_term

    return {
        "score": score_value,
        "train": stats_train,
        "validation": stats_validation,
        "worst_drawdown": worst_drawdown,
        "consistency_penalty": consistency_penalty,
        "combined_profit": combined_profit,
        "combined_trades": combined_trades,
        "components": {
            "profit": profit_term,
            "drawdown_penalty": drawdown_term,
            "trade_bonus": trade_term,
            "consistency_penalty": consistency_term,
        },
    }


def _compute_score_summary(
    result: Dict[str, object],
    weights: OptimizationWeights,
    validation_split: float,
) -> Dict[str, object]:
    trades_raw = result.get("trades", [])
    if not trades_raw:
        total_profit = result.get("summary", {}).get("total_profit", 0.0)
        max_dd = result.get("summary", {}).get("max_drawdown", 0.0)
        score = (weights.profit * total_profit) - (weights.drawdown * max_dd)
        return {
            "score": score,
            "raw_score": score,
            "train": {
                "total_profit": total_profit,
                "count": result.get("summary", {}).get("total_trades", 0),
                "win_rate": result.get("summary", {}).get("win_rate", 0.0),
                "max_drawdown": max_dd,
            },
            "validation": {
                "total_profit": 0.0,
                "count": 0,
                "win_rate": 0.0,
                "max_drawdown": 0.0,
            },
            "worst_drawdown": max_dd,
            "consistency_penalty": 0.0,
            "combined_profit": total_profit,
            "combined_trades": result.get("summary", {}).get("total_trades", 0),
            "walk_forward": None,
            "components": {
                "profit": weights.profit * total_profit,
                "drawdown_penalty": weights.drawdown * max_dd,
                "trade_bonus": weights.trade_count * result.get("summary", {}).get("total_trades", 0),
                "consistency_penalty": 0.0,
            },
        }

    trades = _hydrate_trades(trades_raw)
    summary = _score_trade_slice(trades, weights, validation_split)
    summary["raw_score"] = summary["score"]
    summary["walk_forward"] = None
    return summary


def _compute_walk_forward_summary(
    trades: List[TradeRecord],
    weights: OptimizationWeights,
    validation_split: float,
    config: WalkForwardConfig,
) -> Optional[Dict[str, object]]:
    if config.segments < 2 or len(trades) < config.min_segment_trades * 2:
        return None

    segment_size = len(trades) // config.segments
    if segment_size < max(1, config.min_segment_trades):
        return None

    segment_summaries: List[Dict[str, object]] = []
    for index in range(config.segments):
        start = index * segment_size
        end = (index + 1) * segment_size if index < config.segments - 1 else len(trades)
        window_trades = trades[start:end]
        if len(window_trades) < config.min_segment_trades:
            continue

        segment_summary = _score_trade_slice(
            window_trades,
            weights,
            config.validation_ratio if config.validation_ratio is not None else validation_split,
        )
        segment_summaries.append(segment_summary)

    if len(segment_summaries) < 2:
        return None

    average_score = mean(segment["score"] for segment in segment_summaries)
    min_score = min(segment["score"] for segment in segment_summaries)
    max_score = max(segment["score"] for segment in segment_summaries)
    dispersion = max_score - min_score

    penalty = config.dispersion_penalty * dispersion
    for segment in segment_summaries:
        if segment["score"] <= 0 or segment["combined_profit"] <= 0:
            penalty += config.negative_penalty * abs(segment["score"])

    composite_score = average_score - penalty

    return {
        "segments": segment_summaries,
        "average_score": average_score,
        "min_score": min_score,
        "max_score": max_score,
        "dispersion": dispersion,
        "penalty": penalty,
        "composite": composite_score,
    }


def _encode_feature_value(value: object) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        lowered = value.lower()
        if lowered == "tp":
            return 1.0
        if lowered == "sl":
            return 0.0
        return sum(ord(ch) for ch in value) / max(1, len(value) * 100.0)
    return 0.0


def _flatten_candidate_features(params: Dict[str, object], atr: Dict[str, object]) -> np.ndarray:
    features: List[float] = []
    for key in sorted(params.keys()):
        features.append(_encode_feature_value(params[key]))
    for key in sorted(atr.keys()):
        features.append(_encode_feature_value(atr[key]))
    return np.array(features, dtype=float)


def optimize_strategy(
    base_engine: BacktestEngine,
    strategy_key: str,
    param_grid: Iterable[Dict[str, float]],
    atr_grid: Iterable[Dict[str, float]],
    top_n: int = 5,
    max_evaluations: Optional[int] = None,
    score_weights: Optional[OptimizationWeights] = None,
    validation_split: float = 0.3,
    walk_forward: Optional[WalkForwardConfig] = None,
    use_bayesian: bool = False,
    initial_random: int = 5,
    bayesian_iterations: Optional[int] = None,
    exploration_weight: float = 0.15,
    random_seed: Optional[int] = None,
) -> tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    weights = score_weights or OptimizationWeights()
    evaluations: List[Dict[str, object]] = []
    evaluation_count = 0

    param_candidates = [dict(candidate) for candidate in param_grid]
    atr_candidates = [dict(candidate) for candidate in atr_grid]

    if not param_candidates or not atr_candidates:
        return [], []

    candidate_pairs: List[Tuple[Dict[str, float], Dict[str, float]]] = []
    candidate_features: List[np.ndarray] = []
    candidate_keys: List[Tuple[tuple, tuple]] = []
    seen_keys: set[Tuple[tuple, tuple]] = set()

    for params in param_candidates:
        for atr_settings in atr_candidates:
            key = (freeze_value(params), freeze_value(atr_settings))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            candidate_pairs.append((params, atr_settings))
            candidate_features.append(_flatten_candidate_features(params, atr_settings))
            candidate_keys.append(key)

    if not candidate_pairs:
        return [], []

    candidate_feature_matrix = np.vstack(candidate_features)
    mins = candidate_feature_matrix.min(axis=0)
    ptp_values = np.ptp(candidate_feature_matrix, axis=0)
    ranges = np.where(ptp_values > 0, ptp_values, 1.0)
    normalized_features = (candidate_feature_matrix - mins) / ranges

    evaluated_lookup: Dict[Tuple[tuple, tuple], Dict[str, object]] = {}

    def evaluate_candidate(params: Dict[str, float], atr_settings: Dict[str, float]) -> Optional[Dict[str, object]]:
        nonlocal evaluation_count
        key = (freeze_value(params), freeze_value(atr_settings))
        if key in evaluated_lookup:
            return evaluated_lookup[key]
        if max_evaluations and evaluation_count >= max_evaluations:
            return None

        new_definitions = []
        for definition in base_engine.strategy_definitions:
            if definition.name != strategy_key:
                new_definitions.append(replace(definition))
                continue

            merged_params = definition.params.copy()
            merged_params.update(params)

            new_definitions.append(
                replace(
                    definition,
                    params=merged_params,
                    sl_mult=float(atr_settings.get("sl_mult", definition.sl_mult)),
                    tp_mult=float(atr_settings.get("tp_mult", definition.tp_mult)),
                    intrabar_priority=str(atr_settings.get("priority", definition.intrabar_priority)),
                    atr_period=int(atr_settings.get("period", definition.atr_period)),
                )
            )

        clone = BacktestEngine.from_existing(base_engine, strategy_definitions=new_definitions)
        clone.set_scoring_profile(True)
        result = clone.run()
        summary = result["summary"]

        score_summary = _compute_score_summary(result, weights, validation_split)
        hydrated_trades: List[TradeRecord] = []
        if result.get("trades"):
            hydrated_trades = _hydrate_trades(result["trades"])

        if walk_forward and hydrated_trades:
            wf_summary = _compute_walk_forward_summary(
                hydrated_trades,
                weights,
                validation_split,
                walk_forward,
            )
            if wf_summary:
                base_score_value = score_summary.get("raw_score", score_summary["score"])
                composite_score = (
                    (1.0 - walk_forward.stability_weight) * base_score_value
                    + walk_forward.stability_weight * wf_summary["composite"]
                )
                score_summary["walk_forward"] = wf_summary
                score_summary["score"] = composite_score
            else:
                score_summary["walk_forward"] = None

        evaluation = {
            "params": dict(params),
            "atr": dict(atr_settings),
            "total_profit": summary["total_profit"],
            "win_rate": summary["win_rate"],
            "total_trades": summary["total_trades"],
            "result": result,
            "score": score_summary,
        }
        evaluations.append(evaluation)
        evaluated_lookup[key] = evaluation
        evaluation_count += 1
        return evaluation

    total_candidates = len(candidate_pairs)
    max_allowed = total_candidates if max_evaluations is None else min(total_candidates, max_evaluations)

    if not use_bayesian or max_allowed <= 1:
        for params, atr_settings in candidate_pairs:
            if max_evaluations and evaluation_count >= max_evaluations:
                break
            evaluate_candidate(params, atr_settings)
        evaluations.sort(key=lambda item: item["score"]["score"], reverse=True)
        return evaluations[:top_n], evaluations

    rng = random.Random(random_seed or 42)
    initial_count = max(1, min(initial_random, max_allowed))
    initial_indices = rng.sample(range(total_candidates), initial_count)

    evaluated_indices: set[int] = set()
    evaluated_feature_list: List[np.ndarray] = []
    evaluated_score_list: List[float] = []

    def record_evaluation(idx: int) -> None:
        if idx in evaluated_indices:
            return
        params, atr_settings = candidate_pairs[idx]
        evaluation = evaluate_candidate(params, atr_settings)
        if evaluation is None:
            return
        evaluated_indices.add(idx)
        evaluated_feature_list.append(normalized_features[idx])
        evaluated_score_list.append(evaluation["score"]["score"])

    for idx in initial_indices:
        if max_evaluations and evaluation_count >= max_evaluations:
            break
        record_evaluation(idx)

    remaining_capacity = max_allowed - len(evaluated_indices)
    if remaining_capacity <= 0:
        evaluations.sort(key=lambda item: item["score"]["score"], reverse=True)
        return evaluations[:top_n], evaluations

    if bayesian_iterations is None:
        bayesian_limit = remaining_capacity
    else:
        bayesian_limit = min(remaining_capacity, max(0, bayesian_iterations))

    if len(evaluated_feature_list) > 1:
        sample_indices = rng.sample(range(len(evaluated_feature_list)), min(10, len(evaluated_feature_list)))
        distances = []
        for i in range(len(sample_indices)):
            for j in range(i + 1, len(sample_indices)):
                dist = np.linalg.norm(
                    evaluated_feature_list[sample_indices[i]] - evaluated_feature_list[sample_indices[j]]
                )
                if dist > 0:
                    distances.append(dist)
        median_distance = float(np.median(distances)) if distances else 1.0
    else:
        median_distance = 1.0

    gamma = 1.0 / max(1e-6, 2.0 * (median_distance ** 2))

    additional_evals = 0
    while additional_evals < bayesian_limit and len(evaluated_indices) < max_allowed:
        if max_evaluations and evaluation_count >= max_evaluations:
            break

        evaluated_feature_matrix = np.vstack(evaluated_feature_list) if evaluated_feature_list else np.empty((0, normalized_features.shape[1]))
        evaluated_scores_array = np.array(evaluated_score_list) if evaluated_score_list else np.array([])

        best_idx = None
        best_metric = -float("inf")

        for idx, feature in enumerate(normalized_features):
            if idx in evaluated_indices:
                continue

            if evaluated_feature_matrix.size == 0:
                metric = rng.random()
            else:
                distances = np.linalg.norm(evaluated_feature_matrix - feature, axis=1)
                kernel_weights = np.exp(-gamma * (distances ** 2))
                weight_sum = kernel_weights.sum()
                if weight_sum <= 1e-9:
                    predicted = float(evaluated_scores_array.mean()) if evaluated_scores_array.size else 0.0
                else:
                    predicted = float((kernel_weights * evaluated_scores_array).sum() / weight_sum)
                exploration = float(distances.max()) if distances.size else 0.0
                metric = predicted + exploration_weight * exploration

            if metric > best_metric:
                best_metric = metric
                best_idx = idx

        if best_idx is None:
            break

        record_evaluation(best_idx)
        additional_evals += 1

        evaluated_feature_matrix = np.vstack(evaluated_feature_list)
        evaluated_scores_array = np.array(evaluated_score_list)
        if evaluated_feature_matrix.shape[0] > 1:
            distances = []
            sample_pool = min(15, evaluated_feature_matrix.shape[0])
            sample_ids = rng.sample(range(evaluated_feature_matrix.shape[0]), sample_pool)
            for i in range(len(sample_ids)):
                for j in range(i + 1, len(sample_ids)):
                    dist = np.linalg.norm(
                        evaluated_feature_matrix[sample_ids[i]] - evaluated_feature_matrix[sample_ids[j]]
                    )
                    if dist > 0:
                        distances.append(dist)
            if distances:
                median_distance = float(np.median(distances))
                gamma = 1.0 / max(1e-6, 2.0 * (median_distance ** 2))

    evaluations.sort(key=lambda item: item["score"]["score"], reverse=True)
    return evaluations[:top_n], evaluations


def run_default_backtest() -> Dict[str, object]:
    engine = BacktestEngine()
    return engine.run()


if __name__ == "__main__":
    results = run_default_backtest()
    summary = results["summary"]
    print("\n==== Ultima Strategy Backtest Summary ====")
    print(f"Initial balance: {summary['initial_balance']:.2f}")
    print(f"Final balance:   {summary['final_balance']:.2f}")
    print(f"Total profit:    {summary['total_profit']:.2f}")
    print(f"Max drawdown:    {summary['max_drawdown']:.2f}")
    print(f"Trades taken:    {summary['total_trades']}")

    print("\nStrategy priority (by aggregated profit):")
    for rank, (label, profit) in enumerate(results.get("priority_order", []), start=1):
        print(f"  {rank}. {label} -> {profit:.2f}")

import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
import pytz
from agents import TradingAgent
from itertools import product
from time import time
from functools import lru_cache

# Settings
timeframe = mt5.TIMEFRAME_M15
symbols = ["EURUSD+", "USDJPY+", "GBPUSD+", "GBPJPY+", "XAUUSD+"]  # Add more as needed
start_date = datetime.now() - timedelta(days=60)  # 60 days backtest
end_date = datetime.now()

# Enhanced risk management constants (institutional-level)
ACCOUNT_RISK_PER_TRADE = 0.30  # 30% risk per trade for ultra-aggressive research runs
DEFAULT_ACCOUNT_BALANCE = 1000  # Default balance for backtesting if not specified
MIN_LOT_SIZE = 0.01
MAX_LOT_SIZE = 5.0
RISK_MULTIPLIER_MIN = 1.0
RISK_MULTIPLIER_MAX = 3.5

# ATR exit settings (default fallback; individual strategies override below)
ATR_PERIOD = 5
SL_ATR_MULTIPLIER = 2.0
TP_ATR_MULTIPLIER = 2.0
INTRABAR_PRIORITY = 'TP'  # 'SL' or 'TP' if both hit within the same bar

# Strategy parameters
ma_crossover_fast = 5
ma_crossover_slow = 40
meanrev_ma_period = 10
meanrev_num_std = 1
momentum_ma_period = 30
momentum_roc_period = 5
breakout_lookback = 10
donchian_channel_length = 10

STRATEGY_ATR_CONFIG = {
    'ma_crossover': {'period': 5, 'sl_mult': 2.0, 'tp_mult': 2.0, 'priority': 'TP'},
    'mean_reversion': {'period': 7, 'sl_mult': 2.75, 'tp_mult': 4.0, 'priority': 'TP'},
    'momentum_trend': {'period': 5, 'sl_mult': 2.75, 'tp_mult': 2.0, 'priority': 'TP'},
    'breakout': {'period': 14, 'sl_mult': 2.5, 'tp_mult': 3.0, 'priority': 'TP'},
    'donchian_channel': {'period': 5, 'sl_mult': 3.0, 'tp_mult': 3.5, 'priority': 'TP'},
}

DEFAULT_ATR_SETTINGS = {
    'period': ATR_PERIOD,
    'sl_mult': SL_ATR_MULTIPLIER,
    'tp_mult': TP_ATR_MULTIPLIER,
    'priority': INTRABAR_PRIORITY,
}


def calculate_position_size(
    symbol_info,
    atr_value: float,
    sl_mult: float,
    account_balance: float | None = None,
    risk_multiplier: float = 1.0,
) -> float:
    """Calculate position size based on volatility, risk limits, and dynamic multipliers."""
    if account_balance is None or account_balance <= 0:
        account_balance = DEFAULT_ACCOUNT_BALANCE

    if not atr_value or atr_value <= 0:
        # Fallback to conservative fixed sizing if no ATR
        if hasattr(symbol_info, "name") and "XAUUSD" in str(symbol_info.name).upper():
            return 0.1
        return 0.5

    bounded_multiplier = max(RISK_MULTIPLIER_MIN, min(RISK_MULTIPLIER_MAX, risk_multiplier))

    # Risk amount in account currency
    risk_amount = account_balance * ACCOUNT_RISK_PER_TRADE * bounded_multiplier

    # Expected loss per unit if SL is hit
    sl_distance = atr_value * sl_mult
    base_tick_size = getattr(symbol_info, "point", 0.01) if hasattr(symbol_info, "point") else 0.01
    tick_size = getattr(symbol_info, "trade_tick_size", base_tick_size) or base_tick_size
    tick_value = getattr(symbol_info, "trade_tick_value", 1.0) or 1.0

    # Convert SL distance to account currency
    ticks_at_risk = sl_distance / tick_size
    loss_per_unit = ticks_at_risk * tick_value

    if loss_per_unit <= 0:
        return MIN_LOT_SIZE

    # Calculate optimal position size
    optimal_size = risk_amount / loss_per_unit

    # Apply limits
    position_size = max(MIN_LOT_SIZE, min(optimal_size, MAX_LOT_SIZE))

    return round(position_size, 2)


def compute_atr(df: pd.DataFrame, period: int = ATR_PERIOD):
    """Compute ATR on completed bars and return the last ATR value (float) or None if insufficient data."""
    try:
        if df is None or len(df) < period + 2:
            return None
        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr_series = tr.rolling(period).mean()
        atr_val = atr_series.iloc[-2]  # use last completed bar
        if pd.isna(atr_val):
            return None
        return float(atr_val)
    except Exception:
        return None


def is_market_session_active(timestamp: datetime) -> bool:
    """Check if timestamp is in an active trading session for backtesting."""
    try:
        # Convert to UTC if not already
        if timestamp.tzinfo is None:
            utc_time = pytz.UTC.localize(timestamp)
        else:
            utc_time = timestamp.astimezone(pytz.UTC)
        
        current_hour = utc_time.hour
        current_weekday = utc_time.weekday()  # 0=Monday, 6=Sunday
        
        # Avoid weekends
        if current_weekday == 6:  # Sunday
            return current_hour >= 22
        elif current_weekday == 5:  # Saturday
            return False
        elif current_weekday == 4 and current_hour >= 22:  # Friday after 22:00
            return False
        
        # Avoid dead zone: 07:00-08:00 UTC
        if 7 <= current_hour < 8:
            return False
        
        return True
        
    except Exception:
        return True  # Default to allow trading if error


# Connect to MT5
if not mt5.initialize():
    print("initialize() failed")
    quit()

def get_data(symbol):
    rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
    if rates is None:
        print(f"No data for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df


symbol_data = {symbol: get_data(symbol) for symbol in symbols}
symbol_specs = {}

_symbol_atr_cache: Dict[str, Dict[int, pd.Series]] = {}
_symbol_tr_cache: Dict[str, pd.Series] = {}


def _get_cached_symbol_atr(symbol: str, period: int) -> Optional[pd.Series]:
    period = int(period)
    cache = _symbol_atr_cache.setdefault(symbol, {})
    if period in cache:
        return cache[period]

    df = symbol_data.get(symbol)
    if df is None or df.empty or period <= 0:
        cache[period] = None
        return None

    try:
        if symbol not in _symbol_tr_cache:
            high = df['high']
            low = df['low']
            close = df['close']
            prev_close = close.shift(1)
            tr = pd.concat([
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs()
            ], axis=1).max(axis=1)
            _symbol_tr_cache[symbol] = tr
        else:
            tr = _symbol_tr_cache[symbol]
        atr_series = tr.rolling(period).mean()
    except Exception:
        cache[period] = None
        return None

    cache[period] = atr_series
    return atr_series

results = {}
agent = TradingAgent(initial_balance=10000)
# Inject ATR/exit params into agent (a simple way to pass defaults)
agent._atr_period = ATR_PERIOD
agent._sl_mult = SL_ATR_MULTIPLIER
agent._tp_mult = TP_ATR_MULTIPLIER
agent._intrabar_priority = INTRABAR_PRIORITY
min_lookback = max(ma_crossover_slow, meanrev_ma_period, momentum_ma_period, breakout_lookback, donchian_channel_length)

for symbol in symbols:
    df = symbol_data.get(symbol)
    if df is None:
        print(f"No data for {symbol}")
        continue
    if len(df) <= min_lookback:
        print(f"Not enough data for {symbol}")
        continue

    # Pull tick specs from MT5 for realistic PnL scaling
    sinfo = mt5.symbol_info(symbol)
    if sinfo is not None:
        tick_size = getattr(sinfo, 'trade_tick_size', sinfo.point if hasattr(sinfo, 'point') else 0.01) or 0.01
        tick_value = getattr(sinfo, 'trade_tick_value', 1.0) or 1.0
        lots = getattr(sinfo, 'trade_lots', None)
        agent._tick_size = tick_size
        agent._tick_value = tick_value
        
        # Dynamic position sizing based on volatility
        atr_data = compute_atr(df.copy())
        atr_value = atr_data if atr_data and atr_data > 0 else 0.001
        sl_mult = SL_ATR_MULTIPLIER  # Default multiplier
        agent._lots = calculate_position_size(sinfo, atr_value, sl_mult)
        symbol_specs[symbol] = {
            'tick_size': tick_size,
            'tick_value': tick_value,
            'lots': agent._lots,
        }

    def run_with_atr(strategy_key, func, *args):
        cfg = STRATEGY_ATR_CONFIG.get(strategy_key, DEFAULT_ATR_SETTINGS)
        agent._atr_period = cfg['period']
        agent._sl_mult = cfg['sl_mult']
        agent._tp_mult = cfg['tp_mult']
        agent._intrabar_priority = cfg.get('priority', INTRABAR_PRIORITY)
        return func(df.copy(), *args)

    res_ma = run_with_atr('ma_crossover', agent.ma_crossover, ma_crossover_fast, ma_crossover_slow)
    res_mr = run_with_atr('mean_reversion', agent.mean_reversion, meanrev_ma_period, meanrev_num_std)
    res_mom = run_with_atr('momentum_trend', agent.momentum_trend, momentum_ma_period, momentum_roc_period)
    res_brk = run_with_atr('breakout', agent.breakout, breakout_lookback)
    res_don = run_with_atr('donchian_channel', agent.donchian_channel, donchian_channel_length)
    results[symbol] = {
        'ma_crossover': res_ma,
        'mean_reversion': res_mr,
        'momentum_trend': res_mom,
        'breakout': res_brk,
        'donchian_channel': res_don
    }
    print(f"{symbol} MA Crossover: Trades={res_ma['trades']}, WinRate={res_ma['win_rate']:.2f}%, Profit={res_ma['total_profit']:.2f}")
    print(f"{symbol} Mean Reversion: Trades={res_mr['trades']}, WinRate={res_mr['win_rate']:.2f}%, Profit={res_mr['total_profit']:.2f}")
    print(f"{symbol} Momentum/Trend: Trades={res_mom['trades']}, WinRate={res_mom['win_rate']:.2f}%, Profit={res_mom['total_profit']:.2f}")
    print(f"{symbol} Breakout: Trades={res_brk['trades']}, WinRate={res_brk['win_rate']:.2f}%, Profit={res_brk['total_profit']:.2f}")
    print(f"{symbol} Donchian Channel: Trades={res_don['trades']}, WinRate={res_don['win_rate']:.2f}%, Profit={res_don['total_profit']:.2f}")


# # Summary statistics
# def print_summary(results, strategy):
#     total_profit = sum(r[strategy]['total_profit'] for r in results.values())
#     total_trades = sum(r[strategy]['trades'] for r in results.values())
#     total_win_trades = sum(
#         r[strategy]['trades'] * r[strategy]['win_rate'] / 100 for r in results.values()
#     )
#     avg_winrate = (total_win_trades / total_trades * 100) if total_trades else 0
#     profit_ratio = (total_profit / total_trades) if total_trades else 0
#     max_drawdown = max((r[strategy]['max_drawdown'] for r in results.values()), default=0)
#     final_balance = sum(r[strategy]['final_balance'] for r in results.values())
#     print(f"\n{strategy.replace('_', ' ').title()} SUMMARY:")
#     print(f"Initial Balance (per symbol): 10000")
#     print(f"Total Final Balance: {final_balance:.2f}")
#     print(f"Total Profit: {total_profit:.2f}")
#     print(f"Total Trades: {total_trades}")
#     print(f"Total Winrate: {avg_winrate:.2f}%")
#     print(f"Profit Ratio (Profit/Trade): {profit_ratio:.4f}")
#     print(f"Max Drawdown (worst symbol): {max_drawdown:.2f}")


# Print per-symbol statistics
def print_per_symbol_stats(results, strategy):
    print(f"\n{strategy.replace('_', ' ').title()} Per-Symbol Statistics:")
    print(f"{'Symbol':<8} {'Trades':>6} {'WinRate%':>9} {'Profit':>10} {'FinalBal':>10} {'MaxDD':>10}")
    for symbol, res in results.items():
        r = res[strategy]
        print(f"{symbol:<8} {r['trades']:>6} {r['win_rate']:>9.2f} {r['total_profit']:>10.2f} {r['final_balance']:>10.2f} {r['max_drawdown']:>10.2f}")


# Save per-symbol stats to CSV

# def save_per_symbol_stats_to_excel(results, strategy, filename):
#     import pandas as pd
#     data = []
#     for symbol, res in results.items():
#         r = res[strategy]
#         data.append({
#             'Symbol': symbol,
#             'Trades': r['trades'],
#             'WinRate': round(r['win_rate'], 2),
#             'Profit': round(r['total_profit'], 2),
#             'FinalBalance': round(r['final_balance'], 2),
#             'MaxDrawdown': round(r['max_drawdown'], 2)
#         })
#     df = pd.DataFrame(data)
#     df.to_excel(filename, index=False)

# save_per_symbol_stats_to_excel(results, 'ma_crossover', 'ma_crossover_stats.xlsx')
# save_per_symbol_stats_to_excel(results, 'mean_reversion', 'mean_reversion_stats.xlsx')
# save_per_symbol_stats_to_excel(results, 'momentum_trend', 'momentum_trend_stats.xlsx')
# save_per_symbol_stats_to_excel(results, 'breakout', 'breakout_stats.xlsx')
# save_per_symbol_stats_to_excel(results, 'donchian_channel', 'donchian_channel_stats.xlsx')

print_per_symbol_stats(results, 'ma_crossover')
print_per_symbol_stats(results, 'mean_reversion')
print_per_symbol_stats(results, 'momentum_trend')
print_per_symbol_stats(results, 'breakout')
print_per_symbol_stats(results, 'donchian_channel')

# print_summary(results, 'ma_crossover')
# print_summary(results, 'mean_reversion')
# print_summary(results, 'momentum_trend')
# print_summary(results, 'breakout')


def freeze_value(value):
    if isinstance(value, dict):
        return 'dict', tuple(sorted((k, freeze_value(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return 'tuple', tuple(freeze_value(v) for v in value)
    return 'scalar', value


def thaw_value(frozen):
    kind, data = frozen
    if kind == 'dict':
        return {k: thaw_value(v) for k, v in data}
    if kind == 'tuple':
        return tuple(thaw_value(v) for v in data)
    return data


def run_strategy(agent, strategy, df, params):
    local_df = df.copy()
    if strategy == 'ma_crossover':
        fast, slow = params
        return agent.ma_crossover(local_df, fast, slow)
    if strategy == 'mean_reversion':
        ma_period, num_std = params
        return agent.mean_reversion(local_df, ma_period, num_std)
    if strategy == 'momentum_trend':
        ma_period, roc_period = params
        return agent.momentum_trend(local_df, ma_period, roc_period)
    if strategy == 'breakout':
        lookback = params
        return agent.breakout(local_df, lookback)
    if strategy == 'donchian_channel':
        channel_length = params
        return agent.donchian_channel(local_df, channel_length)
    raise ValueError(f"Unsupported strategy '{strategy}'")


@lru_cache(maxsize=None)
def evaluate_symbol_strategy(strategy, params_key, atr_key, symbol):
    df = symbol_data.get(symbol)
    if df is None or df.empty:
        return None

    params = thaw_value(params_key)
    atr_settings = thaw_value(atr_key)

    agent = TradingAgent(initial_balance=10000)
    agent._atr_period = atr_settings.get('period', ATR_PERIOD)
    agent._sl_mult = atr_settings.get('sl_mult', SL_ATR_MULTIPLIER)
    agent._tp_mult = atr_settings.get('tp_mult', TP_ATR_MULTIPLIER)
    agent._intrabar_priority = atr_settings.get('priority', INTRABAR_PRIORITY)

    specs = symbol_specs.get(symbol)
    if specs:
        agent._tick_size = specs['tick_size']
        agent._tick_value = specs['tick_value']
        agent._lots = specs['lots']

    atr_period = int(agent._atr_period)
    precomputed_atr = _get_cached_symbol_atr(symbol, atr_period)
    if precomputed_atr is not None:
        agent._precomputed_atr_cache = {atr_period: precomputed_atr}
    else:
        agent._precomputed_atr_cache = {}

    return run_strategy(agent, strategy, df, params)


def aggregate_strategy_metrics(strategy, params, atr_settings):
    params_key = freeze_value(params)
    atr_key = freeze_value(atr_settings)
    total_profit = 0.0
    total_trades = 0
    win_trades = 0.0
    max_drawdown = 0.0
    per_symbol = {}

    for symbol in symbols:
        res = evaluate_symbol_strategy(strategy, params_key, atr_key, symbol)
        if not res:
            continue
        total_profit += res['total_profit']
        total_trades += res['trades']
        win_trades += res['trades'] * res['win_rate'] / 100.0
        max_drawdown = max(max_drawdown, res['max_drawdown'])
        per_symbol[symbol] = res

    if total_trades == 0:
        avg_win_rate = 0.0
        profit_per_trade = 0.0
    else:
        avg_win_rate = (win_trades / total_trades) * 100.0
        profit_per_trade = total_profit / total_trades

    return {
        'strategy': strategy,
        'params': params if not isinstance(params, dict) else dict(params),
        'atr': dict(atr_settings),
        'total_profit': total_profit,
        'total_trades': total_trades,
        'avg_win_rate': avg_win_rate,
        'profit_per_trade': profit_per_trade,
        'max_drawdown': max_drawdown,
        'per_symbol': per_symbol,
    }


def _convert_params_for_optimizer(strategy: str, params) -> Dict[str, float]:
    if isinstance(params, dict):
        return dict(params)
    if strategy == 'ma_crossover':
        fast, slow = params
        return {"fast_period": int(fast), "slow_period": int(slow)}
    if strategy == 'mean_reversion':
        ma_period, num_std = params
        return {"ma_period": int(ma_period), "num_std": float(num_std)}
    if strategy == 'momentum_trend':
        ma_period, roc_period = params
        return {"ma_period": int(ma_period), "roc_period": int(roc_period)}
    if strategy == 'breakout':
        return {"lookback": int(params)}
    if strategy == 'donchian_channel':
        return {"channel_length": int(params)}
    raise ValueError(f"Unsupported strategy '{strategy}' for optimizer conversion")


def refine_parameter_grid(strategy, params):
    refined = set()
    if strategy == 'ma_crossover':
        fast, slow = params
        for f in range(max(2, fast - 5), fast + 6, 5):
            for s in range(max(f + 5, slow - 10), slow + 11, 5):
                if f < s:
                    refined.add((f, s))
    elif strategy == 'mean_reversion':
        ma_period, num_std = params
        ma_candidates = range(max(5, ma_period - 5), ma_period + 6, 5)
        std_candidates = [max(0.5, num_std - 0.5), num_std, num_std + 0.5]
        for m in ma_candidates:
            for n in std_candidates:
                refined.add((m, round(n, 2)))
    elif strategy == 'momentum_trend':
        ma_period, roc_period = params
        ma_candidates = range(max(20, ma_period - 10), ma_period + 11, 10)
        roc_candidates = range(max(2, roc_period - 5), roc_period + 6, 5)
        for m in ma_candidates:
            for r in roc_candidates:
                refined.add((m, r))
    elif strategy in ('breakout', 'donchian_channel'):
        lookback = params
        for lb in range(max(5, lookback - 5), lookback + 6, 5):
            refined.add(lb)
    return refined


def refine_atr_grid(atr_settings):
    base_period = atr_settings.get('period', ATR_PERIOD)
    base_sl = atr_settings.get('sl_mult', SL_ATR_MULTIPLIER)
    base_tp = atr_settings.get('tp_mult', TP_ATR_MULTIPLIER)
    base_priority = atr_settings.get('priority', INTRABAR_PRIORITY)

    periods = {base_period}
    for delta in (-2, 2):
        candidate = base_period + delta
        if candidate >= 3:
            periods.add(candidate)

    sl_values = {round(base_sl + delta, 2) for delta in (-0.5, 0.0, 0.5) if base_sl + delta >= 0.5}
    tp_values = {round(base_tp + delta, 2) for delta in (-0.5, 0.0, 0.5) if base_tp + delta >= 0.5}
    priorities = {base_priority}

    return [
        {
            'period': period,
            'sl_mult': sl,
            'tp_mult': tp,
            'priority': priority,
        }
        for period in periods
        for sl in sl_values
        for tp in tp_values
        for priority in priorities
    ]


def search_best_combinations(strategy, param_grid, atr_grid, top_n=5, refine_top=3, max_evaluations=500):
    evaluated = set()
    all_results = []
    eval_count = 0
    dots_cycle = ['.', '..', '...', '....']
    status_index = 0
    last_status_time = 0.0

    def evaluate_combo(params, atr):
        frozen_key = (freeze_value(params), freeze_value(atr))
        if frozen_key in evaluated:
            return None
        nonlocal eval_count
        if max_evaluations and eval_count >= max_evaluations:
            return None
        evaluated.add(frozen_key)
        metrics = aggregate_strategy_metrics(strategy, params, atr)
        if metrics['total_trades'] == 0:
            return None
        all_results.append(metrics)
        eval_count += 1
        emit_progress()
        return metrics

    def emit_progress(force=False):
        nonlocal status_index, last_status_time
        now = time()
        if force or now - last_status_time >= 1.5:
            msg = f"  testing{dots_cycle[status_index % len(dots_cycle)]}"
            print(msg.ljust(20), end='\r', flush=True)
            status_index += 1
            last_status_time = now

    for params in param_grid:
        for atr in atr_grid:
            evaluate_combo(params, atr)
            if max_evaluations and eval_count >= max_evaluations:
                break
        if max_evaluations and eval_count >= max_evaluations:
            break

    all_results.sort(key=lambda x: x['total_profit'], reverse=True)

    if refine_top and all_results:
        seeds = all_results[:min(refine_top, len(all_results))]
        refined_params = set()
        refined_atrs = set()
        for seed in seeds:
            refined_params.update(refine_parameter_grid(strategy, seed['params']))
            for refined_atr in refine_atr_grid(seed['atr']):
                refined_atrs.add(tuple(sorted(refined_atr.items())))

        refined_atr_dicts = [dict(items) for items in refined_atrs] if refined_atrs else []

        for params in refined_params:
            atr_candidates = refined_atr_dicts if refined_atr_dicts else atr_grid
            for atr in atr_candidates:
                evaluate_combo(params, atr)
                if max_evaluations and eval_count >= max_evaluations:
                    break
            if max_evaluations and eval_count >= max_evaluations:
                break

        all_results.sort(key=lambda x: x['total_profit'], reverse=True)

    emit_progress(force=True)
    print("  testing... done!".ljust(20))

    top_results = all_results[:min(top_n, len(all_results))]
    return top_results, all_results, eval_count

# Parameter grids
ma_crossover_fast_values = [5, 10, 15, 20]
ma_crossover_slow_values = [30, 40, 50, 60, 80, 100]
ma_crossover_grid = [(f, s) for f, s in product(ma_crossover_fast_values, ma_crossover_slow_values) if f < s]

meanrev_ma_values = [10, 15, 20, 25, 30]
meanrev_std_values = [1.0, 1.5, 2.0]
meanrev_grid = [(p, round(n, 2)) for p, n in product(meanrev_ma_values, meanrev_std_values)]

momentum_ma_values = [30, 40, 50, 60, 70, 80, 90, 100]
momentum_roc_values = [5, 10, 15, 20, 25, 30]
momentum_grid = [(p, r) for p, r in product(momentum_ma_values, momentum_roc_values)]

breakout_grid = list(range(10, 61, 5))
donchian_grid = list(range(10, 61, 5))

# ATR parameter grids for fine tuning
atr_period_grid = [5, 7, 10, 14, 21]
sl_mult_grid = [1.5, 2.0, 2.25, 2.5, 2.75, 3.0]
tp_mult_grid = [2.0, 2.5, 3.0, 3.5, 4.0]
priority_grid = ['SL', 'TP']

strategy_labels = {
    'ma_crossover': 'MA Crossover',
    'mean_reversion': 'Mean Reversion',
    'momentum_trend': 'Momentum Trend',
    'breakout': 'Breakout',
    'donchian_channel': 'Donchian Channel',
}

optimizer_engine = BacktestEngine(start_date=start_date, end_date=end_date)
optimizer_weights = OptimizationWeights()
walk_forward_config = WalkForwardConfig()
optimizer_validation_split = 0.3

try:
    optimizer_engine._connection_owned = False
    optimizer_engine._load_market_data()
    optimizer_engine._data_loaded = True
except Exception as exc:  # pragma: no cover - best-effort warmup
    print(f"⚠️  Optimizer warmup failed: {exc}")

print("\nImproved parameter sweep results (top combinations by total profit):")
top_results_per_strategy = {}
best_profit_map = {}

base_atr_grid = [DEFAULT_ATR_SETTINGS]
joint_atr_grid = [
    {'period': ap, 'sl_mult': sl, 'tp_mult': tp, 'priority': prio}
    for ap in atr_period_grid
    for sl in sl_mult_grid
    for tp in tp_mult_grid
    for prio in priority_grid
]

for strategy, grid in [
    ('ma_crossover', ma_crossover_grid),
    ('mean_reversion', meanrev_grid),
    ('momentum_trend', momentum_grid),
    ('breakout', breakout_grid),
    ('donchian_channel', donchian_grid)
]:
    strategy_name = strategy_labels[strategy]
    print(f"\n{strategy_name}:")

    top_basic, _, basic_evals = search_best_combinations(
        strategy,
        grid,
        base_atr_grid,
        top_n=5,
        refine_top=3,
        max_evaluations=150,
    )
    if top_basic:
        best_basic = top_basic[0]
        params = best_basic['params']
        print("  Best default ATR params:")
        print(
            f"    params={params}, profit={best_basic['total_profit']:.2f}, "
            f"trades={best_basic['total_trades']}, win%={best_basic['avg_win_rate']:.2f}, "
            f"maxDD={best_basic['max_drawdown']:.2f}"
        )
    else:
        print("  No viable default ATR params found.")

    top_joint, all_joint, joint_evals = search_best_combinations(
        strategy,
        grid,
        joint_atr_grid,
        top_n=5,
        refine_top=5,
        max_evaluations=350,
    )
    if top_joint:
        best_joint = top_joint[0]
        params = best_joint['params']
        atr = best_joint['atr']
        print("  Best parameter + ATR combination:")
        print(
            f"    params={params}, ATR={atr}, profit={best_joint['total_profit']:.2f}, "
            f"trades={best_joint['total_trades']}, win%={best_joint['avg_win_rate']:.2f}, "
            f"maxDD={best_joint['max_drawdown']:.2f}"
        )

        try:
            optimized_params = _convert_params_for_optimizer(strategy, params)
            optimized_atr = dict(atr)
            top_eval, _ = optimize_strategy(
                optimizer_engine,
                strategy,
                [optimized_params],
                [optimized_atr],
                top_n=1,
                max_evaluations=1,
                score_weights=optimizer_weights,
                validation_split=optimizer_validation_split,
                walk_forward=walk_forward_config,
            )
            if top_eval:
                score_block = top_eval[0].get("score", {})
                raw_score = score_block.get("raw_score", score_block.get("score", 0.0))
                final_score = score_block.get("score", 0.0)
                walk_forward_block = score_block.get("walk_forward")
                components_block = score_block.get("components", {})
                profit_term = components_block.get("profit", 0.0)
                drawdown_term = components_block.get("drawdown_penalty", 0.0)
                trade_term = components_block.get("trade_bonus", 0.0)
                consistency_term = components_block.get("consistency_penalty", 0.0)
                combined_profit = score_block.get("combined_profit", 0.0)
                combined_trades = score_block.get("combined_trades", 0)
                worst_dd = score_block.get("worst_drawdown", 0.0)
                if walk_forward_block:
                    print(
                        "      ↳ score={:.2f} (raw={:.2f}, wf_avg={:.2f}, wf_min={:.2f}, dispersion={:.2f}, penalty={:.2f})".format(
                            final_score,
                            raw_score,
                            walk_forward_block.get("average_score", 0.0),
                            walk_forward_block.get("min_score", 0.0),
                            walk_forward_block.get("dispersion", 0.0),
                            walk_forward_block.get("penalty", 0.0),
                        )
                    )
                else:
                    print("      ↳ score={:.2f} (raw={:.2f})".format(final_score, raw_score))
                print(
                    "         components: profit_term={:.2f}, drawdown_penalty={:.2f}, trade_bonus={:.2f}, consistency_penalty={:.2f}, combined_profit={:.2f}, worst_dd={:.2f}, trades={}".format(
                        profit_term,
                        drawdown_term,
                        trade_term,
                        consistency_term,
                        combined_profit,
                        worst_dd,
                        combined_trades,
                    )
                )
        except Exception as exc:  # pragma: no cover - reporting-only path
            print(f"      ↳ scoring unavailable ({exc})")
    else:
        print("  No viable parameter + ATR combinations found.")

    print(
        f"  Evaluations performed: default={basic_evals}, joint={joint_evals}, total={basic_evals + joint_evals}"
    )

    if top_joint:
        best_profit_map[strategy_name] = top_joint[0]['total_profit']
        top_results_per_strategy[strategy_name] = top_joint[0]

if best_profit_map:
    priority_order = sorted(best_profit_map.items(), key=lambda item: item[1], reverse=True)
    print("\nSuggested priority order based on aggregated profits:")
    for idx, (strategy_name, profit) in enumerate(priority_order, start=1):
        print(f"  {idx}. {strategy_name} -> total_profit={profit:.2f}")

    print("\nTop recommendations (strategy, params, ATR):")
    for strategy_name, result in top_results_per_strategy.items():
        atr = result['atr']
        print(
            f"  {strategy_name}: params={result['params']}, ATR_PERIOD={atr['period']}, "
            f"SLx={atr['sl_mult']}, TPx={atr['tp_mult']}, priority={atr['priority']}, profit={result['total_profit']:.2f}"
        )

mt5.shutdown()