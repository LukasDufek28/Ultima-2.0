import MetaTrader5 as mt5

# Core settings
TIMEFRAME = mt5.TIMEFRAME_M15
SYMBOLS = ["EURUSD+", "USDJPY+", "GBPUSD+", "GBPJPY+", "XAUUSD+"]
DEFAULT_LOOKBACK_DAYS = 60

# Risk management
ACCOUNT_RISK_PER_TRADE = 0.30  # Extremely aggressive per-trade exposure for growth-focused research
DEFAULT_ACCOUNT_BALANCE = 1000.0
DEFAULT_INITIAL_BALANCE = 10000.0
MIN_LOT_SIZE = 0.01
MAX_LOT_SIZE = 5.0
RISK_MULTIPLIER_MIN = 1.0
RISK_MULTIPLIER_MAX = 3.5
ALLOW_HEDGING = False

# ATR / exit defaults
ATR_PERIOD = 5
SL_ATR_MULTIPLIER = 2.0
TP_ATR_MULTIPLIER = 2.0
INTRABAR_PRIORITY = "TP"

# Adaptive ATR controls
ENABLE_ADAPTIVE_ATR = True
ATR_VOL_LOOKBACK_MULTIPLIER = 2.0
ATR_VOL_LOW_THRESHOLD = 0.85
ATR_VOL_HIGH_THRESHOLD = 1.2
ATR_VOL_MIN_PERIOD = 3

# Trade quality filters
SPREAD_POINTS_LIMIT = 10
SPREAD_ATR_RATIO_LIMIT = 0.45

# Micro momentum
ENABLE_MICRO_MOMENTUM_CONFIRMATION = True
MICRO_MOMENTUM_TIMEFRAME = mt5.TIMEFRAME_M5
MICRO_MOMENTUM_LOOKBACK = 6
MICRO_MOMENTUM_BASE_THRESHOLD = 0.0006
MICRO_MOMENTUM_MIN_THRESHOLD = 0.0002
MICRO_MOMENTUM_MAX_THRESHOLD = 0.0025
MICRO_MOMENTUM_DYNAMIC_MULTIPLIER = 0.55
MICRO_MOMENTUM_SOFT_PASS_RATIO = 0.45

# Multi-timeframe confirmation
ENABLE_MTF_CONFIRMATION = True
MTF_TIMEFRAME = mt5.TIMEFRAME_H1
MTF_LOOKBACK_BARS = 60

# Sessions
MIN_SESSION_PRIORITY = 3
INSTRUMENT_SESSION_PRIORITY = {
    "EURUSD+": {
        "Sydney/Tokyo": 2,
        "London": 5,
        "London-NY Overlap": 5,
        "New York": 4,
    },
    "GBPUSD+": {
        "Sydney/Tokyo": 1,
        "London": 5,
        "London-NY Overlap": 5,
        "New York": 4,
    },
    "USDJPY+": {
        "Sydney/Tokyo": 5,
        "London": 4,
        "London-NY Overlap": 4,
        "New York": 3,
    },
    "GBPJPY+": {
        "Sydney/Tokyo": 4,
        "London": 5,
        "London-NY Overlap": 4,
        "New York": 2,
    },
    "XAUUSD+": {
        "Sydney/Tokyo": 3,
        "London": 5,
        "London-NY Overlap": 5,
        "New York": 4,
    },
}

# Correlation management
CORRELATION_GROUPS = {
    "EUR_PAIRS": ["EURUSD+"],
    "GBP_PAIRS": ["GBPUSD+", "GBPJPY+"],
    "JPY_PAIRS": ["USDJPY+", "GBPJPY+"],
    "USD_MAJORS": ["EURUSD+", "GBPUSD+", "USDJPY+"],
    "SAFE_HAVEN": ["USDJPY+", "XAUUSD+"],
}
MAX_CORRELATED_POSITIONS = 2
CORRELATION_POSITION_LIMIT = 0.6

# Regime detection
REGIME_LOOKBACK_PERIODS = 50
TREND_THRESHOLD = 0.3
VOLATILITY_THRESHOLD = 1.5
REGIME_STRATEGY_WEIGHTS = {
    "TRENDING": {
        "ma_crossover": 0.35,
        "momentum_trend": 0.30,
        "breakout": 0.25,
        "mean_reversion": 0.05,
        "donchian_channel": 0.05,
    },
    "RANGING": {
        "mean_reversion": 0.40,
        "donchian_channel": 0.25,
        "ma_crossover": 0.15,
        "momentum_trend": 0.10,
        "breakout": 0.10,
    },
    "VOLATILE": {
        "breakout": 0.35,
        "momentum_trend": 0.25,
        "ma_crossover": 0.20,
        "donchian_channel": 0.15,
        "mean_reversion": 0.05,
    },
}

# Performance adaptation
PERFORMANCE_LOOKBACK_DAYS = 30
MIN_TRADES_FOR_ADAPTATION = 10

# Strategy defaults (match live definitions)
STRATEGY_DEFINITIONS = [
    {
        "label": "Breakout",
        "name": "breakout",
        "params": {"lookback": 10},
        "sl_mult": 2.5,
        "tp_mult": 3.0,
        "priority": 3,
        "atr_period": 14,
        "atr_bands": {
            "low": {"period": 12, "sl_mult": 2.3, "tp_mult": 3.2},
            "normal": {"period": 14, "sl_mult": 2.5, "tp_mult": 3.0},
            "high": {"period": 16, "sl_mult": 2.8, "tp_mult": 2.7},
        },
    },
    {
        "label": "Donchian Channel",
        "name": "donchian_channel",
        "params": {"channel_length": 10},
        "sl_mult": 3.0,
        "tp_mult": 3.5,
        "priority": 4,
        "atr_period": 5,
        "atr_bands": {
            "low": {"period": 4, "sl_mult": 2.7, "tp_mult": 3.8},
            "normal": {"period": 5, "sl_mult": 3.0, "tp_mult": 3.5},
            "high": {"period": 6, "sl_mult": 3.2, "tp_mult": 3.2},
        },
    },
    {
        "label": "MA Crossover",
        "name": "ma_crossover",
        "params": {"fast_period": 5, "slow_period": 40},
        "sl_mult": 2.0,
        "tp_mult": 2.0,
        "priority": 2,
        "atr_period": 5,
        "atr_bands": {
            "low": {"period": 4, "sl_mult": 1.8, "tp_mult": 2.2},
            "normal": {"period": 5, "sl_mult": 2.0, "tp_mult": 2.0},
            "high": {"period": 6, "sl_mult": 2.3, "tp_mult": 1.9},
        },
    },
    {
        "label": "Momentum Trend",
        "name": "momentum_trend",
        "params": {"ma_period": 30, "roc_period": 5},
        "sl_mult": 2.75,
        "tp_mult": 2.0,
        "priority": 5,
        "atr_period": 5,
        "atr_bands": {
            "low": {"period": 4, "sl_mult": 2.5, "tp_mult": 2.2},
            "normal": {"period": 5, "sl_mult": 2.75, "tp_mult": 2.0},
            "high": {"period": 6, "sl_mult": 3.0, "tp_mult": 1.8},
        },
    },
    {
        "label": "Mean Reversion",
        "name": "mean_reversion",
        "params": {"ma_period": 10, "num_std": 1.0},
        "sl_mult": 2.75,
        "tp_mult": 4.0,
        "priority": 1,
        "atr_period": 7,
        "atr_bands": {
            "low": {"period": 6, "sl_mult": 2.5, "tp_mult": 4.2},
            "normal": {"period": 7, "sl_mult": 2.75, "tp_mult": 4.0},
            "high": {"period": 8, "sl_mult": 3.0, "tp_mult": 3.6},
        },
    },
]
