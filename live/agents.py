import numpy as np

class MeanReversionAgent:
    def __init__(self, ma_period=10, num_std=1):
        self.ma_period = ma_period
        self.num_std = num_std
        self.last_signal = None

    def get_signal(self, df):
        if len(df) < self.ma_period:
            return None
        close = df['close'].iloc[-1]
        ma = df['close'].rolling(self.ma_period).mean().iloc[-1]
        std = df['close'].rolling(self.ma_period).std().iloc[-1]
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std
        if close < lower:
            signal = 'buy'
        elif close > upper:
            signal = 'sell'
        else:
            signal = None
        self.last_signal = signal
        return signal

class MACrossoverAgent:
    def __init__(self, fast_period=5, slow_period=40):
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.last_signal = None

    def get_signal(self, df):
        if len(df) < self.slow_period:
            return None
        fast_ma = df['close'].rolling(self.fast_period).mean().iloc[-2]
        slow_ma = df['close'].rolling(self.slow_period).mean().iloc[-2]
        prev_fast_ma = df['close'].rolling(self.fast_period).mean().iloc[-3]
        prev_slow_ma = df['close'].rolling(self.slow_period).mean().iloc[-3]
        signal = None
        if prev_fast_ma < prev_slow_ma and fast_ma > slow_ma:
            signal = 'buy'
        elif prev_fast_ma > prev_slow_ma and fast_ma < slow_ma:
            signal = 'sell'
        self.last_signal = signal
        return signal

class MomentumTrendAgent:
    def __init__(self, ma_period=30, roc_period=5):
        self.ma_period = ma_period
        self.roc_period = roc_period
        self.last_signal = None

    def get_signal(self, df):
        if len(df) < max(self.ma_period, self.roc_period) + 1:
            return None
        ma = df['close'].rolling(self.ma_period).mean().iloc[-2]
        roc = df['close'].pct_change(self.roc_period).iloc[-2]
        close = df['close'].iloc[-2]
        signal = None
        if close > ma and roc > 0:
            signal = 'buy'
        elif close < ma and roc < 0:
            signal = 'sell'
        self.last_signal = signal
        return signal

class BreakoutAgent:
    def __init__(self, lookback=10):
        self.lookback = lookback
        self.last_signal = None

    def get_signal(self, df):
        if len(df) < self.lookback + 2:
            return None
        highest = df['high'].rolling(self.lookback).max().iloc[-2]
        lowest = df['low'].rolling(self.lookback).min().iloc[-2]
        close = df['close'].iloc[-2]
        signal = None
        if close > highest:
            signal = 'buy'
        elif close < lowest:
            signal = 'sell'
        self.last_signal = signal
        return signal

class DonchianChannelAgent:
    def __init__(self, channel_length=10):
        self.channel_length = channel_length
        self.last_signal = None

    def get_signal(self, df):
        if len(df) < self.channel_length + 2:
            return None
        donchian_high = df['high'].rolling(self.channel_length).max().iloc[-2]
        donchian_low = df['low'].rolling(self.channel_length).min().iloc[-2]
        channel_width = (donchian_high - donchian_low)
        close = df['close'].iloc[-2]
        signal = None
        if channel_width <= close * 0.0005:
            signal = None
        elif close <= donchian_low:
            signal = 'buy'
        elif close >= donchian_high:
            signal = 'sell'
        self.last_signal = signal
        return signal