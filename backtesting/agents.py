import numpy as np
import pandas as pd

class TradingAgent:
    def __init__(self, initial_balance=10000):
        self.initial_balance = initial_balance
        # Defaults for PnL scaling; can be overridden per symbol from outside
        self._tick_size = 0.01
        self._tick_value = 1.0
        self._lots = 1.0

    def compute_atr_series(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute ATR series using True Range rolling mean with optional cached overrides."""
        cache = getattr(self, "_precomputed_atr_cache", None)
        if cache:
            cached = cache.get(period)
            if cached is not None and len(cached) == len(df):
                return cached

        high = df['high']
        low = df['low']
        close = df['close']
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean()
        if cache is not None:
            cache[period] = atr
        return atr

    def _simulate_long_with_sl_tp(
        self,
        df: pd.DataFrame,
        signal_series: pd.Series,
        atr_series: pd.Series,
        atr_period: int,
        sl_atr_mult: float,
        tp_atr_mult: float,
        intrabar_priority: str = 'SL'
    ):
        """
        Simulate long-only strategy: enter when signal==1, exit only on SL/TP (or at dataset end).
        No lookahead: decisions at bar i execute at open of bar i+1, ATR taken from bar i.
        intrabar_priority: 'SL' or 'TP' when both hit in same bar.
        """
        # Validate priority
        intrabar_priority = intrabar_priority.upper()
        if intrabar_priority not in ('SL', 'TP'):
            intrabar_priority = 'SL'

        trades = []
        position = 0
        entry_price = None
        sl_price = None
        tp_price = None
        balance = self.initial_balance
        balance_curve = [balance]

        # Start after both indicators and ATR are available
        start_idx = max(
            int(np.nanargmax([atr_period])) if atr_period else 0,
            atr_period
        )
        # ensure signals have at least one value before next_open access
        start_idx = max(start_idx, 1)

        for i in range(start_idx, len(df) - 1):
            prev_signal = signal_series.iloc[i]
            next_open = df['open'].iloc[i + 1]

            if position == 0 and prev_signal == 1:
                # Enter long at next open
                position = 1
                entry_price = next_open
                atr_val = atr_series.iloc[i]
                if pd.notna(atr_val) and atr_val > 0:
                    sl_dist = sl_atr_mult * atr_val
                    tp_dist = tp_atr_mult * atr_val
                    sl_price = entry_price - sl_dist
                    tp_price = entry_price + tp_dist
                else:
                    sl_price = None
                    tp_price = None
                continue

            if position == 1:
                # Check SL/TP within bar i+1
                bar_high = df['high'].iloc[i + 1]
                bar_low = df['low'].iloc[i + 1]
                exited = False
                exit_price = None

                hit_sl = sl_price is not None and bar_low <= sl_price
                hit_tp = tp_price is not None and bar_high >= tp_price

                if hit_sl and hit_tp:
                    if intrabar_priority == 'SL':
                        exit_price = sl_price
                    else:
                        exit_price = tp_price
                    exited = True
                elif hit_sl:
                    exit_price = sl_price
                    exited = True
                elif hit_tp:
                    exit_price = tp_price
                    exited = True

                if exited:
                    # Convert price diff to cash using tick_size and tick_value and lot size
                    tick_size = getattr(self, '_tick_size', 0.01) or 0.01
                    tick_value = getattr(self, '_tick_value', 1.0) or 1.0
                    lots = getattr(self, '_lots', 1.0) or 1.0
                    ticks = (exit_price - entry_price) / tick_size
                    profit = ticks * tick_value * lots
                    trades.append(profit)
                    balance += profit
                    balance_curve.append(balance)
                    position = 0
                    entry_price = sl_price = tp_price = None

        # Close any open position at last close
        if position == 1 and entry_price is not None:
            last_close = df['close'].iloc[-1]
            tick_size = getattr(self, '_tick_size', 0.01) or 0.01
            tick_value = getattr(self, '_tick_value', 1.0) or 1.0
            lots = getattr(self, '_lots', 1.0) or 1.0
            ticks = (last_close - entry_price) / tick_size
            profit = ticks * tick_value * lots
            trades.append(profit)
            balance += profit
            balance_curve.append(balance)

        total_profit = float(sum(trades))
        win_trades = len([t for t in trades if t > 0])
        win_rate = (win_trades / len(trades) * 100) if trades else 0.0
        max_drawdown = 0.0
        peak = balance_curve[0]
        for b in balance_curve:
            if b > peak:
                peak = b
            dd = (peak - b)
            if dd > max_drawdown:
                max_drawdown = dd
        return {
            'trades': len(trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'max_drawdown': max_drawdown,
            'final_balance': balance
        }

    def ma_crossover(self, df, fast_period, slow_period):
        df['fast_ma'] = df['close'].rolling(fast_period).mean()
        df['slow_ma'] = df['close'].rolling(slow_period).mean()
        df['signal'] = 0
        df.loc[df.index[slow_period:], 'signal'] = np.where(
            df['fast_ma'][slow_period:] > df['slow_ma'][slow_period:], 1, 0
        )
        df['signal'] = df['signal'].shift(1)
        # Default ATR params; allow override by caller via kwargs
        atr_period = getattr(self, '_atr_period', 14)
        sl_mult = getattr(self, '_sl_mult', 2.0)
        tp_mult = getattr(self, '_tp_mult', 3.0)
        priority = getattr(self, '_intrabar_priority', 'SL')
        atr_series = self.compute_atr_series(df, atr_period)
        return self._simulate_long_with_sl_tp(df, df['signal'], atr_series, atr_period, sl_mult, tp_mult, priority)

    def mean_reversion(self, df, ma_period, num_std):
        df['ma'] = df['close'].rolling(ma_period).mean()
        df['std'] = df['close'].rolling(ma_period).std()
        df['upper'] = df['ma'] + num_std * df['std']
        df['lower'] = df['ma'] - num_std * df['std']
        df['signal'] = 0
        df.loc[df.index[ma_period:], 'signal'] = np.where(df['close'][ma_period:] < df['lower'][ma_period:], 1, 0)
        df.loc[df.index[ma_period:], 'signal'] = np.where(df['close'][ma_period:] > df['upper'][ma_period:], -1, df['signal'][ma_period:])
        df['signal'] = df['signal'].shift(1)
        atr_period = getattr(self, '_atr_period', 14)
        sl_mult = getattr(self, '_sl_mult', 2.0)
        tp_mult = getattr(self, '_tp_mult', 3.0)
        priority = getattr(self, '_intrabar_priority', 'SL')
        atr_series = self.compute_atr_series(df, atr_period)
        return self._simulate_long_with_sl_tp(df, df['signal'], atr_series, atr_period, sl_mult, tp_mult, priority)

    def momentum_trend(self, df, ma_period=50, roc_period=10):
        df['ma'] = df['close'].rolling(ma_period).mean()
        df['roc'] = df['close'].pct_change(roc_period)
        df['signal'] = 0
        df.loc[(df['close'] > df['ma']) & (df['roc'] > 0), 'signal'] = 1
        df.loc[(df['close'] < df['ma']) & (df['roc'] < 0), 'signal'] = -1
        df['signal'] = df['signal'].shift(1)
        atr_period = getattr(self, '_atr_period', 14)
        sl_mult = getattr(self, '_sl_mult', 2.0)
        tp_mult = getattr(self, '_tp_mult', 3.0)
        priority = getattr(self, '_intrabar_priority', 'SL')
        atr_series = self.compute_atr_series(df, atr_period)
        return self._simulate_long_with_sl_tp(df, df['signal'], atr_series, atr_period, sl_mult, tp_mult, priority)

    def breakout(self, df, lookback=20):
        df['highest'] = df['high'].rolling(lookback).max()
        df['lowest'] = df['low'].rolling(lookback).min()
        df['signal'] = 0
        df.loc[df['close'] > df['highest'].shift(1), 'signal'] = 1
        df.loc[df['close'] < df['lowest'].shift(1), 'signal'] = -1
        df['signal'] = df['signal'].shift(1)
        atr_period = getattr(self, '_atr_period', 14)
        sl_mult = getattr(self, '_sl_mult', 2.0)
        tp_mult = getattr(self, '_tp_mult', 3.0)
        priority = getattr(self, '_intrabar_priority', 'SL')
        atr_series = self.compute_atr_series(df, atr_period)
        return self._simulate_long_with_sl_tp(df, df['signal'], atr_series, atr_period, sl_mult, tp_mult, priority)

    def donchian_channel(self, df, channel_length=20):
        df['donchian_high'] = df['high'].rolling(channel_length).max()
        df['donchian_low'] = df['low'].rolling(channel_length).min()
        df['donchian_mid'] = (df['donchian_high'] + df['donchian_low']) / 2.0
        channel_width = (df['donchian_high'] - df['donchian_low']).shift(1)
        df['signal'] = 0
        entry_condition = (
            (df['close'] <= df['donchian_low'].shift(1))
            & (channel_width > (df['close'] * 0.0005).shift(1))
        )
        df.loc[entry_condition, 'signal'] = 1
        df['signal'] = df['signal'].shift(1)
        atr_period = getattr(self, '_atr_period', 14)
        sl_mult = getattr(self, '_sl_mult', 2.0)
        tp_mult = getattr(self, '_tp_mult', 3.0)
        priority = getattr(self, '_intrabar_priority', 'SL')
        atr_series = self.compute_atr_series(df, atr_period)
        return self._simulate_long_with_sl_tp(df, df['signal'], atr_series, atr_period, sl_mult, tp_mult, priority)
