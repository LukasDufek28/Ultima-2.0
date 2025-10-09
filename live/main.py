import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta
from agents import MeanReversionAgent, MACrossoverAgent, MomentumTrendAgent, BreakoutAgent, DonchianChannelAgent
from agents import create_agent, is_within_agent_window
from gui import launch_gui
from bestconfig_loader import load_bestconfig, get_params_for, available_symbols, available_strategies, find_bestconfig
import time
import os
import numpy as np

"""
Live runner that uses bestconfig.json produced by backtesting to configure agents
and enforces reverse-close logic per strategy via unique magic numbers.
"""

# Settings (fallbacks if bestconfig is missing)
symbols = ["USDJPY"]
timeframe = mt5.TIMEFRAME_M15
bars = 100  # Number of bars to fetch

# Risk/target settings (ATR-based)
ATR_PERIOD = 14
SL_ATR_MULTIPLIER = 2.0
TP_ATR_MULTIPLIER = 3.0
RESPECT_STOPS_LEVEL = True  # Enforce broker minimal stop distance if provided

# Toggle to launch GUI from this script
USE_GUI = True

def build_agents_from_bestconfig():
    """Construct live agents using the new symbol-level _meta schema.

    Returns (agents_by_symbol, meta) where meta contains per-symbol timeframe & bars
    plus original cfg for downstream use.
    """
    cfg = load_bestconfig()
    agent_dict: dict[str, list] = {}
    symbol_list = available_symbols(cfg) or symbols
    strategy_magic_base = {
        'ma_crossover': 101,
        'mean_reversion': 102,
        'momentum_trend': 103,
        'breakout': 104,
        'donchian_channel': 105,
    }

    # Helper: approximate minutes per bar for lookback estimation
    def minutes_per_bar(tf_code: int) -> int:
        mapping = {
            mt5.TIMEFRAME_M1: 1,
            mt5.TIMEFRAME_M5: 5,
            mt5.TIMEFRAME_M15: 15,
            mt5.TIMEFRAME_M30: 30,
            mt5.TIMEFRAME_H1: 60,
            mt5.TIMEFRAME_H4: 240,
            mt5.TIMEFRAME_D1: 1440,
        }
        return mapping.get(int(tf_code), 15)

    symbol_timeframes: dict[str, int] = {}
    symbol_bars: dict[str, int] = {}
    symbol_htf_filter: dict[str, dict] = {}

    for symbol in symbol_list:
        sym_node = (cfg.get('symbols', {}) or {}).get(symbol, {}) or {}
        sym_meta = sym_node.get('_meta', {}) if isinstance(sym_node, dict) else {}
        tf_code = sym_meta.get('timeframe_code') or sym_meta.get('timeframe') or timeframe
        try:
            tf_code = int(tf_code)
        except Exception:
            tf_code = timeframe
        symbol_timeframes[symbol] = tf_code
        # Days -> bars approximation (guard against missing meta)
        days_val = sym_meta.get('days', 5)
        try:
            days_val = int(days_val)
        except Exception:
            days_val = 5
        bars_guess = max(200, int((days_val * 24 * 60) / max(1, minutes_per_bar(tf_code))))
        symbol_bars[symbol] = bars_guess
        # Symbol-level HTF filter
        symbol_htf_filter[symbol] = sym_meta.get('htf_filter', {}) or {}

        # Determine active strategies
        strat_keys: list[str] = []
        for s in available_strategies(cfg, symbol):
            entry = get_params_for(cfg, symbol, s)
            if entry and bool(entry.get('active', True)):
                strat_keys.append(s)
        entries: list = []
        for strat in strat_keys:
            params_entry = get_params_for(cfg, symbol, strat)
            if not params_entry:
                continue
            agent = create_agent(strat, params_entry.get('strategy_params', {}))
            if not agent:
                continue
            agent._symbol = symbol
            agent._strategy_key = strat
            agent._lots = float(params_entry.get('risk', {}).get('lots', 0.1))
            agent._atr_cfg = params_entry.get('atr', {})
            # Trading window from symbol meta first then strategy
            tw = sym_meta.get('trading_window') or params_entry.get('trading_window', {}) or {}
            agent._trade_24_7 = bool(tw.get('trade_24_7', True))
            agent._trade_start_hour = int(tw.get('start_hour', 0))
            agent._trade_end_hour = int(tw.get('end_hour', 24))
            # Store own timeframe for clarity
            agent._timeframe_code = tf_code
            # Unique magic per symbol+strategy
            base = strategy_magic_base.get(strat, 999)
            agent._magic = (base * 100000) + abs(hash(symbol) % 99999)
            entries.append(agent)
        if entries:
            agent_dict[symbol] = entries

    meta = {
        'strategy_magic_base': strategy_magic_base,
        'cfg': cfg,
        'symbol_timeframes': symbol_timeframes,
        'symbol_bars': symbol_bars,
        'symbol_htf_filter': symbol_htf_filter,
    }
    return agent_dict, meta

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


def compute_htf_bias_scalar(symbol: str, htf_code: int, ma_p: int) -> int | None:
    """
    Returns -1, 0, 1 using the last COMPLETED HTF bar relative to its MA(ma_p).
    None if insufficient HTF history.
    """
    try:
        # Pull enough HTF bars; fallback to time range based on now
        now = datetime.now()
        # rough window to ensure enough bars for MA + some margin
        est_minutes = 1440 if htf_code == mt5.TIMEFRAME_D1 else 240 if htf_code == mt5.TIMEFRAME_H4 else 60
        from_dt = now - timedelta(minutes=est_minutes * (ma_p + 5))
        rates_htf = mt5.copy_rates_range(symbol, htf_code, from_dt, now)
        if rates_htf is None or len(rates_htf) < ma_p + 1:
            return None
        dfh = pd.DataFrame(rates_htf)
        dfh['time'] = pd.to_datetime(dfh['time'], unit='s')

        # Drop the last (potentially incomplete) HTF bar
        if len(dfh) >= 2:
            dfh = dfh.iloc[:-1]

        if len(dfh) < ma_p:
            return None

        dfh['ma'] = dfh['close'].rolling(ma_p).mean()
        last_ma = dfh['ma'].iloc[-1]
        last_close = dfh['close'].iloc[-1]
        if pd.isna(last_ma):
            return None
        b = np.sign(last_close - last_ma)
        # Cast to int -1/0/1
        return int(b) if not pd.isna(b) else None
    except Exception:
        return None


def send_order(symbol, signal, lot=None, deviation=5, comment="", atr: float | None = None, magic: int | None = None):
    # Fetch symbol details and tick once
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if symbol_info is None or tick is None:
        print(f"Cannot send order for {symbol}: symbol_info or tick is None (symbol_info={symbol_info}, tick={tick})")
        return

    # Set lot size based on symbol
    if lot is None:
        if symbol.upper() == 'XAUUSD':
            lot = 0.01
        elif symbol.upper() in ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']:
            lot = 1  # For stocks, usually 1
        else:
            lot = 0.1
    price = tick.ask if signal == 'buy' else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL

    # Compute SL/TP from ATR if available
    sl = None
    tp = None
    if atr is not None and atr > 0:
        sl_dist = SL_ATR_MULTIPLIER * atr
        tp_dist = TP_ATR_MULTIPLIER * atr
        if signal == 'buy':
            sl = price - sl_dist
            tp = price + tp_dist
        else:  # sell
            sl = price + sl_dist
            tp = price - tp_dist

        # Respect minimal stop distance in points if requested
        if RESPECT_STOPS_LEVEL and hasattr(symbol_info, 'trade_stops_level') and symbol_info.trade_stops_level:
            point = symbol_info.point
            min_dist = symbol_info.trade_stops_level * point
            # Adjust SL
            if sl is not None and abs(price - sl) < min_dist:
                if signal == 'buy':
                    sl = price - min_dist
                else:
                    sl = price + min_dist
            # Adjust TP
            if tp is not None and abs(price - tp) < min_dist:
                if signal == 'buy':
                    tp = price + min_dist
                else:
                    tp = price - min_dist

        # Round to symbol digits
        digits = getattr(symbol_info, 'digits', 5)
        sl = round(sl, digits) if sl is not None else None
        tp = round(tp, digits) if tp is not None else None

    if atr is None:
        print(f"ATR unavailable for {symbol}; sending order without SL/TP.")
    else:
        print(f"{symbol} ATR={atr:.6f} | SL={sl} | TP={tp}")

    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': lot,
        'type': order_type,
        'price': price,
        'deviation': deviation,
        'magic': magic or 123456,
        'comment': comment,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_FOK,  # Use Fill or Kill
    }
    # Attach SL/TP if computed
    if sl is not None:
        request['sl'] = sl
    if tp is not None:
        request['tp'] = tp
    result = mt5.order_send(request)
    if result is None:
        print(f"OrderSend returned None for {symbol}. Check connection, symbol, and request parameters.")
        print(f"Request: {request}")
        return
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"OrderSend failed for {symbol}: {result.retcode}")
        print(f"Reason: {result.comment}")
    else:
        print(f"OrderSend success for {symbol}: {signal} at {price} | SL={request.get('sl')} TP={request.get('tp')}")

def close_positions_for_strategy(symbol: str, desired_side: str, agent_magic: int):
    """Close positions that conflict with desired_side for this strategy (identified by magic)."""
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return
    for p in positions:
        # Only manage positions opened by this strategy
        if p.magic != agent_magic:
            continue
        is_buy = (p.type == mt5.POSITION_TYPE_BUY)
        if desired_side == 'buy' and is_buy:
            # No conflict
            continue
        if desired_side == 'sell' and not is_buy:
            continue
        # Close conflicting position
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            continue
        req = {
            'action': mt5.TRADE_ACTION_DEAL,
            'symbol': symbol,
            'volume': p.volume,
            'type': mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY,
            'position': p.ticket,
            'price': tick.bid if is_buy else tick.ask,
            'deviation': 10,
            'magic': p.magic,
            'comment': 'reverse close',
            'type_time': mt5.ORDER_TIME_GTC,
            'type_filling': mt5.ORDER_FILLING_IOC,
        }
        mt5.order_send(req)


def main():
    # Build from bestconfig first
    agent_dict, meta = build_agents_from_bestconfig()
    syms = list(agent_dict.keys()) or symbols

    try:
        print("Initializing MT5 (headless)...")
        if not mt5.initialize():
            print("initialize() failed")
            return
        while True:
            print(f"\nChecking signals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            for symbol in syms:
                # Ensure symbol is selected/visible
                info = mt5.symbol_info(symbol)
                if info is None:
                    print(f"Symbol {symbol} not found")
                    continue
                if not info.visible:
                    mt5.symbol_select(symbol, True)
                # Per-symbol timeframe & lookback
                tf_code = (meta.get('symbol_timeframes', {}) or {}).get(symbol, timeframe)
                lookback_bars = (meta.get('symbol_bars', {}) or {}).get(symbol, bars)
                # Map timeframe code to minutes (simple approximation)
                def _mpb(code):
                    m = {mt5.TIMEFRAME_M1:1, mt5.TIMEFRAME_M5:5, mt5.TIMEFRAME_M15:15, mt5.TIMEFRAME_M30:30, mt5.TIMEFRAME_H1:60, mt5.TIMEFRAME_H4:240, mt5.TIMEFRAME_D1:1440}
                    return m.get(int(code), 15)
                today = datetime.now()
                from_date = today - timedelta(minutes=lookback_bars * _mpb(tf_code))
                rates = mt5.copy_rates_range(symbol, tf_code, from_date, today)
                if rates is None:
                    print(f"No data for {symbol}")
                    continue
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                # Choose ATR settings per symbol+strategy if provided
                atr_value = None
                # We'll compute per agent, since ATR period and multipliers could differ per strategy
                if atr_value is not None:
                    print(f"{symbol} ATR({ATR_PERIOD}) = {atr_value:.6f}")
                # Compute HTF bias for this symbol once per loop (if enabled globally or in any agent)
                cfg = meta.get('cfg', {})
                sym_cfg = (cfg or {}).get('symbols', {}).get(symbol, {}) or {}
                sym_meta = sym_cfg.get('_meta', {}) if isinstance(sym_cfg, dict) else {}
                # Symbol-level HTF filter settings
                htf_meta = (meta.get('symbol_htf_filter', {}) or {}).get(symbol, {}) or {}
                htf_enabled_global = bool(htf_meta.get('enabled', False))
                # Compute bias if any agent needs it
                def compute_bias(htf_code: int, ma_p: int):
                    try:
                        from_date = df['time'].iloc[0]
                        to_date = df['time'].iloc[-1]
                        rates_htf = mt5.copy_rates_range(symbol, htf_code, from_date.to_pydatetime(), to_date.to_pydatetime())
                        if rates_htf is None or len(rates_htf) == 0:
                            return None
                        dfh = pd.DataFrame(rates_htf)
                        dfh['time'] = pd.to_datetime(dfh['time'], unit='s')
                        dfh['ma'] = dfh['close'].rolling(ma_p).mean()
                        dfh['bias'] = np.sign(dfh['close'] - dfh['ma'])
                        merged = pd.merge_asof(
                            df[['time']],
                            dfh[['time', 'bias']].dropna(subset=['time']),
                            on='time',
                            direction='backward'
                        )
                        return merged['bias']
                    except Exception:
                        return None
                # Prepare per-strategy bias cache
                bias_cache = {}
                for agent in agent_dict.get(symbol, []):
                    strategy_name = agent._strategy_key
                    # Enforce trading window
                    now_bar_time = df['time'].iloc[-1] if 'time' in df.columns else datetime.now()
                    if not is_within_agent_window(agent, now_bar_time):
                        print(f"{symbol} {strategy_name}: outside trading window; skipping.")
                        continue
                    # Determine HTF for this agent (from symbol+strategy config or defaults)
                    entry = sym_cfg.get(strategy_name, {})
                    # Strategy-level override else symbol meta
                    htf_cfg = entry.get('htf_filter') or htf_meta or {}
                    htf_enabled = bool(htf_cfg.get('enabled', htf_enabled_global))
                    bias_series = None
                    if htf_enabled:
                        htf_code = int(htf_cfg.get('timeframe_code', sym_meta.get('htf_filter', {}).get('timeframe_code', mt5.TIMEFRAME_H1)))
                        ma_p = int(htf_cfg.get('ma_period', sym_meta.get('htf_filter', {}).get('ma_period', 50)))
                        key = (htf_code, ma_p)
                        if key not in bias_cache:
                            bias_cache[key] = compute_bias(htf_code, ma_p)
                        bias_series = bias_cache.get(key)
                    # Compute ATR per agent's config period using completed bars
                    try:
                        atr_period = int(agent._atr_cfg.get('period', ATR_PERIOD))
                    except Exception:
                        atr_period = ATR_PERIOD
                    atr_value = compute_atr(df, atr_period)
                    # Override global multipliers when sending order
                    global SL_ATR_MULTIPLIER, TP_ATR_MULTIPLIER
                    SL_ATR_MULTIPLIER = float(agent._atr_cfg.get('sl_mult', SL_ATR_MULTIPLIER))
                    TP_ATR_MULTIPLIER = float(agent._atr_cfg.get('tp_mult', TP_ATR_MULTIPLIER))
                    # Apply HTF bias gating to signals (long only if bias>0, short only if bias<0)
                    signal = agent.get_signal(df)
                    if bias_series is not None and signal in ('buy', 'sell'):
                        try:
                            b = float(bias_series.iloc[-1])
                        except Exception:
                            b = None
                        if b is not None:
                            if signal == 'buy' and b <= 0:
                                print(f"{symbol} {strategy_name}: blocked by HTF bias (b={b}).")
                                signal = None
                            elif signal == 'sell' and b >= 0:
                                print(f"{symbol} {strategy_name}: blocked by HTF bias (b={b}).")
                                signal = None
                    print(f"{symbol} {strategy_name} signal: {signal}")
                    if signal in ('buy', 'sell'):
                        # First, close conflicting positions for this strategy's magic
                        close_positions_for_strategy(symbol, signal, agent._magic)
                        # Then, check if a position already exists for this strategy side
                        positions = mt5.positions_get(symbol=symbol)
                        have_side = False
                        if positions:
                            for p in positions:
                                if p.magic == agent._magic:
                                    if (signal == 'buy' and p.type == mt5.POSITION_TYPE_BUY) or (signal == 'sell' and p.type == mt5.POSITION_TYPE_SELL):
                                        have_side = True
                                        break
                        if not have_side:
                            send_order(
                                symbol,
                                signal,
                                lot=getattr(agent, '_lots', None),
                                comment=f"{strategy_name} signal",
                                atr=atr_value,
                                magic=getattr(agent, '_magic', None)
                            )
            time.sleep(300)
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    if USE_GUI:
        # Preload bestconfig and pass into GUI so it auto-imports at startup
        best_path = find_bestconfig()
        cfg = load_bestconfig(best_path) if best_path else {}
        if best_path:
            print(f"Loaded bestconfig from: {best_path}")
            print(f"Symbols: {available_symbols(cfg)}")
        else:
            print("No bestconfig.json found; GUI will start with defaults.")
        launch_gui(preloaded_cfg=cfg)
    else:
        main()