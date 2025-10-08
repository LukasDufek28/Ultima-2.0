import threading
import time
import tkinter as tk
from tkinter import ttk
from datetime import datetime, timedelta
import MetaTrader5 as mt5
from bestconfig_loader import load_bestconfig, available_symbols
from agents import create_agent, is_within_agent_window
import pandas as pd
import numpy as np


def _to_py_dt(x):
    return x.to_pydatetime() if hasattr(x, 'to_pydatetime') else x


def timeframe_to_minutes(tf_code: int) -> int:
    mapping = {
        mt5.TIMEFRAME_M1: 1,
        mt5.TIMEFRAME_M5: 5,
        mt5.TIMEFRAME_M15: 15,
        mt5.TIMEFRAME_M30: 30,
        mt5.TIMEFRAME_H1: 60,
        mt5.TIMEFRAME_H4: 240,
        mt5.TIMEFRAME_D1: 60 * 24,
    }
    return mapping.get(tf_code, 15)


class LiveController:
    def __init__(self, get_state_cb, status_cb):
        self.get_state_cb = get_state_cb
        self.status_cb = status_cb
        self.running = False
        self.thread = None
        self.lock = threading.Lock()
        # Timer cadence
        self.interval_secs = 60
        self.next_run_ts = None
        self.last_run_ts = None
        # Live stats cache: {symbol: {strategy: { 'buy': {...}, 'sell': {...} } } }
        self.stats = {}

    def _log(self, msg):
        if self.status_cb:
            self.status_cb(msg)

    def start(self):
        with self.lock:
            if self.running:
                return
            self.running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self):
        with self.lock:
            self.running = False

    def _compute_bias(self, symbol, df, htf_code, ma_p):
        try:
            from_date = _to_py_dt(df['time'].iloc[0])
            to_date = _to_py_dt(df['time'].iloc[-1])
            rates_htf = mt5.copy_rates_range(symbol, htf_code, from_date, to_date)
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

    def _run_loop(self):
        try:
            if not mt5.initialize():
                self._log("MT5 initialize() failed")
                return
            self._log("Live bots started")
            # Strategy magic numbers base
            strategy_magic_base = {
                'ma_crossover': 101,
                'mean_reversion': 102,
                'momentum_trend': 103,
                'breakout': 104,
                'donchian_channel': 105,
            }
            # ATR defaults
            DEFAULT_ATR_PERIOD = 14
            DEFAULT_SL_MULT = 2.0
            DEFAULT_TP_MULT = 3.0
            # Initialize next run to immediate
            self.next_run_ts = datetime.now()
            while True:
                with self.lock:
                    if not self.running:
                        self._log("Live bots stopped")
                        break
                now = datetime.now()
                # Only execute when next_run reached
                if self.next_run_ts is None or now >= self.next_run_ts:
                    state = self.get_state_cb() or {}
                    symbols = state.get('symbols', {})
                    # Clear stats for fresh accumulation this cycle
                    with self.lock:
                        self.stats = {}
                    for symbol, conf in symbols.items():
                        if not conf.get('active', False):
                            continue
                        info = mt5.symbol_info(symbol)
                        if info is None:
                            self._log(f"Symbol {symbol} not found")
                            continue
                        if not info.visible:
                            mt5.symbol_select(symbol, True)
                        tf = int(conf.get('timeframe_code', mt5.TIMEFRAME_M15))
                        bars = int(conf.get('lookback_bars', 400))
                        minutes_per_bar = timeframe_to_minutes(tf)
                        from_date = now - timedelta(minutes=bars * minutes_per_bar)
                        rates = mt5.copy_rates_range(symbol, tf, from_date, now)
                        if rates is None or len(rates) == 0:
                            self._log(f"No data for {symbol}")
                            continue
                        df = pd.DataFrame(rates)
                        df['time'] = pd.to_datetime(df['time'], unit='s')

                        # HTF bias
                        htf = conf.get('htf_filter', {}) or {}
                        bias_series = None
                        if bool(htf.get('enabled', False)):
                            bias_series = self._compute_bias(symbol, df, int(htf.get('timeframe_code', mt5.TIMEFRAME_H1)), int(htf.get('ma_period', 50)))

                        # Trading window config
                        tw = conf.get('trading_window', {}) or {}

                        for strat_key, sconf in (conf.get('strategies') or {}).items():
                            if not sconf.get('enabled', False):
                                continue
                            agent = create_agent(strat_key, sconf.get('params', {}) or {})
                            if not agent:
                                continue
                            agent._symbol = symbol
                            agent._strategy_key = strat_key
                            agent._lots = float(conf.get('lots', 0.1))
                            agent._trade_24_7 = bool(tw.get('trade_24_7', True))
                            agent._trade_start_hour = int(tw.get('start_hour', 0))
                            agent._trade_end_hour = int(tw.get('end_hour', 24))

                            # Window gate
                            if not is_within_agent_window(agent, df['time'].iloc[-1]):
                                self._log(f"{symbol} {strat_key}: outside trading window")
                                continue

                            signal = agent.get_signal(df)
                            if bias_series is not None and signal in ('buy', 'sell'):
                                try:
                                    b = float(bias_series.iloc[-1])
                                except Exception:
                                    b = None
                                if b is not None:
                                    if signal == 'buy' and b <= 0:
                                        signal = None
                                    elif signal == 'sell' and b >= 0:
                                        signal = None
                            self._log(f"{symbol} {strat_key}: {signal}")

                            # Compute magic for stats and order management
                            base = strategy_magic_base.get(strat_key, 999)
                            magic = (base * 100000) + abs(hash(symbol) % 99999)

                            # Execute orders
                            if signal in ('buy', 'sell'):
                                # Compute ATR per strategy
                                atr_cfg = (sconf.get('atr') or {})
                                try:
                                    atr_period = int(float(atr_cfg.get('period', DEFAULT_ATR_PERIOD)))
                                except Exception:
                                    atr_period = DEFAULT_ATR_PERIOD
                                atr_val = compute_atr_local(df, atr_period)
                                sl_mult = safe_float(atr_cfg.get('sl_mult'), DEFAULT_SL_MULT)
                                tp_mult = safe_float(atr_cfg.get('tp_mult'), DEFAULT_TP_MULT)
                                # Reverse close conflicting positions
                                close_positions_for_strategy_local(symbol, signal, magic)
                                # Skip if already have same side position with same magic
                                if have_position_for_magic(symbol, signal, magic):
                                    pass
                                else:
                                    # Send
                                    send_order_local(symbol, signal, lot=float(conf.get('lots', 0.1)), atr=atr_val, sl_mult=sl_mult, tp_mult=tp_mult, magic=magic, log=self._log)

                            # Collect per-strategy open position stats
                            try:
                                positions = mt5.positions_get(symbol=symbol)
                            except Exception:
                                positions = None
                            stats_entry = {'buy': {'volume': 0.0, 'avg_price': 0.0, 'unrealized_pl': 0.0, 'positions': 0},
                                           'sell': {'volume': 0.0, 'avg_price': 0.0, 'unrealized_pl': 0.0, 'positions': 0}}
                            if positions:
                                # Aggregate by side for this magic
                                sum_price_buy = 0.0; sum_vol_buy = 0.0; sum_pl_buy = 0.0; cnt_buy = 0
                                sum_price_sell = 0.0; sum_vol_sell = 0.0; sum_pl_sell = 0.0; cnt_sell = 0
                                for p in positions:
                                    if p.magic != magic:
                                        continue
                                    if p.type == mt5.POSITION_TYPE_BUY:
                                        sum_vol_buy += p.volume
                                        sum_price_buy += p.price_open * p.volume
                                        sum_pl_buy += getattr(p, 'profit', 0.0)
                                        cnt_buy += 1
                                    else:
                                        sum_vol_sell += p.volume
                                        sum_price_sell += p.price_open * p.volume
                                        sum_pl_sell += getattr(p, 'profit', 0.0)
                                        cnt_sell += 1
                                if sum_vol_buy > 0:
                                    stats_entry['buy']['volume'] = sum_vol_buy
                                    stats_entry['buy']['avg_price'] = sum_price_buy / sum_vol_buy
                                    stats_entry['buy']['unrealized_pl'] = sum_pl_buy
                                    stats_entry['buy']['positions'] = cnt_buy
                                if sum_vol_sell > 0:
                                    stats_entry['sell']['volume'] = sum_vol_sell
                                    stats_entry['sell']['avg_price'] = sum_price_sell / sum_vol_sell
                                    stats_entry['sell']['unrealized_pl'] = sum_pl_sell
                                    stats_entry['sell']['positions'] = cnt_sell
                            with self.lock:
                                self.stats.setdefault(symbol, {})[strat_key] = stats_entry

                    # Update timestamps and schedule next run at next minute boundary
                    self.last_run_ts = now
                    # Next run at the next minute boundary
                    boundary = now.replace(second=0, microsecond=0) + timedelta(minutes=1)
                    self.next_run_ts = boundary
                # Sleep in small increments to allow UI countdown to refresh smoothly
                time.sleep(1)
        finally:
            mt5.shutdown()


def safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)


def compute_atr_local(df: pd.DataFrame, period: int) -> float | None:
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
        atr_val = atr_series.iloc[-2]
        if pd.isna(atr_val):
            return None
        return float(atr_val)
    except Exception:
        return None


def close_positions_for_strategy_local(symbol: str, desired_side: str, magic: int):
    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return
        for p in positions:
            if p.magic != magic:
                continue
            is_buy = (p.type == mt5.POSITION_TYPE_BUY)
            if desired_side == 'buy' and is_buy:
                continue
            if desired_side == 'sell' and not is_buy:
                continue
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
    except Exception:
        return


def have_position_for_magic(symbol: str, desired_side: str, magic: int) -> bool:
    try:
        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return False
        for p in positions:
            if p.magic != magic:
                continue
            is_buy = (p.type == mt5.POSITION_TYPE_BUY)
            if (desired_side == 'buy' and is_buy) or (desired_side == 'sell' and not is_buy):
                return True
        return False
    except Exception:
        return False


def send_order_local(symbol: str, signal: str, lot: float, atr: float | None, sl_mult: float, tp_mult: float, magic: int, log=print):
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if symbol_info is None or tick is None:
        log(f"Cannot send order for {symbol}: symbol_info or tick is None")
        return
    price = tick.ask if signal == 'buy' else tick.bid
    order_type = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL

    sl = None
    tp = None
    if atr is not None and atr > 0:
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr
        if signal == 'buy':
            sl = price - sl_dist
            tp = price + tp_dist
        else:
            sl = price + sl_dist
            tp = price - tp_dist
        # Respect minimal stop distance
        try:
            if hasattr(symbol_info, 'trade_stops_level') and symbol_info.trade_stops_level:
                point = symbol_info.point
                min_dist = symbol_info.trade_stops_level * point
                if sl is not None and abs(price - sl) < min_dist:
                    sl = price - min_dist if signal == 'buy' else price + min_dist
                if tp is not None and abs(price - tp) < min_dist:
                    tp = price + min_dist if signal == 'buy' else price - min_dist
        except Exception:
            pass
        digits = getattr(symbol_info, 'digits', 5)
        sl = round(sl, digits) if sl is not None else None
        tp = round(tp, digits) if tp is not None else None
    else:
        log(f"{symbol}: ATR unavailable; sending market order without SL/TP.")

    req = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': lot,
        'type': order_type,
        'price': price,
        'deviation': 10,
        'magic': magic,
        'comment': 'live-gui',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_FOK,
    }
    if sl is not None:
        req['sl'] = sl
    if tp is not None:
        req['tp'] = tp
    res = mt5.order_send(req)
    if res is None:
        log(f"OrderSend returned None for {symbol}")
        return
    if res.retcode != mt5.TRADE_RETCODE_DONE:
        log(f"OrderSend failed for {symbol}: {res.retcode} {getattr(res, 'comment', '')}")
    else:
        log(f"OrderSend success for {symbol}: {signal} at {price} | SL={req.get('sl')} TP={req.get('tp')}")


def parse_timeframe_input(val):
    s = str(val).strip().upper()
    try:
        return int(float(s))
    except Exception:
        pass
    m = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5, 'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30}
    h = {'H1': mt5.TIMEFRAME_H1, 'H4': mt5.TIMEFRAME_H4}
    d = {'D1': mt5.TIMEFRAME_D1}
    if s in m:
        return m[s]
    if s in h:
        return h[s]
    if s in d:
        return d[s]
    return mt5.TIMEFRAME_M15


def timeframe_code_to_str(tf_code: int) -> str:
    mapping = {
        mt5.TIMEFRAME_M1: 'M1',
        mt5.TIMEFRAME_M5: 'M5',
        mt5.TIMEFRAME_M15: 'M15',
        mt5.TIMEFRAME_M30: 'M30',
        mt5.TIMEFRAME_H1: 'H1',
        mt5.TIMEFRAME_H4: 'H4',
        mt5.TIMEFRAME_D1: 'D1',
    }
    return mapping.get(int(tf_code), 'M15')


def launch_gui(preloaded_cfg: dict | None = None):
    """Launch live GUI.

    If preloaded_cfg is provided it will be used instead of loading from disk.
    This allows the caller (main script) to control when/where bestconfig is loaded
    and report path or validation errors earlier.
    """
    cfg = preloaded_cfg if isinstance(preloaded_cfg, dict) else load_bestconfig()
    symbols_list = available_symbols(cfg) or ["XAUUSD"]

    root = tk.Tk()
    root.title("Ultima Live Trading")
    root.geometry("1100x700")

    log_box = tk.Text(root, height=10)
    log_box.pack(fill=tk.X, padx=8, pady=6)

    def log(msg):
        ts = datetime.now().strftime('%H:%M:%S')
        log_box.insert(tk.END, f"[{ts}] {msg}\n")
        log_box.see(tk.END)

    # UI state
    ui = {}  # symbol -> vars

    # Tabs container
    notebook = ttk.Notebook(root)
    notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=6)

    def build_symbol_tab(symbol):
        frm = ttk.Frame(notebook, padding=8)
        notebook.add(frm, text=symbol)

        sym_cfg = (cfg.get('symbols', {}).get(symbol, {}) or {})
        sym_meta = sym_cfg.get('_meta', {}) if isinstance(sym_cfg, dict) else {}
        def_htf = (sym_meta.get('htf_filter', {}) or {})

        active = tk.BooleanVar(value=True)
        tf_str = tk.StringVar(value=str(sym_meta.get('timeframe', 'M15')))
        bars_var = tk.StringVar(value=str(400))
        init_lots = float((sym_meta.get('risk') or {}).get('lots', 0.1))
        lots_var = tk.StringVar(value=str(init_lots))
        tw_247 = tk.BooleanVar(value=True)
        tw_start = tk.StringVar(value="0")
        tw_end = tk.StringVar(value="24")
        htf_enabled = tk.BooleanVar(value=bool(def_htf.get('enabled', False)))
        htf_tf = tk.StringVar(value=str(def_htf.get('timeframe', 'H1')))
        htf_ma = tk.StringVar(value=str(def_htf.get('ma_period', 50)))

        strat_defs = {
            'ma_crossover': {'enabled': False, 'params': {'fast': '5', 'slow': '20'}},
            'mean_reversion': {'enabled': False, 'params': {'ma_period': '10', 'num_std': '1'}},
            'momentum_trend': {'enabled': False, 'params': {'ma_period': '100', 'roc_period': '20'}},
            'breakout': {'enabled': False, 'params': {'lookback': '20'}},
            'donchian_channel': {'enabled': False, 'params': {'channel_length': '20'}},
        }
        def_atr = {'period': '21', 'sl_mult': '1.5', 'tp_mult': '2.5', 'priority': 'SL'}

        svars = {}
        row = 0
        ttk.Checkbutton(frm, text="Active", variable=active).grid(row=row, column=0, sticky='w')
        ttk.Label(frm, text="Timeframe").grid(row=row, column=1, sticky='e')
        ttk.Entry(frm, textvariable=tf_str, width=8).grid(row=row, column=2, sticky='w')
        ttk.Label(frm, text="Bars").grid(row=row, column=3, sticky='e')
        ttk.Entry(frm, textvariable=bars_var, width=6).grid(row=row, column=4, sticky='w')
        ttk.Label(frm, text="Lots").grid(row=row, column=5, sticky='e')
        ttk.Entry(frm, textvariable=lots_var, width=6).grid(row=row, column=6, sticky='w')
        row += 1

        box = ttk.LabelFrame(frm, text="Trading Window", padding=6)
        box.grid(row=row, column=0, columnspan=7, sticky='ew', pady=6)
        ttk.Checkbutton(box, text="24/7", variable=tw_247).grid(row=0, column=0, sticky='w')
        ttk.Label(box, text="Start").grid(row=0, column=1)
        ttk.Entry(box, textvariable=tw_start, width=4).grid(row=0, column=2)
        ttk.Label(box, text="End").grid(row=0, column=3)
        ttk.Entry(box, textvariable=tw_end, width=4).grid(row=0, column=4)
        row += 1

        box2 = ttk.LabelFrame(frm, text="HTF Filter", padding=6)
        box2.grid(row=row, column=0, columnspan=7, sticky='ew', pady=6)
        ttk.Checkbutton(box2, text="Enable", variable=htf_enabled).grid(row=0, column=0, sticky='w')
        ttk.Label(box2, text="TF").grid(row=0, column=1)
        ttk.Entry(box2, textvariable=htf_tf, width=6).grid(row=0, column=2)
        ttk.Label(box2, text="MA").grid(row=0, column=3)
        ttk.Entry(box2, textvariable=htf_ma, width=6).grid(row=0, column=4)
        row += 1

        tbl = ttk.LabelFrame(frm, text="Strategies", padding=6)
        tbl.grid(row=row, column=0, columnspan=7, sticky='ew', pady=6)
        rr = 0
        ttk.Label(tbl, text="Strategy").grid(row=rr, column=0)
        ttk.Label(tbl, text="Enabled").grid(row=rr, column=1)
        ttk.Label(tbl, text="Params").grid(row=rr, column=2)
        ttk.Label(tbl, text="ATR (period, SLx, TPx, prio)").grid(row=rr, column=3)
        rr += 1
        for sk, d in strat_defs.items():
            cent = sym_cfg.get(sk, {}) or {}
            params = cent.get('strategy_params', d['params'])
            atr = cent.get('atr', def_atr)
            en = tk.BooleanVar(value=bool(cent.get('active', False)) if cent else bool(d['enabled']))
            svars.setdefault(sk, {})
            svars[sk]['enabled'] = en
            ttk.Label(tbl, text=sk).grid(row=rr, column=0, sticky='w')
            ttk.Checkbutton(tbl, variable=en).grid(row=rr, column=1)
            if sk == 'ma_crossover':
                p1 = tk.StringVar(value=str(params.get('fast', '5')))
                p2 = tk.StringVar(value=str(params.get('slow', '20')))
                ent = ttk.Entry(tbl, width=18)
                ent.insert(0, f"fast={p1.get()},slow={p2.get()}")
                svars[sk]['p_fast'] = p1; svars[sk]['p_slow'] = p2; svars[sk]['params_ent'] = ent
            elif sk == 'mean_reversion':
                p1 = tk.StringVar(value=str(params.get('ma_period', '10')))
                p2 = tk.StringVar(value=str(params.get('num_std', '1')))
                ent = ttk.Entry(tbl, width=18)
                ent.insert(0, f"ma={p1.get()},std={p2.get()}")
                svars[sk]['p_ma'] = p1; svars[sk]['p_std'] = p2; svars[sk]['params_ent'] = ent
            elif sk == 'momentum_trend':
                p1 = tk.StringVar(value=str(params.get('ma_period', '100')))
                p2 = tk.StringVar(value=str(params.get('roc_period', '20')))
                ent = ttk.Entry(tbl, width=18)
                ent.insert(0, f"ma={p1.get()},roc={p2.get()}")
                svars[sk]['p_ma'] = p1; svars[sk]['p_roc'] = p2; svars[sk]['params_ent'] = ent
            elif sk in ('breakout', 'donchian_channel'):
                key = 'lookback' if sk == 'breakout' else 'channel_length'
                p1 = tk.StringVar(value=str(params.get(key, '20')))
                ent = ttk.Entry(tbl, width=18)
                ent.insert(0, f"{key}={p1.get()}")
                svars[sk]['p_len'] = p1; svars[sk]['params_ent'] = ent
            else:
                ent = ttk.Entry(tbl, width=18)
                svars[sk]['params_ent'] = ent
            ent.grid(row=rr, column=2, sticky='w')
            av_p = tk.StringVar(value=str(atr.get('period', def_atr['period'])))
            av_sl = tk.StringVar(value=str(atr.get('sl_mult', def_atr['sl_mult'])))
            av_tp = tk.StringVar(value=str(atr.get('tp_mult', def_atr['tp_mult'])))
            av_pr = tk.StringVar(value=str(atr.get('priority', def_atr['priority'])))
            svars[sk]['atr_p'] = av_p; svars[sk]['atr_sl'] = av_sl; svars[sk]['atr_tp'] = av_tp; svars[sk]['atr_pr'] = av_pr
            atr_frm = ttk.Frame(tbl)
            atr_frm.grid(row=rr, column=3, sticky='w')
            ttk.Entry(atr_frm, width=4, textvariable=av_p).pack(side=tk.LEFT, padx=2)
            ttk.Entry(atr_frm, width=4, textvariable=av_sl).pack(side=tk.LEFT, padx=2)
            ttk.Entry(atr_frm, width=4, textvariable=av_tp).pack(side=tk.LEFT, padx=2)
            ttk.Entry(atr_frm, width=4, textvariable=av_pr).pack(side=tk.LEFT, padx=2)
            rr += 1

        row += 1
        stats_box = ttk.LabelFrame(frm, text="Live Stats (per strategy)", padding=6)
        stats_box.grid(row=row, column=0, columnspan=7, sticky='nsew', pady=6)
        frm.rowconfigure(row, weight=1)
        frm.columnconfigure(0, weight=1)
        cols = ('Strategy', 'Side', 'Volume', 'Avg Price', 'Unrealized P/L', 'Positions')
        tree = ttk.Treeview(stats_box, columns=cols, show='headings', height=8)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, width=110, anchor='center')
        tree.pack(fill=tk.BOTH, expand=True)

        ui[symbol] = {
            'frame': frm,
            'active': active,
            'tf_str': tf_str,
            'bars': bars_var,
            'lots': lots_var,
            'tw_247': tw_247,
            'tw_start': tw_start,
            'tw_end': tw_end,
            'htf_enabled': htf_enabled,
            'htf_tf': htf_tf,
            'htf_ma': htf_ma,
            'strats': svars,
            'stats_tree': tree,
        }

    for sym in symbols_list:
        build_symbol_tab(sym)

    def sync_symbol_from_cfg(symbol, cfg_snapshot):
        if symbol not in ui:
            build_symbol_tab(symbol)
        sv = ui[symbol]
        sym_cfg = (cfg_snapshot.get('symbols', {}).get(symbol, {}) or {})
        sym_meta = sym_cfg.get('_meta', {}) if isinstance(sym_cfg, dict) else {}
        dhtf = (sym_meta.get('htf_filter', {}) or {}) 

        # active: keep as-is (user control)
        # timeframe from symbol meta
        tf_code = sym_meta.get('timeframe_code') or sym_meta.get('timeframe')
        if tf_code is not None:
            try:
                sv['tf_str'].set(timeframe_code_to_str(int(tf_code)))
            except Exception:
                pass
        # bars: keep current
        # lots & trading window from symbol meta
        try:
            lots_val = float((sym_meta.get('risk') or {}).get('lots'))
            sv['lots'].set(str(lots_val))
        except Exception:
            pass
        tw = sym_meta.get('trading_window', {}) or {}
        sv['tw_247'].set(bool(tw.get('trade_24_7', True)))
        sv['tw_start'].set(str(int(tw.get('start_hour', 0))))
        sv['tw_end'].set(str(int(tw.get('end_hour', 24))))

        # HTF
        sv['htf_enabled'].set(bool(dhtf.get('enabled', False)))
        # dhtf may have timeframe_code or timeframe string
        htf_code = dhtf.get('timeframe_code', None)
        if htf_code is not None:
            sv['htf_tf'].set(timeframe_code_to_str(int(htf_code)))
        else:
            if dhtf.get('timeframe'):
                sv['htf_tf'].set(str(dhtf.get('timeframe')))
        sv['htf_ma'].set(str(int(dhtf.get('ma_period', 50))))

        # Strategies: update enabled, params, ATR. Respect bestconfig 'active' flag; do not auto-enable missing entries.
        def set_entry(ent: tk.Entry, text: str):
            try:
                ent.delete(0, tk.END)
                ent.insert(0, text)
            except Exception:
                pass
        for sk, d in (sv['strats'] or {}).items():
            cent = sym_cfg.get(sk, {}) or {}
            d.get('enabled').set(bool(cent.get('active', False)) if cent else False)
            params = cent.get('strategy_params', {}) or {}
            atr = cent.get('atr', {}) or {}
            if sk == 'ma_crossover':
                d['p_fast'].set(str(int(params.get('fast', d['p_fast'].get() or 5))))
                d['p_slow'].set(str(int(params.get('slow', d['p_slow'].get() or 20))))
                set_entry(d['params_ent'], f"fast={d['p_fast'].get()},slow={d['p_slow'].get()}")
            elif sk == 'mean_reversion':
                d['p_ma'].set(str(int(params.get('ma_period', d['p_ma'].get() or 10))))
                d['p_std'].set(str(int(params.get('num_std', d['p_std'].get() or 1))))
                set_entry(d['params_ent'], f"ma={d['p_ma'].get()},std={d['p_std'].get()}")
            elif sk == 'momentum_trend':
                d['p_ma'].set(str(int(params.get('ma_period', d['p_ma'].get() or 100))))
                d['p_roc'].set(str(int(params.get('roc_period', d['p_roc'].get() or 20))))
                set_entry(d['params_ent'], f"ma={d['p_ma'].get()},roc={d['p_roc'].get()}")
            elif sk in ('breakout', 'donchian_channel'):
                key = 'lookback' if sk == 'breakout' else 'channel_length'
                d['p_len'].set(str(int(params.get(key, d['p_len'].get() or 20))))
                set_entry(d['params_ent'], f"{key}={d['p_len'].get()}")
            d['atr_p'].set(str(int(atr.get('period', d['atr_p'].get() or 21))))
            d['atr_sl'].set(str(float(atr.get('sl_mult', d['atr_sl'].get() or 1.5))))
            d['atr_tp'].set(str(float(atr.get('tp_mult', d['atr_tp'].get() or 2.5))))
            d['atr_pr'].set(str(atr.get('priority', d['atr_pr'].get() or 'SL')))
        log(f"Finished sync for {symbol} (lots={sv['lots'].get()} tf={sv['tf_str'].get()} htf={'on' if sv['htf_enabled'].get() else 'off'})")

    # Perform initial sync after definition to ensure function exists
    for sym in symbols_list:
        try:
            sync_symbol_from_cfg(sym, cfg)
        except Exception as e:
            log(f"Sync error for {sym}: {e}")

    def get_state_snapshot():
        state = {'symbols': {}}
        for symbol, sv in ui.items():
            tf_code = parse_timeframe_input(sv['tf_str'].get())
            try:
                lookback = int(float(sv['bars'].get()))
            except Exception:
                lookback = 400
            conf = {
                'active': bool(sv['active'].get()),
                'timeframe_code': int(tf_code),
                'lookback_bars': lookback,
                'lots': float(sv['lots'].get() or 0.1),
                'trading_window': {
                    'trade_24_7': bool(sv['tw_247'].get()),
                    'start_hour': int(float(sv['tw_start'].get() or 0)),
                    'end_hour': int(float(sv['tw_end'].get() or 24)),
                },
                'htf_filter': {
                    'enabled': bool(sv['htf_enabled'].get()),
                    'timeframe_code': int(parse_timeframe_input(sv['htf_tf'].get() or 'H1')),
                    'ma_period': int(float(sv['htf_ma'].get() or 50)),
                },
                'strategies': {}
            }
            for sk, d in (sv['strats'] or {}).items():
                en = bool(d.get('enabled').get())
                if sk == 'ma_crossover':
                    params = {'fast': int(float(d['p_fast'].get() or 5)), 'slow': int(float(d['p_slow'].get() or 20))}
                elif sk == 'mean_reversion':
                    params = {'ma_period': int(float(d['p_ma'].get() or 10)), 'num_std': int(float(d['p_std'].get() or 1))}
                elif sk == 'momentum_trend':
                    params = {'ma_period': int(float(d['p_ma'].get() or 100)), 'roc_period': int(float(d['p_roc'].get() or 20))}
                elif sk in ('breakout', 'donchian_channel'):
                    key = 'lookback' if sk == 'breakout' else 'channel_length'
                    params = {key: int(float(d['p_len'].get() or 20))}
                else:
                    params = {}
                atr = {
                    'period': int(float(d['atr_p'].get() or 21)),
                    'sl_mult': float(d['atr_sl'].get() or 1.5),
                    'tp_mult': float(d['atr_tp'].get() or 2.5),
                    'priority': str(d['atr_pr'].get() or 'SL')
                }
                conf['strategies'][sk] = {'enabled': en, 'params': params, 'atr': atr}
            state['symbols'][symbol] = conf
        return state

    ctl = LiveController(get_state_snapshot, log)

    btns = ttk.Frame(root, padding=8)
    btns.pack(fill=tk.X)
    ttk.Button(btns, text="Start All", command=ctl.start).pack(side=tk.LEFT, padx=4)
    ttk.Button(btns, text="Stop All", command=ctl.stop).pack(side=tk.LEFT, padx=4)

    def reload_cfg():
        nonlocal cfg
        new_cfg = load_bestconfig()
        if not new_cfg:
            log("Reload: bestconfig.json not found or invalid")
            return
        cfg = new_cfg
        new_symbols = available_symbols(cfg) or []
        if not new_symbols:
            log("Reloaded bestconfig.json (no symbols)")
        else:
            for sym in new_symbols:
                sync_symbol_from_cfg(sym, cfg)
            log("Reloaded bestconfig.json and updated UI")

    ttk.Button(btns, text="Reload bestconfig", command=reload_cfg).pack(side=tk.LEFT, padx=4)

    # Countdown timer UI (circular)
    timer_frame = ttk.Frame(root, padding=8)
    timer_frame.pack(fill=tk.X)
    canvas = tk.Canvas(timer_frame, width=80, height=80)
    canvas.pack(side=tk.LEFT)
    cd_label = ttk.Label(timer_frame, text="Next run in --s")
    cd_label.pack(side=tk.LEFT, padx=8)

    def draw_countdown(secs_left: int, total: int):
        canvas.delete("all")
        # Circle background
        canvas.create_oval(10, 10, 70, 70, outline="#ddd", width=8)
        # Foreground arc based on remaining fraction
        frac = max(0.0, min(1.0, secs_left / float(total if total else 1)))
        extent = 360 * frac
        # Draw arc from top (90 deg), negative extent to go clockwise
        canvas.create_arc(10, 10, 70, 70, start=90, extent=-extent, style=tk.ARC, outline="#2b8a3e", width=8)
        # Text
        canvas.create_text(40, 40, text=str(secs_left), font=("Segoe UI", 12))
        cd_label.configure(text=f"Next run in {secs_left}s")

    def update_countdown_and_stats():
        # Countdown
        try:
            interval = int(getattr(ctl, 'interval_secs', 60) or 60)
            nxt = getattr(ctl, 'next_run_ts', None)
            if nxt is not None:
                now = datetime.now()
                remaining = int(max(0, (nxt - now).total_seconds()))
                if remaining > interval:
                    remaining = interval
                draw_countdown(remaining, interval)
        except Exception:
            pass
        # Live Stats refresh
        try:
            for symbol, sv in ui.items():
                tree = sv.get('stats_tree')
                if tree is None:
                    continue
                # Snapshot stats under lock
                with ctl.lock:
                    sym_stats = (ctl.stats or {}).get(symbol, {})
                # Rebuild tree: clear and insert rows
                for i in tree.get_children():
                    tree.delete(i)
                for strat, entry in sym_stats.items():
                    for side in ('buy', 'sell'):
                        e = entry.get(side, {})
                        if e.get('positions', 0) <= 0 and e.get('volume', 0.0) <= 0:
                            continue
                        vol = e.get('volume', 0.0)
                        avgp = e.get('avg_price', 0.0)
                        upl = e.get('unrealized_pl', 0.0)
                        tree.insert('', tk.END, values=(strat, side.upper(), f"{vol:.2f}", f"{avgp:.5f}", f"{upl:.2f}", int(e.get('positions', 0))))
        except Exception:
            pass
        root.after(1000, update_countdown_and_stats)

    root.protocol("WM_DELETE_WINDOW", lambda: (ctl.stop(), root.destroy()))
    # Start UI updaters
    update_countdown_and_stats()
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
