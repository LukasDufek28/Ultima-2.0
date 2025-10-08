import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from agents import TradingAgent
from itertools import product
import matplotlib.pyplot as plt  # NEW: for equity charts
import tkinter as tk  # NEW: GUI
from tkinter import ttk, messagebox  # NEW: GUI
import os, json  # NEW: persistence
import copy  # NEW: for deep copies of saved results
import tkinter.font as tkfont  # NEW: monospaced font for alignment
import time  # NEW: high-precision timer

# Settings
timeframe = mt5.TIMEFRAME_M15
symbols = ["XAUUSD"]  # Add more as needed
start_date = datetime.now() - timedelta(days=60)  # 60 days backtest
end_date = datetime.now()

# ATR exit settings
ATR_PERIOD = 21
SL_ATR_MULTIPLIER = 1.5
TP_ATR_MULTIPLIER = 2.5
INTRABAR_PRIORITY = 'SL'  # 'SL' or 'TP' if both hit within the same bar

# Strategy parameters
ma_crossover_fast = 5
ma_crossover_slow = 20
meanrev_ma_period = 10
meanrev_num_std = 1
momentum_ma_period = 100
momentum_roc_period = 20
breakout_lookback = 20

# NEW: trading hours control
TRADE_24_7 = True       # set to False to restrict entries to a time window
TRADING_START_HOUR = 8  # 0..23 (ignored if TRADE_24_7=True)
TRADING_END_HOUR = 20   # 1..24 (24 means end-of-day; ignored if TRADE_24_7=True)

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

# NEW: where to save last-used params (alongside this file)
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "ultima_backtester_last_params.json")

def load_last_params(path=CONFIG_PATH):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_last_params(state: dict, path=CONFIG_PATH):
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception:
        pass

# NEW: parse "15M", "1H", "1D" (or numeric MT5 code) to MT5 timeframe constant
def parse_timeframe_input(val):
    s = str(val).strip().upper()
    # Try numeric MT5 code directly
    try:
        return int(float(s))
    except Exception:
        pass
    # Accept "15M" / "M15" / "1H" / "H1" / "1D" / "D1"
    def extract_num_unit(txt):
        if not txt:
            return None, None
        if txt[-1] in ('M', 'H', 'D'):
            num = txt[:-1]
            unit = txt[-1]
        elif txt[0] in ('M', 'H', 'D'):
            num = txt[1:]
            unit = txt[0]
        else:
            return None, None
        try:
            return int(num), unit
        except Exception:
            return None, None

    n, u = extract_num_unit(s)
    if n is None or u is None:
        # Fallback to default if unparsable
        return timeframe

    # Maps of supported MT5 timeframes
    m_map = {
        1: mt5.TIMEFRAME_M1, 2: mt5.TIMEFRAME_M2, 3: mt5.TIMEFRAME_M3, 4: mt5.TIMEFRAME_M4,
        5: mt5.TIMEFRAME_M5, 6: mt5.TIMEFRAME_M6, 10: mt5.TIMEFRAME_M10, 12: mt5.TIMEFRAME_M12,
        15: mt5.TIMEFRAME_M15, 20: mt5.TIMEFRAME_M20, 30: mt5.TIMEFRAME_M30
    }
    h_map = {
        1: mt5.TIMEFRAME_H1, 2: mt5.TIMEFRAME_H2, 3: mt5.TIMEFRAME_H3, 4: mt5.TIMEFRAME_H4,
        6: mt5.TIMEFRAME_H6, 8: mt5.TIMEFRAME_H8, 12: mt5.TIMEFRAME_H12
    }
    d_map = {1: mt5.TIMEFRAME_D1}

    def choose(mapping, num):
        if num in mapping:
            return mapping[num]
        # pick the next available higher, else the highest available
        keys = sorted(mapping.keys())
        for k in keys:
            if num <= k:
                return mapping[k]
        return mapping[keys[-1]]

    if u == 'M':
        return choose(m_map, n)
    if u == 'H':
        return choose(h_map, n)
    if u == 'D':
        return choose(d_map, n)
    return timeframe

# NEW: enable GUI and provide GUI implementation
USE_GUI = True

def gui_main():
    root = tk.Tk()
    root.title("Ultima Backtester")
    # Wider and resizable
    root.geometry("1350x900")
    root.rowconfigure(3, weight=1)
    root.columnconfigure(0, weight=1)

    # State holders
    last_results = {'value': {}}
    opt_store = {'best': {}}
    # NEW: optimization control state
    opt_state = {'stop': False}

    # ===== LAYOUT: grid of “cells” =====
    # Create a container grid 2x2 for settings, plus a bottom output row
    grid = ttk.Frame(root, padding=10)
    grid.grid(row=0, column=0, sticky="nsew")
    for r in range(2):
        grid.rowconfigure(r, weight=0)
    grid.columnconfigure(0, weight=1)
    grid.columnconfigure(1, weight=1)

    # Cells/sections
    frm_bt = ttk.LabelFrame(grid, text="Backtest", padding=10)
    frm_bt.grid(row=0, column=0, sticky="nsew", padx=6, pady=6)
    frm_strat = ttk.LabelFrame(grid, text="Strategies", padding=10)
    frm_strat.grid(row=0, column=1, sticky="nsew", padx=6, pady=6)
    frm_risk = ttk.LabelFrame(grid, text="Risk & ATR", padding=10)
    frm_risk.grid(row=1, column=0, sticky="nsew", padx=6, pady=6)
    frm_opt = ttk.LabelFrame(grid, text="Optimization", padding=10)
    frm_opt.grid(row=1, column=1, sticky="nsew", padx=6, pady=6)

    # NEW: High Timeframe Filter
    frm_htf = ttk.LabelFrame(grid, text="High TF Filter", padding=10)
    frm_htf.grid(row=3, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)
    use_htf = tk.BooleanVar(value=False)
    ttk.Checkbutton(frm_htf, text="Enable HTF bias filter", variable=use_htf).grid(row=0, column=0, sticky="w")
    ttk.Label(frm_htf, text="HTF Timeframe (e.g., H1, H4, D1):").grid(row=0, column=1, sticky="e")
    htf_tf_var = tk.StringVar(value="H1")
    ttk.Entry(frm_htf, textvariable=htf_tf_var, width=8).grid(row=0, column=2, sticky="w", padx=4)
    ttk.Label(frm_htf, text="HTF MA period:").grid(row=0, column=3, sticky="e")
    htf_ma_var = tk.StringVar(value="50")
    ttk.Entry(frm_htf, textvariable=htf_ma_var, width=6).grid(row=0, column=4, sticky="w", padx=4)

    # NEW: Optimization controls (objective + ATR tuning)
    objective = tk.StringVar(value="highest_total_profit")
    ttk.Label(frm_opt, text="Objective:").grid(row=0, column=0, sticky="e")
    ttk.Combobox(
        frm_opt,
        textvariable=objective,
        values=[
            "highest_total_profit",
            "highest_final_balance",
            "lowest_max_drawdown",
            "highest_win_rate",
            "highest_expectancy",
            "highest_trade_sharpe"
        ],
        state="readonly",
        width=24
    ).grid(row=0, column=1, sticky="w", padx=4)
    opt_atr = tk.BooleanVar(value=False)
    ttk.Checkbutton(frm_opt, text="Tune ATR params", variable=opt_atr).grid(row=0, column=2, sticky="w", padx=4)

    # NEW: Optimization constraints
    min_trades_var = tk.StringVar(value="1")
    min_profit_var = tk.StringVar(value="0")
    min_winrate_var = tk.StringVar(value="0")
    min_expectancy_var = tk.StringVar(value="0")
    min_sharpe_var = tk.StringVar(value="0")
    ttk.Label(frm_opt, text="Min Trades:").grid(row=1, column=0, sticky="e")
    ttk.Entry(frm_opt, textvariable=min_trades_var, width=8).grid(row=1, column=1, sticky="w", padx=4)
    ttk.Label(frm_opt, text="Min Profit:").grid(row=1, column=2, sticky="e")
    ttk.Entry(frm_opt, textvariable=min_profit_var, width=10).grid(row=1, column=3, sticky="w", padx=4)
    ttk.Label(frm_opt, text="Min WinRate%:").grid(row=1, column=4, sticky="e")
    ttk.Entry(frm_opt, textvariable=min_winrate_var, width=8).grid(row=1, column=5, sticky="w", padx=4)
    # Additional constraints row
    ttk.Label(frm_opt, text="Min Expectancy:").grid(row=2, column=0, sticky="e")
    ttk.Entry(frm_opt, textvariable=min_expectancy_var, width=8).grid(row=2, column=1, sticky="w", padx=4)
    ttk.Label(frm_opt, text="Min TradeSharpe:").grid(row=2, column=2, sticky="e")
    ttk.Entry(frm_opt, textvariable=min_sharpe_var, width=10).grid(row=2, column=3, sticky="w", padx=4)

    frm_actions = ttk.Frame(root, padding=10)
    frm_actions.grid(row=1, column=0, sticky="ew")
    frm_out = ttk.LabelFrame(root, text="Output", padding=10)
    frm_out.grid(row=2, column=0, sticky="nsew")
    # CHANGED: two rows (header + text row) and two columns (left/right panes)
    frm_out.rowconfigure(1, weight=1)
    frm_out.columnconfigure(0, weight=1)
    frm_out.columnconfigure(1, weight=1)

    # ===== Backtest cell =====
    default_symbols = ",".join(symbols)
    ttk.Label(frm_bt, text="Symbols (comma-separated):").grid(row=0, column=0, sticky="e")
    sym_var = tk.StringVar(value=default_symbols)
    ttk.Entry(frm_bt, textvariable=sym_var, width=50).grid(row=0, column=1, sticky="ew", padx=5)

    ttk.Label(frm_bt, text="Timeframe (e.g., 15M, 1H, 1D):").grid(row=0, column=2, sticky="e")
    tf_var = tk.StringVar(value=str(timeframe))
    ttk.Entry(frm_bt, textvariable=tf_var, width=8).grid(row=0, column=3, sticky="w", padx=5)

    ttk.Label(frm_bt, text="Days:").grid(row=0, column=4, sticky="e")
    days_var = tk.StringVar(value="60")
    ttk.Entry(frm_bt, textvariable=days_var, width=6).grid(row=0, column=5, sticky="w", padx=5)

    ttk.Label(frm_bt, text="Initial Balance:").grid(row=1, column=0, sticky="e")
    init_bal = tk.StringVar(value="100")
    ttk.Entry(frm_bt, textvariable=init_bal, width=10).grid(row=1, column=1, sticky="w")

    trade_247 = tk.BooleanVar(value=TRADE_24_7)
    ttk.Checkbutton(frm_bt, text="Trade 24/7", variable=trade_247).grid(row=1, column=2, sticky="w")
    ttk.Label(frm_bt, text="Start Hour:").grid(row=1, column=3, sticky="e")
    start_h = tk.StringVar(value=str(TRADING_START_HOUR))
    ttk.Entry(frm_bt, textvariable=start_h, width=4).grid(row=1, column=4, sticky="w")
    ttk.Label(frm_bt, text="End Hour:").grid(row=1, column=5, sticky="e")
    end_h = tk.StringVar(value=str(TRADING_END_HOUR))
    ttk.Entry(frm_bt, textvariable=end_h, width=4).grid(row=1, column=6, sticky="w")

    # ===== Strategies cell =====
    ttk.Label(frm_strat, text="Enable:").grid(row=0, column=0, sticky="w")
    strat_vars = {
        'ma_crossover': tk.BooleanVar(value=True),
        'mean_reversion': tk.BooleanVar(value=True),
        'momentum_trend': tk.BooleanVar(value=True),
        'breakout': tk.BooleanVar(value=True),
        'donchian_channel': tk.BooleanVar(value=True),
    }
    c = 0
    for name, var in strat_vars.items():
        ttk.Checkbutton(frm_strat, text=name, variable=var).grid(row=0, column=1 + c, sticky="w", padx=4)
        c += 1

    # MA crossover params
    ttk.Label(frm_strat, text="MA Cross (fast, slow):").grid(row=1, column=0, sticky="e")
    mac_fast = tk.StringVar(value=str(ma_crossover_fast))
    mac_slow = tk.StringVar(value=str(ma_crossover_slow))
    ttk.Entry(frm_strat, textvariable=mac_fast, width=6).grid(row=1, column=1, sticky="w")
    ttk.Entry(frm_strat, textvariable=mac_slow, width=6).grid(row=1, column=2, sticky="w")

    # Mean reversion params
    ttk.Label(frm_strat, text="MeanRev (MA, Std):").grid(row=1, column=3, sticky="e")
    mr_ma = tk.StringVar(value=str(meanrev_ma_period))
    mr_std = tk.StringVar(value=str(meanrev_num_std))
    ttk.Entry(frm_strat, textvariable=mr_ma, width=6).grid(row=1, column=4, sticky="w")
    ttk.Entry(frm_strat, textvariable=mr_std, width=6).grid(row=1, column=5, sticky="w")

    # Momentum params
    ttk.Label(frm_strat, text="Momentum (MA, ROC):").grid(row=2, column=0, sticky="e")
    mom_ma = tk.StringVar(value=str(momentum_ma_period))
    mom_roc = tk.StringVar(value=str(momentum_roc_period))
    ttk.Entry(frm_strat, textvariable=mom_ma, width=6).grid(row=2, column=1, sticky="w")
    ttk.Entry(frm_strat, textvariable=mom_roc, width=6).grid(row=2, column=2, sticky="w")

    # Breakout / Donchian param
    ttk.Label(frm_strat, text="Breakout/Donchian (lookback):").grid(row=2, column=3, sticky="e")
    brk_lookback = tk.StringVar(value=str(breakout_lookback))
    ttk.Entry(frm_strat, textvariable=brk_lookback, width=6).grid(row=2, column=4, sticky="w")
    # Donchian advanced params
    ttk.Label(frm_strat, text="Donchian (exit_len, confirm, ATRbuf):").grid(row=3, column=3, sticky="e")
    don_exit_len = tk.StringVar(value="10")
    don_confirm = tk.StringVar(value="2")
    don_atr_buf = tk.StringVar(value="0.5")
    ttk.Entry(frm_strat, textvariable=don_exit_len, width=6).grid(row=3, column=4, sticky="w")
    ttk.Entry(frm_strat, textvariable=don_confirm, width=6).grid(row=3, column=5, sticky="w")
    ttk.Entry(frm_strat, textvariable=don_atr_buf, width=6).grid(row=3, column=6, sticky="w")

    # ===== Risk & ATR cell =====
    ttk.Label(frm_risk, text="ATR Period:").grid(row=0, column=0, sticky="e")
    atr_p = tk.StringVar(value=str(ATR_PERIOD))
    ttk.Entry(frm_risk, textvariable=atr_p, width=6).grid(row=0, column=1, sticky="w")
    ttk.Label(frm_risk, text="SL x ATR:").grid(row=0, column=2, sticky="e")
    sl_x = tk.StringVar(value=str(SL_ATR_MULTIPLIER))
    ttk.Entry(frm_risk, textvariable=sl_x, width=6).grid(row=0, column=3, sticky="w")
    ttk.Label(frm_risk, text="TP x ATR:").grid(row=0, column=4, sticky="e")
    tp_x = tk.StringVar(value=str(TP_ATR_MULTIPLIER))
    ttk.Entry(frm_risk, textvariable=tp_x, width=6).grid(row=0, column=5, sticky="w")
    ttk.Label(frm_risk, text="Intrabar priority:").grid(row=0, column=6, sticky="e")
    prio = tk.StringVar(value=INTRABAR_PRIORITY)
    ttk.Combobox(frm_risk, textvariable=prio, values=['SL', 'TP'], width=5, state="readonly").grid(row=0, column=7, sticky="w")
    ttk.Label(frm_risk, text="Lots:").grid(row=1, column=0, sticky="e")
    lots_var = tk.StringVar(value="0.1")
    ttk.Entry(frm_risk, textvariable=lots_var, width=6).grid(row=1, column=1, sticky="w")

    # ===== Per-Strategy ATR (optional overrides) =====
    frm_psatr = ttk.LabelFrame(grid, text="Per-Strategy ATR (optional overrides)", padding=10)
    frm_psatr.grid(row=2, column=0, columnspan=2, sticky="nsew", padx=6, pady=6)

    ttk.Label(frm_psatr, text="Strategy").grid(row=0, column=0, sticky="w")
    ttk.Label(frm_psatr, text="ATR Period").grid(row=0, column=1, sticky="w")
    ttk.Label(frm_psatr, text="SL x ATR").grid(row=0, column=2, sticky="w")
    ttk.Label(frm_psatr, text="TP x ATR").grid(row=0, column=3, sticky="w")

    # StringVars default to empty -> use general defaults
    mac_atr_p, mac_atr_sl, mac_atr_tp = tk.StringVar(value=""), tk.StringVar(value=""), tk.StringVar(value="")
    mr_atr_p,  mr_atr_sl,  mr_atr_tp  = tk.StringVar(value=""), tk.StringVar(value=""), tk.StringVar(value="")
    mom_atr_p, mom_atr_sl, mom_atr_tp = tk.StringVar(value=""), tk.StringVar(value=""), tk.StringVar(value="")
    brk_atr_p, brk_atr_sl, brk_atr_tp = tk.StringVar(value=""), tk.StringVar(value=""), tk.StringVar(value="")
    don_atr_p, don_atr_sl, don_atr_tp = tk.StringVar(value=""), tk.StringVar(value=""), tk.StringVar(value="")

    def _row(frm, r, name, v_p, v_sl, v_tp):
        ttk.Label(frm, text=name).grid(row=r, column=0, sticky="w")
        ttk.Entry(frm, textvariable=v_p,  width=6).grid(row=r, column=1, sticky="w", padx=2)
        ttk.Entry(frm, textvariable=v_sl, width=6).grid(row=r, column=2, sticky="w", padx=2)
        ttk.Entry(frm, textvariable=v_tp, width=6).grid(row=r, column=3, sticky="w", padx=2)

    _row(frm_psatr, 1, "MA Crossover",      mac_atr_p, mac_atr_sl, mac_atr_tp)
    _row(frm_psatr, 2, "Mean Reversion",    mr_atr_p,  mr_atr_sl,  mr_atr_tp)
    _row(frm_psatr, 3, "Momentum/Trend",    mom_atr_p, mom_atr_sl, mom_atr_tp)
    _row(frm_psatr, 4, "Breakout",          brk_atr_p, brk_atr_sl, brk_atr_tp)
    _row(frm_psatr, 5, "Donchian Channel",  don_atr_p, don_atr_sl, don_atr_tp)

    # ===== Output (colored) split in two panes =====
    ttk.Label(frm_out, text="Current Output").grid(row=0, column=0, sticky="w")
    ttk.Label(frm_out, text="Saved Results").grid(row=0, column=1, sticky="w")

    # Left pane: current output
    txt = tk.Text(frm_out, width=80, height=22)
    txt.grid(row=1, column=0, sticky="nsew", padx=(0, 6))
    txt.tag_configure("green", foreground="#0a7d00")
    txt.tag_configure("red", foreground="#b00020")
    txt.tag_configure("info", foreground="#004a9f")

    # Right pane: saved results
    txt_saved = tk.Text(frm_out, width=80, height=22)
    txt_saved.grid(row=1, column=1, sticky="nsew", padx=(6, 0))
    txt_saved.tag_configure("green", foreground="#0a7d00")
    txt_saved.tag_configure("red", foreground="#b00020")
    txt_saved.tag_configure("info", foreground="#004a9f")

    # NEW: use a fixed-width font so column padding aligns reliably
    try:
        fixed_font = tkfont.nametofont("TkFixedFont")
    except Exception:
        fixed_font = tkfont.Font(family="Courier New", size=10)
    txt.configure(font=fixed_font)
    txt_saved.configure(font=fixed_font)

    # NEW: keep a structured last saved snapshot for comparisons
    saved_state = {"results": None, "ts": None}

    # NEW: helpers to compare metrics and render saved pane with colors
    def metric_tag(key, saved_val, current_val):
        try:
            if saved_val is None or current_val is None:
                return None
            # equal -> no tag
            if abs(float(saved_val) - float(current_val)) < 1e-12:
                return None
            if key == "max_drawdown":
                # lower drawdown is better
                return "green" if saved_val < current_val else "red"
            else:
                # higher is better for the rest
                return "green" if saved_val > current_val else "red"
        except Exception:
            return None

    def render_saved_snapshot(snapshot_results, compare_to=None):
        # compare_to: current last_results dict or None
        txt_saved.delete("1.0", tk.END)
        hdr = f"=== Saved at {saved_state['ts']} ===\n" if saved_state.get("ts") else "=== Saved ===\n"
        txt_saved.insert(tk.END, hdr, "info")
        if not snapshot_results:
            return

        # NEW: aligned rendering
        metrics_spec = [
            ("trades", "Trades"),
            ("win_rate", "WinRate"),
            ("total_profit", "Profit"),
            ("final_balance", "Final"),
            ("max_drawdown", "MaxDD"),
        ]
        rows = []
        for symbol in sorted(snapshot_results.keys()):
            sym_saved = snapshot_results.get(symbol, {})
            sym_curr = (compare_to or {}).get(symbol, {}) if compare_to else {}
            for strat_name, r_saved in sym_saved.items():
                r_curr = sym_curr.get(strat_name, {})
                left = f"{symbol} {strat_name}: "
                rows.append((left, r_saved, r_curr))

        # compute widths
        left_w = max((len(left) for left, _, _ in rows), default=0)
        col_w = {}
        for key, label in metrics_spec:
            maxw = 0
            for _, r_saved, _ in rows:
                sv = r_saved.get(key)
                if key == "win_rate" and sv is not None:
                    val_str = f"{float(sv):.2f}%"
                elif isinstance(sv, (int, np.integer)) and key == "trades":
                    val_str = f"{int(sv)}"
                elif sv is not None:
                    val_str = f"{float(sv):.2f}"
                else:
                    val_str = "NA"
                maxw = max(maxw, len(f"{label}={val_str}"))
            col_w[key] = maxw

        # render aligned rows
        for left, r_saved, r_curr in rows:
            txt_saved.insert(tk.END, left.ljust(left_w))
            spacer_between = "  "
            for idx, (key, label) in enumerate(metrics_spec):
                # label
                lbl = f"{label}="
                txt_saved.insert(tk.END, lbl)
                # value string
                sv = r_saved.get(key)
                cv = r_curr.get(key) if r_curr else None
                if key == "win_rate" and sv is not None:
                    vstr = f"{float(sv):.2f}%"
                elif isinstance(sv, (int, np.integer)) and key == "trades":
                    vstr = f"{int(sv)}"
                elif sv is not None:
                    vstr = f"{float(sv):.2f}"
                else:
                    vstr = "NA"
                tag = metric_tag(key, sv, cv) if compare_to else None
                if tag:
                    txt_saved.insert(tk.END, vstr, tag)
                else:
                    txt_saved.insert(tk.END, vstr)
                # pad to column width (label + value)
                pad = col_w[key] - len(lbl) - len(vstr)
                if pad > 0:
                    txt_saved.insert(tk.END, " " * pad)
                # spacer to next column
                if idx < len(metrics_spec) - 1:
                    txt_saved.insert(tk.END, spacer_between)
            txt_saved.insert(tk.END, "\n")
        txt_saved.see(tk.END)

    # NEW: render current snapshot with same coloring logic as saved,
    # comparing current metrics vs saved snapshot (if present).
    def render_current_snapshot(current_results, compare_to=None):
        # compare_to: saved results dict or None
        txt.delete("1.0", tk.END)
        txt.insert(tk.END, "=== Current ===\n", "info")
        if not current_results:
            return

        metrics_spec = [
            ("trades", "Trades"),
            ("win_rate", "WinRate"),
            ("total_profit", "Profit"),
            ("final_balance", "Final"),
            ("max_drawdown", "MaxDD"),
        ]
        rows = []
        for symbol in sorted(current_results.keys()):
            sym_curr = current_results.get(symbol, {})
            sym_saved = (compare_to or {}).get(symbol, {}) if compare_to else {}
            for strat_name, r_curr in sym_curr.items():
                r_saved = sym_saved.get(strat_name, {})
                left = f"{symbol} {strat_name}: "
                rows.append((left, r_curr, r_saved))

        # compute widths
        left_w = max((len(left) for left, _, _ in rows), default=0)
        col_w = {}
        for key, label in metrics_spec:
            maxw = 0
            for _, r_curr, _ in rows:
                cv = r_curr.get(key)
                if key == "win_rate" and cv is not None:
                    val_str = f"{float(cv):.2f}%"
                elif isinstance(cv, (int, np.integer)) and key == "trades":
                    val_str = f"{int(cv)}"
                elif cv is not None:
                    val_str = f"{float(cv):.2f}"
                else:
                    val_str = "NA"
                maxw = max(maxw, len(f"{label}={val_str}"))
            col_w[key] = maxw

        # render aligned rows
        for left, r_curr, r_saved in rows:
            txt.insert(tk.END, left.ljust(left_w))
            spacer_between = "  "
            for idx, (key, label) in enumerate(metrics_spec):
                lbl = f"{label}="
                txt.insert(tk.END, lbl)
                cv = r_curr.get(key)
                sv = r_saved.get(key) if r_saved else None
                if key == "win_rate" and cv is not None:
                    vstr = f"{float(cv):.2f}%"
                elif isinstance(cv, (int, np.integer)) and key == "trades":
                    vstr = f"{int(cv)}"
                elif cv is not None:
                    vstr = f"{float(cv):.2f}"
                else:
                    vstr = "NA"
                # Use same metric_tag but swap args so "current vs saved" gives green when current is better
                tag = metric_tag(key, cv, sv) if compare_to else None
                if tag:
                    txt.insert(tk.END, vstr, tag)
                else:
                    txt.insert(tk.END, vstr)
                pad = col_w[key] - len(lbl) - len(vstr)
                if pad > 0:
                    txt.insert(tk.END, " " * pad)
                if idx < len(metrics_spec) - 1:
                    txt.insert(tk.END, spacer_between)
            txt.insert(tk.END, "\n")
        txt.see(tk.END)

    def append_line(line, tag=None):
        if tag:
            txt.insert(tk.END, line + "\n", tag)
        else:
            txt.insert(tk.END, line + "\n")
        txt.see(tk.END)

    # NEW: save current output to the right pane (with timestamp) and keep structured snapshot
    def save_results_to_right():
        # use structured last_results instead of parsing text
        snapshot = last_results.get('value') or {}
        if not snapshot:
            messagebox.showinfo("Nothing to save", "Run a backtest first.")
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        saved_state['results'] = copy.deepcopy(snapshot)
        saved_state['ts'] = ts
        render_saved_snapshot(saved_state['results'], compare_to=None)
        # NEW: recolor current pane against the newly saved baseline
        render_current_snapshot(last_results.get('value') or {}, compare_to=saved_state['results'])

    # NEW: export current results to JSON for analysis (filtered, no CSV)
    def export_analysis():
        snapshot = last_results.get('value') or {}
        if not snapshot:
            messagebox.showinfo("No results", "Run a backtest first.")
            return
        base_dir = os.path.dirname(__file__)
        results_dir = os.path.join(base_dir, "results")
        try:
            os.makedirs(results_dir, exist_ok=True)
        except Exception as e:
            append_line(f"Failed to create results folder: {e}", "red")
            return

        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        json_path = os.path.join(results_dir, f"ultima_analysis_{ts}.json")

        # Filter out heavy fields from export
        def _filter_result_dict(d: dict) -> dict:
            if not isinstance(d, dict):
                return d
            # keys to drop inside each strategy result
            drop_keys = {"equity_times", "equity_values", "trades_list"}
            return {k: v for k, v in d.items() if k not in drop_keys}

        filtered = {}
        try:
            for symbol, strat_map in snapshot.items():
                filtered[symbol] = {}
                for strat, r in (strat_map or {}).items():
                    filtered[symbol][strat] = _filter_result_dict(r or {})
        except Exception as e:
            append_line(f"Failed to filter results: {e}", "red")
            return

        try:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(filtered, f, default=str, indent=2)
            append_line(f"Exported analysis JSON to results/{os.path.basename(json_path)}", "info")
        except Exception as e:
            append_line(f"Failed to write JSON: {e}", "red")

    # NEW: Save Best Config for live trading consumption
    def save_best_config():
        """
        Persist a bestconfig.json with per-symbol, per-strategy effective parameters and recent performance.
        Uses optimized combos if available; otherwise falls back to current GUI inputs.
        SCHEMA CHANGE (2025-10-08): Common fields (timeframe, timeframe_code, days, risk.lots,
        trading_window, htf_filter) are now stored ONCE per symbol instead of inside each
        strategy object to reduce duplication. For backward compatibility, loader will
        merge symbol-level fields into strategy views.
        """
        snapshot = last_results.get('value') or {}
        if not snapshot:
            # We can still save params without metrics, but prompt for clarity
            if not messagebox.askyesno("No results", "No backtest results found. Save configs from current inputs anyway?"):
                return

        base_dir = os.path.dirname(__file__)
        best_path = os.path.join(base_dir, "bestconfig.json")

        # Load existing config
        bestcfg = {}
        try:
            if os.path.exists(best_path):
                with open(best_path, 'r', encoding='utf-8') as f:
                    bestcfg = json.load(f) or {}
        except Exception:
            bestcfg = {}

        # Helpers to derive effective ATR per strategy (override or general)
        def effective_atr_for(strat_key: str):
            gen_ap = parse_int(atr_p.get(), ATR_PERIOD)
            gen_sl = parse_float(sl_x.get(), SL_ATR_MULTIPLIER)
            gen_tp = parse_float(tp_x.get(), TP_ATR_MULTIPLIER)
            pri = prio.get()
            m = {
                'ma_crossover': (mac_atr_p, mac_atr_sl, mac_atr_tp),
                'mean_reversion': (mr_atr_p, mr_atr_sl, mr_atr_tp),
                'momentum_trend': (mom_atr_p, mom_atr_sl, mom_atr_tp),
                'breakout': (brk_atr_p, brk_atr_sl, brk_atr_tp),
                'donchian_channel': (don_atr_p, don_atr_sl, don_atr_tp),
            }
            v_p, v_sl, v_tp = m.get(strat_key, (None, None, None))
            o_ap = parse_opt_int(v_p.get()) if v_p else None
            o_sl = parse_opt_float(v_sl.get()) if v_sl else None
            o_tp = parse_opt_float(v_tp.get()) if v_tp else None
            return {
                'period': int(o_ap or gen_ap),
                'sl_mult': float(o_sl if o_sl is not None else gen_sl),
                'tp_mult': float(o_tp if o_tp is not None else gen_tp),
                'priority': pri,
            }

        # Strategy param extractors (from current inputs)
        def strat_params_for(strat_key: str):
            if strat_key == 'ma_crossover':
                return {'fast': parse_int(mac_fast.get(), ma_crossover_fast), 'slow': parse_int(mac_slow.get(), ma_crossover_slow)}
            if strat_key == 'mean_reversion':
                return {'ma_period': parse_int(mr_ma.get(), meanrev_ma_period), 'num_std': parse_int(mr_std.get(), meanrev_num_std)}
            if strat_key == 'momentum_trend':
                return {'ma_period': parse_int(mom_ma.get(), momentum_ma_period), 'roc_period': parse_int(mom_roc.get(), momentum_roc_period)}
            if strat_key == 'breakout':
                return {'lookback': parse_int(brk_lookback.get(), breakout_lookback)}
            if strat_key == 'donchian_channel':
                return {
                    'channel_length': parse_int(brk_lookback.get(), breakout_lookback),
                    'exit_length': parse_opt_int(don_exit_len.get()) if str(don_exit_len.get()).strip() else None,
                    'confirm_bars': parse_opt_int(don_confirm.get()),
                    'atr_buffer_mult': parse_opt_float(don_atr_buf.get()),
                    'atr_period': parse_opt_int(don_atr_p.get()) or parse_int(atr_p.get(), ATR_PERIOD),
                }
            return {}

        # Pull optimized combos if available
        optimized_map = {}
        for strat, val in (opt_store.get('best') or {}).items():
            try:
                params, ap, slm, tpm, pri, score = val
                # Normalize params structure
                if strat == 'ma_crossover':
                    sp = {'fast': int(params[0]), 'slow': int(params[1])}
                elif strat == 'mean_reversion':
                    sp = {'ma_period': int(params[0]), 'num_std': int(params[1])}
                elif strat == 'momentum_trend':
                    sp = {'ma_period': int(params[0]), 'roc_period': int(params[1])}
                elif strat in ('breakout', 'donchian_channel'):
                    sp = {'lookback': int(params)} if strat == 'breakout' else {'channel_length': int(params)}
                else:
                    sp = {}
                optimized_map[strat] = {
                    'strategy_params': sp,
                    'atr': {'period': int(ap), 'sl_mult': float(slm), 'tp_mult': float(tpm), 'priority': str(pri)},
                    'objective': str(objective.get()),
                    'score': float(score)
                }
            except Exception:
                continue

        # Base/common metadata
        tf_input = tf_var.get()
        tf_code = int(parse_timeframe_input(tf_input))
        try:
            days_used = int(days_var.get())
        except Exception:
            days_used = 60

        trade_settings = {
            'trade_24_7': bool(trade_247.get()),
            'start_hour': parse_int(start_h.get(), TRADING_START_HOUR),
            'end_hour': parse_int(end_h.get(), TRADING_END_HOUR),
            'lots': parse_float(lots_var.get(), 0.1)
        }

        # Ensure symbols container
        if 'symbols' not in bestcfg or not isinstance(bestcfg.get('symbols'), dict):
            bestcfg['symbols'] = {}

        # Iterate over symbols in current results or inputs
        syms = list(snapshot.keys()) if snapshot else collect_symbols()
        now_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        for symbol in syms:
            # Ensure symbol container; embed symbol-level meta under special key '_meta'
            bestcfg['symbols'].setdefault(symbol, {})
            sym_node = bestcfg['symbols'][symbol]

            # Build/overwrite meta
            # HTF filter (shared) derivation
            try:
                htf_enabled = bool((load_last_params() or {}).get('htf_filter', {}).get('enabled', False))
            except Exception:
                htf_enabled = False
            htf_tf_input = htf_tf_var.get() if 'htf_tf_var' in locals() else 'H1'
            try:
                htf_code = int(parse_timeframe_input(htf_tf_input))
            except Exception:
                htf_code = int(parse_timeframe_input('H1'))
            try:
                htf_ma_p = parse_int(htf_ma_var.get(), 50)
            except Exception:
                htf_ma_p = 50
            sym_meta = {
                'timeframe': str(tf_input),
                'timeframe_code': tf_code,
                'days': int(days_used),
                'risk': {'lots': trade_settings['lots']},
                'trading_window': {
                    'trade_24_7': trade_settings['trade_24_7'],
                    'start_hour': trade_settings['start_hour'],
                    'end_hour': trade_settings['end_hour']
                },
                'htf_filter': {
                    'enabled': bool(use_htf.get()) if 'use_htf' in locals() else htf_enabled,
                    'timeframe': str(htf_tf_input),
                    'timeframe_code': int(htf_code),
                    'ma_period': int(htf_ma_p)
                }
            }
            sym_node['_meta'] = sym_meta

            # Determine strategies to save
            present_strats = list((snapshot.get(symbol) or {}).keys()) if snapshot else [k for k, v in strat_vars.items() if v.get()]
            for strat_key in present_strats:
                strat_entry = sym_node.get(strat_key, {})
                strat_entry['strategy_params'] = strat_params_for(strat_key)
                strat_entry['atr'] = effective_atr_for(strat_key)
                # Active flag
                try:
                    is_active = bool((strat_vars.get(strat_key).get())) if (isinstance(strat_vars, dict) and strat_key in strat_vars) else True
                except Exception:
                    is_active = True
                strat_entry['active'] = bool(is_active)
                # Performance & optimized intentionally omitted from persisted config per user request.
                # Remove if legacy keys already exist.
                strat_entry.pop('performance', None)
                strat_entry.pop('optimized', None)
                strat_entry['updated_at'] = now_ts
                # Remove duplicated symbol-level fields if exist from previous schema
                for dup_key in ['timeframe','timeframe_code','days','risk','trading_window','htf_filter']:
                    if dup_key in strat_entry:
                        strat_entry.pop(dup_key, None)
                sym_node[strat_key] = strat_entry

        # Top-level metadata (no more separate 'defaults' section per user request)
        bestcfg['updated_at'] = now_ts
        # Remove legacy defaults if present
        if 'defaults' in bestcfg:
            try:
                del bestcfg['defaults']
            except Exception:
                pass

        # Write out
        try:
            with open(best_path, 'w', encoding='utf-8') as f:
                json.dump(bestcfg, f, indent=2)
            append_line(f"Best config saved to {os.path.basename(best_path)}", "info")
        except Exception as e:
            append_line(f"Failed to save best config: {e}", "red")

    def clear_saved():
        txt_saved.delete("1.0", tk.END)
        saved_state['results'] = None
        saved_state['ts'] = None

    # ===== helpers =====
    def parse_int(val, default):
        try:
            return int(float(val))
        except Exception:
            return default

    def parse_float(val, default):
        try:
            return float(val)
        except Exception:
            return default

    # NEW: optional parse (empty -> None)
    def parse_opt_int(s):
        try:
            s = str(s).strip()
            return int(float(s)) if s else None
        except Exception:
            return None

    def parse_opt_float(s):
        try:
            s = str(s).strip()
            return float(s) if s else None
        except Exception:
            return None

    def collect_symbols():
        return [s.strip() for s in sym_var.get().split(",") if s.strip()]

    def apply_dates_and_tf():
        # update globals for data fetch according to Days and TF
        tf_const = parse_timeframe_input(tf_var.get())
        globals()['timeframe'] = tf_const
        try:
            days = int(days_var.get())
        except Exception:
            days = 60
        now = datetime.now()
        globals()['end_date'] = now
        globals()['start_date'] = now - timedelta(days=days)

    # NEW: collect current GUI state for saving
    def collect_gui_state():
        return {
            "symbols": sym_var.get(),
            "timeframe": tf_var.get(),
            "days": days_var.get(),
            "initial_balance": init_bal.get(),
            "trade_24_7": bool(trade_247.get()),
            "start_hour": start_h.get(),
            "end_hour": end_h.get(),
            "strategies": {k: bool(v.get()) for k, v in strat_vars.items()},
            "params": {
                "ma_crossover_fast": mac_fast.get(),
                "ma_crossover_slow": mac_slow.get(),
                "meanrev_ma_period": mr_ma.get(),
                "meanrev_num_std": mr_std.get(),
                "momentum_ma_period": mom_ma.get(),
                "momentum_roc_period": mom_roc.get(),
                "breakout_lookback": brk_lookback.get(),
                "don_exit_length": don_exit_len.get(),
                "don_confirm_bars": don_confirm.get(),
                "don_atr_buffer_mult": don_atr_buf.get(),
            },
            "atr": {
                "period": atr_p.get(),
                "sl_mult": sl_x.get(),
                "tp_mult": tp_x.get(),
                "priority": prio.get(),
                "lots": lots_var.get()
            },
            "htf_filter": {
                "enabled": bool(use_htf.get()),
                "timeframe": htf_tf_var.get(),
                "ma_period": htf_ma_var.get(),
            },
            "per_atr": {
                "ma_crossover": {"period": mac_atr_p.get(), "sl": mac_atr_sl.get(), "tp": mac_atr_tp.get()},
                "mean_reversion": {"period": mr_atr_p.get(),  "sl": mr_atr_sl.get(),  "tp": mr_atr_tp.get()},
                "momentum_trend": {"period": mom_atr_p.get(), "sl": mom_atr_sl.get(), "tp": mom_atr_tp.get()},
                "breakout":       {"period": brk_atr_p.get(),  "sl": brk_atr_sl.get(),  "tp": brk_atr_tp.get()},
                "donchian_channel": {"period": don_atr_p.get(), "sl": don_atr_sl.get(), "tp": don_atr_tp.get()},
            },
            "optimization": {
                "objective": objective.get(),
                "opt_atr": bool(opt_atr.get()),
                # NEW: persist constraints
                "min_trades": min_trades_var.get(),
                "min_profit": min_profit_var.get(),
                "min_winrate": min_winrate_var.get(),
                "min_expectancy": min_expectancy_var.get(),
                "min_sharpe": min_sharpe_var.get(),
            },
            "geometry": root.geometry()
        }

    # NEW: apply loaded state to GUI widgets
    def apply_gui_state(state: dict):
        try:
            sym_var.set(state.get("symbols", sym_var.get()))
            tf_var.set(state.get("timeframe", tf_var.get()))
            days_var.set(state.get("days", days_var.get()))
            init_bal.set(state.get("initial_balance", init_bal.get()))
            trade_247.set(state.get("trade_24_7", trade_247.get()))
            start_h.set(state.get("start_hour", start_h.get()))
            end_h.set(state.get("end_hour", end_h.get()))

            sdict = state.get("strategies", {})
            for k, var in strat_vars.items():
                if k in sdict:
                    var.set(bool(sdict[k]))

            pdict = state.get("params", {})
            mac_fast.set(pdict.get("ma_crossover_fast", mac_fast.get()))
            mac_slow.set(pdict.get("ma_crossover_slow", mac_slow.get()))
            mr_ma.set(pdict.get("meanrev_ma_period", mr_ma.get()))
            mr_std.set(pdict.get("meanrev_num_std", mr_std.get()))
            mom_ma.set(pdict.get("momentum_ma_period", mom_ma.get()))
            mom_roc.set(pdict.get("momentum_roc_period", mom_roc.get()))
            brk_lookback.set(pdict.get("breakout_lookback", brk_lookback.get()))
            if "don_exit_length" in pdict: don_exit_len.set(pdict["don_exit_length"])
            if "don_confirm_bars" in pdict: don_confirm.set(pdict["don_confirm_bars"])
            if "don_atr_buffer_mult" in pdict: don_atr_buf.set(pdict["don_atr_buffer_mult"])

            adict = state.get("atr", {})
            atr_p.set(adict.get("period", atr_p.get()))
            sl_x.set(adict.get("sl_mult", sl_x.get()))
            tp_x.set(adict.get("tp_mult", tp_x.get()))
            prio.set(adict.get("priority", prio.get()))
            lots_var.set(adict.get("lots", lots_var.get()))

            per = state.get("per_atr", {})
            def _set_per(d, k, v_p, v_sl, v_tp):
                if k in d:
                    v_p.set(d[k].get("period", v_p.get()))
                    v_sl.set(d[k].get("sl", v_sl.get()))
                    v_tp.set(d[k].get("tp", v_tp.get()))

            _set_per(per, "ma_crossover", mac_atr_p, mac_atr_sl, mac_atr_tp)
            _set_per(per, "mean_reversion", mr_atr_p, mr_atr_sl, mr_atr_tp)
            _set_per(per, "momentum_trend", mom_atr_p, mom_atr_sl, mom_atr_tp)
            _set_per(per, "breakout", brk_atr_p, brk_atr_sl, brk_atr_tp)
            _set_per(per, "donchian_channel", don_atr_p, don_atr_sl, don_atr_tp)

            odict = state.get("optimization", {})
            objective.set(odict.get("objective", objective.get()))
            opt_atr.set(odict.get("opt_atr", opt_atr.get()))
            # NEW: restore constraints
            if "min_trades" in odict: min_trades_var.set(odict.get("min_trades"))
            if "min_profit" in odict: min_profit_var.set(odict.get("min_profit"))
            if "min_winrate" in odict: min_winrate_var.set(odict.get("min_winrate"))
            if "min_expectancy" in odict: min_expectancy_var.set(odict.get("min_expectancy"))
            if "min_sharpe" in odict: min_sharpe_var.set(odict.get("min_sharpe"))

            # NEW: restore HTF filter
            htf = state.get("htf_filter", {})
            if isinstance(htf, dict):
                use_htf.set(htf.get("enabled", use_htf.get()))
                htf_tf_var.set(htf.get("timeframe", htf_tf_var.get()))
                htf_ma_var.set(htf.get("ma_period", htf_ma_var.get()))

            geom = state.get("geometry")
            if geom:
                try:
                    root.geometry(geom)
                except Exception:
                    pass
        except Exception:
            pass

    # Load and apply last params on startup
    apply_gui_state(load_last_params())

    # ===== actions =====
    def run_backtest():
        apply_dates_and_tf()

        sel_strats = [name for name, v in strat_vars.items() if v.get()]
        if not sel_strats:
            messagebox.showwarning("No strategies", "Please select at least one strategy.")
            return

        # Agent config
        agent = TradingAgent(initial_balance=parse_float(init_bal.get(), 100.0))
        agent._atr_period = parse_int(atr_p.get(), ATR_PERIOD)
        agent._sl_mult = parse_float(sl_x.get(), SL_ATR_MULTIPLIER)
        agent._tp_mult = parse_float(tp_x.get(), TP_ATR_MULTIPLIER)
        agent._intrabar_priority = prio.get()
        agent._trade_24_7 = bool(trade_247.get())
        agent._trade_start_hour = parse_int(start_h.get(), TRADING_START_HOUR)
        agent._trade_end_hour = parse_int(end_h.get(), TRADING_END_HOUR)
        agent._lots = parse_float(lots_var.get(), 0.1)

        results = {}
        syms = collect_symbols()
        txt.delete("1.0", tk.END)
        append_line("Backtest started...", "info")

        # NEW: start timer with progress callback
        timer['start'] = time.perf_counter()
        timer['running'] = True
        timer['get_progress'] = lambda: (pb['value'], pb['maximum'])
        lbl_timer.config(text="Elapsed: 00:00:00  |  Left: 00:00:00")
        root.after(0, _tick_timer)

        # NEW: setup progress bar for backtest
        total_steps = max(1, len(syms) * len(sel_strats))
        pb['maximum'] = total_steps
        pb['value'] = 0

        for symbol in syms:
            df = get_data(symbol)
            if df is None:
                append_line(f"No data for {symbol}", "red")
                # still step progress for each selected strategy to keep bar consistent
                for _ in sel_strats:
                    pb['value'] += 1
                root.update_idletasks()
                continue
            # Compute HTF bias if enabled
            bias_series = None
            if bool(use_htf.get()):
                try:
                    htf_code = int(parse_timeframe_input(htf_tf_var.get()))
                    ma_p = parse_int(htf_ma_var.get(), 50)
                    rates = mt5.copy_rates_range(symbol, htf_code, start_date, end_date)
                    if rates is not None and len(rates):
                        dfh = pd.DataFrame(rates)
                        dfh['time'] = pd.to_datetime(dfh['time'], unit='s')
                        dfh['ma'] = dfh['close'].rolling(ma_p).mean()
                        # Compute bias on completed HTF bars; then shift by 1 to avoid using current HTF bar info
                        dfh['bias'] = np.sign(dfh['close'] - dfh['ma']).shift(1)
                        # align to LTF bars
                        ltf_times = df['time'] if 'time' in df.columns else pd.Series(range(len(df)))
                        merged = pd.merge_asof(
                            pd.DataFrame({'time': ltf_times}),
                            dfh[['time', 'bias']].dropna(subset=['time']),
                            on='time',
                            direction='backward'
                        )
                        bias_series = merged['bias']
                    else:
                        bias_series = None
                except Exception:
                    bias_series = None
            sinfo = mt5.symbol_info(symbol)
            if sinfo is not None:
                agent._tick_size = getattr(sinfo, 'trade_tick_size', sinfo.point if hasattr(sinfo, 'point') else 0.01) or 0.01
                agent._tick_value = getattr(sinfo, 'trade_tick_value', 1.0) or 1.0
            sym_res = {}
            # General defaults
            gen_ap = parse_int(atr_p.get(), ATR_PERIOD)
            gen_sl = parse_float(sl_x.get(), SL_ATR_MULTIPLIER)
            gen_tp = parse_float(tp_x.get(), TP_ATR_MULTIPLIER)
            gen_prio = prio.get()

            # MA Crossover with optional overrides
            if 'ma_crossover' in sel_strats:
                o_ap = parse_opt_int(mac_atr_p.get()) or gen_ap
                o_sl = parse_opt_float(mac_atr_sl.get()) or gen_sl
                o_tp = parse_opt_float(mac_atr_tp.get()) or gen_tp
                sym_res['ma_crossover'] = agent.ma_crossover(
                    df.copy(),
                    parse_int(mac_fast.get(), ma_crossover_fast),
                    parse_int(mac_slow.get(), ma_crossover_slow),
                    atr_period=o_ap, sl_mult=o_sl, tp_mult=o_tp, priority=gen_prio, bias_series=bias_series
                )
                pb['value'] += 1; root.update_idletasks()

            # Mean Reversion
            if 'mean_reversion' in sel_strats:
                o_ap = parse_opt_int(mr_atr_p.get()) or gen_ap
                o_sl = parse_opt_float(mr_atr_sl.get()) or gen_sl
                o_tp = parse_opt_float(mr_atr_tp.get()) or gen_tp
                sym_res['mean_reversion'] = agent.mean_reversion(
                    df.copy(),
                    parse_int(mr_ma.get(), meanrev_ma_period),
                    parse_int(mr_std.get(), meanrev_num_std),
                    atr_period=o_ap, sl_mult=o_sl, tp_mult=o_tp, priority=gen_prio, bias_series=bias_series
                )
                pb['value'] += 1; root.update_idletasks()

            # Momentum Trend
            if 'momentum_trend' in sel_strats:
                o_ap = parse_opt_int(mom_atr_p.get()) or gen_ap
                o_sl = parse_opt_float(mom_atr_sl.get()) or gen_sl
                o_tp = parse_opt_float(mom_atr_tp.get()) or gen_tp
                sym_res['momentum_trend'] = agent.momentum_trend(
                    df.copy(),
                    parse_int(mom_ma.get(), momentum_ma_period),
                    parse_int(mom_roc.get(), momentum_roc_period),
                    atr_period=o_ap, sl_mult=o_sl, tp_mult=o_tp, priority=gen_prio, bias_series=bias_series
                )
                pb['value'] += 1; root.update_idletasks()

            # Breakout
            if 'breakout' in sel_strats:
                o_ap = parse_opt_int(brk_atr_p.get()) or gen_ap
                o_sl = parse_opt_float(brk_atr_sl.get()) or gen_sl
                o_tp = parse_opt_float(brk_atr_tp.get()) or gen_tp
                sym_res['breakout'] = agent.breakout(
                    df.copy(),
                    parse_int(brk_lookback.get(), breakout_lookback),
                    atr_period=o_ap, sl_mult=o_sl, tp_mult=o_tp, priority=gen_prio, bias_series=bias_series
                )
                pb['value'] += 1; root.update_idletasks()

            # Donchian
            if 'donchian_channel' in sel_strats:
                o_ap = parse_opt_int(don_atr_p.get()) or gen_ap
                o_sl = parse_opt_float(don_atr_sl.get()) or gen_sl
                o_tp = parse_opt_float(don_atr_tp.get()) or gen_tp
                sym_res['donchian_channel'] = agent.donchian_channel(
                    df.copy(),
                    parse_int(brk_lookback.get(), breakout_lookback),
                    exit_length=parse_opt_int(don_exit_len.get()),
                    confirm_bars=parse_opt_int(don_confirm.get()) or 2,
                    atr_buffer_mult=parse_opt_float(don_atr_buf.get()) or 0.5,
                    atr_period=o_ap, sl_mult=o_sl, tp_mult=o_tp, priority=gen_prio, bias_series=bias_series
                )
                pb['value'] += 1; root.update_idletasks()

            results[symbol] = sym_res

            # REMOVED: per-line single color based on total_profit
            # for strat_name, r in sym_res.items():
            #     line = f"{symbol} {strat_name}: Trades={r['trades']}, WinRate={r['win_rate']:.2f}%, Profit={r['total_profit']:.2f}, Final={r['final_balance']:.2f}, MaxDD={r['max_drawdown']:.2f}"
            #     tag = "green" if r['total_profit'] >= 0 else "red"
            #     append_line(line, tag)
        last_results['value'] = results
        # NEW: render current snapshot with metric-level coloring (vs saved baseline if exists)
        render_current_snapshot(last_results['value'], compare_to=saved_state['results'])
        append_line("Backtest completed.", "info")
        # NEW: if there is a saved snapshot, re-render it with red/green vs current
        if saved_state['results'] is not None:
            render_saved_snapshot(saved_state['results'], compare_to=last_results['value'])
        # CHANGED: reset progress bar so it doesn't stay green/full
        pb['value'] = 0
        root.update_idletasks()
        # persist on run
        save_last_params(collect_gui_state())

    def plot_charts():
        results = last_results['value']
        if not results:
            messagebox.showinfo("No results", "Run a backtest first.")
            return
        for symbol, res in results.items():
            plt.figure(figsize=(10, 6))
            plotted_any = False
            for strat_name, r in res.items():
                times = r.get('equity_times')
                values = r.get('equity_values')
                if times is not None and values is not None and len(times) and len(times) == len(values):
                    plt.step(times, values, where='post', label=strat_name)
                    plotted_any = True
            if plotted_any:
                plt.title(f"Equity Curves - {symbol}")
                plt.xlabel("Time")
                plt.ylabel("Balance")
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.close()
        plt.show()

    def run_optimization():
        apply_dates_and_tf()

        sel_strats = [name for name, v in strat_vars.items() if v.get()]
        if not sel_strats:
            messagebox.showwarning("No strategies", "Please select at least one strategy for optimization.")
            return

        # Grids
        ma_crossover_grid = [(f, s) for f, s in product([5, 10, 20, 30, 50, 80, 100], [20, 50, 100, 200, 300]) if f < s]
        meanrev_grid = [(p, n) for p in [10, 20, 30, 50, 80, 100] for n in [1, 2, 3, 4, 5, 7, 10]]
        momentum_grid = [(p, r) for p in [20, 50, 100, 200, 300] for r in [5, 10, 20, 30, 50]]
        breakout_grid = [l for l in [10, 20, 50, 100, 200]]
        donchian_grid = [l for l in [10, 20, 50, 100, 200]]
        atr_period_grid = [7, 10, 14, 21, 30, 50]
        sl_mult_grid = [1.0, 1.5, 2.0, 2.5, 3.0]
        tp_mult_grid = [1.0, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
        priority_grid = ['SL']

        objective_key = objective.get()
        tune_atr = bool(opt_atr.get())

        def score(res_list):
            if objective_key == "highest_total_profit":
                return sum(r['total_profit'] for r in res_list)
            if objective_key == "highest_final_balance":
                return sum(r['final_balance'] for r in res_list)
            if objective_key == "lowest_max_drawdown":
                return -max((r['max_drawdown'] for r in res_list), default=0.0)
            if objective_key == "highest_win_rate":
                total_trades = sum(r['trades'] for r in res_list)
                if not total_trades:
                    return -np.inf
                wr = sum(r['trades'] * r['win_rate'] for r in res_list) / total_trades
                return wr
            if objective_key == "highest_expectancy":
                total_trades = sum(r.get('trades', 0) for r in res_list)
                total_profit = sum(r.get('total_profit', 0.0) for r in res_list)
                if total_trades <= 0:
                    return -np.inf
                return total_profit / total_trades
            if objective_key == "highest_trade_sharpe":
                sharpe_vals = [r.get('trade_sharpe', 0.0) for r in res_list if r.get('trades', 0) > 0]
                if not sharpe_vals:
                    return -np.inf
                return float(np.mean(sharpe_vals))
            return sum(r['total_profit'] for r in res_list)

        append_line(f"Optimization started (objective={objective_key}, tune_atr={tune_atr})...", "info")

        # NEW: start timer with progress callback
        timer['start'] = time.perf_counter()
        timer['running'] = True
        timer['get_progress'] = lambda: (pb['value'], pb['maximum'])
        lbl_timer.config(text="Elapsed: 00:00:00  |  Left: 00:00:00")
        root.after(0, _tick_timer)

        # Enable stop button
        opt_state['stop'] = False
        btn_stop_opt['state'] = 'normal'
        stopped = False

        # Base config
        base_init_bal = parse_float(init_bal.get(), 100.0)
        base_atr = parse_int(atr_p.get(), ATR_PERIOD)
        base_sl = parse_float(sl_x.get(), SL_ATR_MULTIPLIER)
        base_tp = parse_float(tp_x.get(), TP_ATR_MULTIPLIER)
        base_prio = prio.get()
        base_247 = bool(trade_247.get())
        base_start = parse_int(start_h.get(), TRADING_START_HOUR)
        base_end = parse_int(end_h.get(), TRADING_END_HOUR)
        base_lots = parse_float(lots_var.get(), 0.1)
        syms = collect_symbols()

        # NEW: setup progress bar for optimization (use actual grids)
        atr_count = (len(atr_period_grid) * len(sl_mult_grid) * len(tp_mult_grid) * len(priority_grid)) if tune_atr else 1
        grid_map = {
            'ma_crossover': ma_crossover_grid,
            'mean_reversion': meanrev_grid,
            'momentum_trend': momentum_grid,
            'breakout': breakout_grid,
            'donchian_channel': donchian_grid,
        }
        total_steps_opt = sum(len(grid_map[s]) * atr_count * len(syms) for s in sel_strats if s in grid_map)

        pb['maximum'] = max(1, total_steps_opt)
        pb['value'] = 0
        root.update_idletasks()

        # NEW: parse optimization constraints
        min_trades_th = parse_int(min_trades_var.get(), 1)
        min_profit_th = parse_float(min_profit_var.get(), 0.0)
        min_winrate_th = parse_float(min_winrate_var.get(), 0.0)
        min_expect_th = parse_float(min_expectancy_var.get(), 0.0)
        min_sharpe_th = parse_float(min_sharpe_var.get(), 0.0)

        for strategy in sel_strats:
            if stopped:
                break

            if strategy == 'ma_crossover':
                strat_grid = ma_crossover_grid
            elif strategy == 'mean_reversion':
                strat_grid = meanrev_grid
            elif strategy == 'momentum_trend':
                strat_grid = momentum_grid
            elif strategy == 'breakout':
                strat_grid = breakout_grid
            elif strategy == 'donchian_channel':
                strat_grid = donchian_grid
            else:
                continue

            best_score = -np.inf
            best_combo = None

            atr_grid_iter_all = [(base_atr, base_sl, base_tp, base_prio)]
            if tune_atr:
                atr_grid_iter_all = list(product(atr_period_grid, sl_mult_grid, tp_mult_grid, priority_grid))

            for strat_params in strat_grid:
                if stopped or opt_state['stop']:
                    stopped = True
                    break
                for ap, slm, tpm, prio_sel in atr_grid_iter_all:
                    root.update()
                    if opt_state['stop']:
                        stopped = True
                        break
                    sym_results = []
                    for symbol in syms:
                        if opt_state['stop']:
                            stopped = True
                            break
                        try:
                            agent = TradingAgent(initial_balance=base_init_bal)
                            agent._atr_period = ap
                            agent._sl_mult = slm
                            agent._tp_mult = tpm
                            agent._intrabar_priority = prio_sel
                            agent._trade_24_7 = base_247
                            agent._trade_start_hour = base_start
                            agent._trade_end_hour = base_end
                            agent._lots = base_lots

                            df = get_data(symbol)
                            if df is None:
                                continue
                            # Compute HTF bias if enabled
                            bias_series = None
                            if bool(use_htf.get()):
                                try:
                                    htf_code = int(parse_timeframe_input(htf_tf_var.get()))
                                    ma_p = base_atr  # use existing var? better use parse_int(htf_ma_var.get(), 50) but not available here; replicate
                                    try:
                                        ma_p = int(float(htf_ma_var.get()))
                                    except Exception:
                                        ma_p = 50
                                    rates = mt5.copy_rates_range(symbol, htf_code, start_date, end_date)
                                    if rates is not None and len(rates):
                                        dfh = pd.DataFrame(rates)
                                        dfh['time'] = pd.to_datetime(dfh['time'], unit='s')
                                        dfh['ma'] = dfh['close'].rolling(ma_p).mean()
                                        # Use previous completed HTF bar bias only
                                        dfh['bias'] = np.sign(dfh['close'] - dfh['ma']).shift(1)
                                        ltf_times = df['time'] if 'time' in df.columns else pd.Series(range(len(df)))
                                        merged = pd.merge_asof(
                                            pd.DataFrame({'time': ltf_times}),
                                            dfh[['time', 'bias']].dropna(subset=['time']),
                                            on='time',
                                            direction='backward'
                                        )
                                        bias_series = merged['bias']
                                except Exception:
                                    bias_series = None
                            sinfo = mt5.symbol_info(symbol)
                            if sinfo is not None:
                                agent._tick_size = getattr(sinfo, 'trade_tick_size', sinfo.point if hasattr(sinfo, 'point') else 0.01) or 0.01
                                agent._tick_value = getattr(sinfo, 'trade_tick_value', 1.0) or 1.0

                            # run strategy
                            if strategy == 'ma_crossover':
                                res = agent.ma_crossover(df.copy(), strat_params[0], strat_params[1], bias_series=bias_series)
                            elif strategy == 'mean_reversion':
                                res = agent.mean_reversion(df.copy(), strat_params[0], strat_params[1], bias_series=bias_series)
                            elif strategy == 'momentum_trend':
                                res = agent.momentum_trend(df.copy(), strat_params[0], strat_params[1], bias_series=bias_series)
                            elif strategy == 'breakout':
                                res = agent.breakout(df.copy(), strat_params, bias_series=bias_series)
                            elif strategy == 'donchian_channel':
                                res = agent.donchian_channel(df.copy(), strat_params, bias_series=bias_series)
                            else:
                                res = None
                            if res is not None:
                                sym_results.append(res)
                        finally:
                            # NEW: progress tick per symbol attempt (even if no data)
                            pb['value'] += 1
                            root.update_idletasks()
                    if stopped:
                        break

                    # NEW: enforce constraints to reject invalid combos
                    total_trades = sum(r.get('trades', 0) for r in sym_results)
                    total_profit = sum(r.get('total_profit', 0.0) for r in sym_results)
                    weighted_wr = (sum(r.get('trades', 0) * r.get('win_rate', 0.0) for r in sym_results) / total_trades) if total_trades > 0 else 0.0
                    expectancy_total = (total_profit / total_trades) if total_trades > 0 else 0.0
                    sharpe_vals = [r.get('trade_sharpe', 0.0) for r in sym_results if r.get('trades', 0) > 0]
                    avg_sharpe = float(np.mean(sharpe_vals)) if len(sharpe_vals) else 0.0
                    if (
                        (total_trades < min_trades_th) or
                        (total_profit < min_profit_th) or
                        (weighted_wr < min_winrate_th) or
                        (expectancy_total < min_expect_th) or
                        (avg_sharpe < min_sharpe_th)
                    ):
                        s = -np.inf
                    else:
                        s = score(sym_results)

                    if s > best_score:
                        best_score = s
                        best_combo = (strat_params, ap, slm, tpm, prio_sel)

            if best_combo is not None:
                opt_store['best'][strategy] = (best_combo[0], best_combo[1], best_combo[2], best_combo[3], best_combo[4], best_score)
                append_line(f"Best for {strategy}: params={best_combo[0]}, ATR_PERIOD={best_combo[1]}, SLx={best_combo[2]}, TPx={best_combo[3]}, priority={best_combo[4]}, objective_score={best_score:.2f}", "info")
            elif stopped:
                append_line(f"Optimization stopped before completing {strategy}.", "info")

        # Disable stop button
        btn_stop_opt['state'] = 'disabled'
        if stopped:
            append_line("Optimization stopped by user.", "info")
        else:
            append_line("Optimization completed.", "info")

        # NEW: stop/finalize timer (freeze final times)
        if timer['start'] is not None:
            elapsed = time.perf_counter() - timer['start']
            timer['running'] = False
            lbl_timer.config(text=f"Elapsed: {_fmt_hms(elapsed)}  |  Left: 00:00:00")

        # reset progress bar after optimization
        pb['value'] = 0
        root.update_idletasks()

        # persist options/state
        save_last_params(collect_gui_state())

    def apply_optimized_params():
        best_map = opt_store['best']
        if not best_map:
            messagebox.showinfo("No optimized results", "Run optimization first.")
            return

        # per-strategy fields
        for strat, (params, ap, slm, tpm, prio_sel, score) in best_map.items():
            if strat == 'ma_crossover':
                try:
                    f, s = params
                    mac_fast.set(str(int(f)))
                    mac_slow.set(str(int(s)))
                    # NEW: apply ATR overrides for MA Crossover
                    mac_atr_p.set(str(int(ap)))
                    mac_atr_sl.set(str(float(slm)))
                    mac_atr_tp.set(str(float(tpm)))
                except Exception:
                    pass
            elif strat == 'mean_reversion':
                try:
                    p, n = params
                    mr_ma.set(str(int(p)))
                    mr_std.set(str(int(n)))
                    # NEW: apply ATR overrides for Mean Reversion
                    mr_atr_p.set(str(int(ap)))
                    mr_atr_sl.set(str(float(slm)))
                    mr_atr_tp.set(str(float(tpm)))
                except Exception:
                    pass
            elif strat == 'momentum_trend':
                try:
                    p, r = params
                    mom_ma.set(str(int(p)))
                    mom_roc.set(str(int(r)))
                    # NEW: apply ATR overrides for Momentum/Trend
                    mom_atr_p.set(str(int(ap)))
                    mom_atr_sl.set(str(float(slm)))
                    mom_atr_tp.set(str(float(tpm)))
                except Exception:
                    pass
            elif strat == 'breakout':
                try:
                    brk_lookback.set(str(int(params)))
                    # NEW: apply ATR overrides for Breakout
                    brk_atr_p.set(str(int(ap)))
                    brk_atr_sl.set(str(float(slm)))
                    brk_atr_tp.set(str(float(tpm)))
                except Exception:
                    pass
            elif strat == 'donchian_channel':
                try:
                    brk_lookback.set(str(int(params)))
                    # NEW: apply ATR overrides for Donchian Channel
                    don_atr_p.set(str(int(ap)))
                    don_atr_sl.set(str(float(slm)))
                    don_atr_tp.set(str(float(tpm)))
                except Exception:
                    pass

        # REMOVED: setting global ATR fields from overall best
        # (ATR now applied to per-strategy overrides)

        messagebox.showinfo("Applied", "Optimized parameters applied to inputs.")
        # NEW: persist after applying best
        save_last_params(collect_gui_state())

    # NEW: save on window close
    def on_close():
        try:
            save_last_params(collect_gui_state())
        finally:
            root.destroy()

    # Buttons
    ttk.Button(frm_actions, text="Run Backtest", command=run_backtest).grid(row=0, column=0, padx=6)
    ttk.Button(frm_actions, text="Show Charts", command=plot_charts).grid(row=0, column=1, padx=6)
    ttk.Button(frm_actions, text="Run Optimization", command=run_optimization).grid(row=0, column=2, padx=6)
    ttk.Button(frm_actions, text="Apply Best to Params", command=apply_optimized_params).grid(row=0, column=3, padx=6)
    # NEW: Save/clear results to right pane
    ttk.Button(frm_actions, text="Save Results (->)", command=save_results_to_right).grid(row=0, column=4, padx=6)
    ttk.Button(frm_actions, text="Clear Saved", command=clear_saved).grid(row=0, column=5, padx=6)
    ttk.Button(frm_actions, text="Export Analysis", command=export_analysis).grid(row=0, column=6, padx=6)
    # NEW: Save Best Config for live
    ttk.Button(frm_actions, text="Save Best Config", command=save_best_config).grid(row=0, column=7, padx=6)
    # NEW: Load Config (single symbol)
    ttk.Button(frm_actions, text="Load Config", command=lambda: load_symbol_config()).grid(row=0, column=8, padx=6)
    # Move Stop and Quit to the right
    btn_stop_opt = ttk.Button(frm_actions, text="Stop Optimization", command=lambda: opt_state.update(stop=True), state='disabled')
    btn_stop_opt.grid(row=0, column=9, padx=6)
    # CHANGED: route Quit through on_close so it saves state consistently
    ttk.Button(frm_actions, text="Quit", command=on_close).grid(row=0, column=10, padx=6)

    # Progress bar spans all action columns
    pb = ttk.Progressbar(frm_actions, orient="horizontal", mode="determinate", length=600)
    pb.grid(row=1, column=0, columnspan=10, sticky="ew", pady=(10, 0))

    # NEW: elapsed time + ETA label
    lbl_timer = ttk.Label(frm_actions, text="Elapsed: 00:00:00  |  Left: 00:00:00")
    lbl_timer.grid(row=2, column=0, columnspan=10, sticky="e", pady=(6, 0))

    # NEW: timer state with progress callback
    timer = {'running': False, 'start': None, 'get_progress': None}

    def _fmt_hms(seconds: float) -> str:
        s = int(max(0, seconds))
        h = s // 3600
        m = (s % 3600) // 60
        sec = s % 60
        return f"{h:02d}:{m:02d}:{sec:02d}"

    def _tick_timer():
        if timer['running'] and timer['start'] is not None:
            elapsed = time.perf_counter() - timer['start']
            # Estimate remaining from progress
            left_str = "00:00:00"
            if callable(timer.get('get_progress')):
                try:
                    done, total = timer['get_progress']()
                    done = float(done or 0.0)
                    total = float(total or 0.0)
                    if total > 0 and 0 < done <= total:
                        frac = done / total
                        rem = elapsed * (1.0 / max(frac, 1e-9) - 1.0)
                        left_str = _fmt_hms(rem)
                except Exception:
                    pass
            lbl_timer.config(text=f"Elapsed: {_fmt_hms(elapsed)}  |  Left: {left_str}")
            # schedule next update
            root.after(250, _tick_timer)

    # =============================
    # Load symbol config from bestconfig.json (single symbol only)
    # =============================
    def load_symbol_config():
        base_dir = os.path.dirname(__file__)
        best_path = os.path.join(base_dir, "bestconfig.json")
        if not os.path.exists(best_path):
            messagebox.showwarning("Missing", "bestconfig.json not found. Save one first.")
            return
        try:
            with open(best_path, 'r', encoding='utf-8') as f:
                cfg = json.load(f) or {}
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read bestconfig.json: {e}")
            return
        # Determine symbols in GUI entry
        symbols_input = collect_symbols()
        if len(symbols_input) != 1:
            messagebox.showinfo("Info", "Load Config only works when exactly one symbol is specified.")
            return
        symbol = symbols_input[0]
        sym_node = (cfg.get('symbols') or {}).get(symbol)
        if not sym_node:
            messagebox.showwarning("Not Found", f"Symbol '{symbol}' not in bestconfig.json")
            return
        # Support new schema meta
        meta = sym_node.get('_meta', {}) if isinstance(sym_node, dict) else {}
        # Apply timeframe/days
        try:
            tf_val = meta.get('timeframe') or cfg.get('defaults', {}).get('timeframe')
            if tf_val:
                tf_var.set(str(tf_val))
        except Exception:
            pass
        try:
            days_val = meta.get('days') or cfg.get('defaults', {}).get('days')
            if days_val:
                days_var.set(str(days_val))
        except Exception:
            pass
        # Risk lots
        try:
            lots_val = (meta.get('risk') or {}).get('lots')
            if lots_val is not None:
                lots_var.set(str(lots_val))
        except Exception:
            pass
        # Trading window
        tw = meta.get('trading_window') or {}
        try:
            trade_247.set(bool(tw.get('trade_24_7', True)))
            start_h.set(str(tw.get('start_hour', TRADING_START_HOUR)))
            end_h.set(str(tw.get('end_hour', TRADING_END_HOUR)))
        except Exception:
            pass
        # HTF Filter
        htf = meta.get('htf_filter') or {}
        try:
            use_htf.set(bool(htf.get('enabled', False)))
            if htf.get('timeframe'):
                htf_tf_var.set(str(htf.get('timeframe')))
            if htf.get('ma_period') is not None:
                htf_ma_var.set(str(htf.get('ma_period')))
        except Exception:
            pass
        # Strategy activation + params + per-strategy ATR overrides
        # Clear all strategies to False first then set those present as active flag
        try:
            for k, var in strat_vars.items():
                var.set(False)
        except Exception:
            pass
        for strat_key, strat_data in sym_node.items():
            if strat_key == '_meta':
                continue
            try:
                # Active flag
                if strat_key in strat_vars:
                    strat_vars[strat_key].set(bool(strat_data.get('active', True)))
                sp = strat_data.get('strategy_params', {}) or {}
                if strat_key == 'ma_crossover':
                    if 'fast' in sp: mac_fast.set(str(sp['fast']))
                    if 'slow' in sp: mac_slow.set(str(sp['slow']))
                elif strat_key == 'mean_reversion':
                    if 'ma_period' in sp: mr_ma.set(str(sp['ma_period']))
                    if 'num_std' in sp: mr_std.set(str(sp['num_std']))
                elif strat_key == 'momentum_trend':
                    if 'ma_period' in sp: mom_ma.set(str(sp['ma_period']))
                    if 'roc_period' in sp: mom_roc.set(str(sp['roc_period']))
                elif strat_key == 'breakout':
                    if 'lookback' in sp: brk_lookback.set(str(sp['lookback']))
                elif strat_key == 'donchian_channel':
                    if 'channel_length' in sp: brk_lookback.set(str(sp['channel_length']))
                # ATR overrides (period/sl/tp) if they differ from general defaults store them in per-strategy fields
                atr = strat_data.get('atr') or {}
                def _assign_per(v_p, v_sl, v_tp, a):
                    if a.get('period') is not None: v_p.set(str(a.get('period')))
                    if a.get('sl_mult') is not None: v_sl.set(str(a.get('sl_mult')))
                    if a.get('tp_mult') is not None: v_tp.set(str(a.get('tp_mult')))
                if strat_key == 'ma_crossover':
                    _assign_per(mac_atr_p, mac_atr_sl, mac_atr_tp, atr)
                elif strat_key == 'mean_reversion':
                    _assign_per(mr_atr_p, mr_atr_sl, mr_atr_tp, atr)
                elif strat_key == 'momentum_trend':
                    _assign_per(mom_atr_p, mom_atr_sl, mom_atr_tp, atr)
                elif strat_key == 'breakout':
                    _assign_per(brk_atr_p, brk_atr_sl, brk_atr_tp, atr)
                elif strat_key == 'donchian_channel':
                    _assign_per(don_atr_p, don_atr_sl, don_atr_tp, atr)
            except Exception:
                continue
        # Persist GUI state after loading
        save_last_params(collect_gui_state())
        append_line(f"Loaded config for {symbol}", 'info')

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

# If GUI is enabled, run it and exit before the legacy batch/grids block executes
if USE_GUI:
    if not mt5.initialize():
        print("initialize() failed")
    else:
        gui_main()
        mt5.shutdown()
    raise SystemExit

# ===== Legacy non-GUI batch flow below (unchanged) =====

results = {}
agent = TradingAgent(initial_balance=100)
# Inject ATR/exit params into agent (a simple way to pass defaults)
agent._atr_period = ATR_PERIOD
agent._sl_mult = SL_ATR_MULTIPLIER
agent._tp_mult = TP_ATR_MULTIPLIER
agent._intrabar_priority = INTRABAR_PRIORITY
# NEW: apply trading hours to agent
agent._trade_24_7 = TRADE_24_7
agent._trade_start_hour = TRADING_START_HOUR
agent._trade_end_hour = TRADING_END_HOUR

for symbol in symbols:
    df = get_data(symbol)
    # Pull tick specs from MT5 for realistic PnL scaling
    sinfo = mt5.symbol_info(symbol)
    if sinfo is not None:
        agent._tick_size = getattr(sinfo, 'trade_tick_size', sinfo.point if hasattr(sinfo, 'point') else 0.01) or 0.01
        agent._tick_value = getattr(sinfo, 'trade_tick_value', 1.0) or 1.0
        # Simple default lots: 1 for FX; allow custom logic per symbol as needed
        agent._lots = 0.1
    min_lookback = max(ma_crossover_slow, meanrev_ma_period, momentum_ma_period, breakout_lookback)
    # Optional: compute a default HTF bias in legacy path (H1/50)
    bias_series = None
    try:
        htf_code = mt5.TIMEFRAME_H1
        rates = mt5.copy_rates_range(symbol, htf_code, start_date, end_date)
        if rates is not None and len(rates):
            dfh = pd.DataFrame(rates)
            dfh['time'] = pd.to_datetime(dfh['time'], unit='s')
            dfh['ma'] = dfh['close'].rolling(50).mean()
            # Bias from previous completed HTF bar to avoid lookahead
            dfh['bias'] = np.sign(dfh['close'] - dfh['ma']).shift(1)
            ltf_times = df['time'] if 'time' in df.columns else pd.Series(range(len(df)))
            merged = pd.merge_asof(
                pd.DataFrame({'time': ltf_times}),
                dfh[['time', 'bias']].dropna(subset=['time']),
                on='time',
                direction='backward'
            )
            bias_series = merged['bias']
    except Exception:
        bias_series = None
    if df is not None and len(df) > min_lookback:
        res_ma = agent.ma_crossover(df.copy(), ma_crossover_fast, ma_crossover_slow, bias_series=bias_series)
        res_mr = agent.mean_reversion(df.copy(), meanrev_ma_period, meanrev_num_std, bias_series=bias_series)
        res_mom = agent.momentum_trend(df.copy(), momentum_ma_period, momentum_roc_period, bias_series=bias_series)
        res_brk = agent.breakout(df.copy(), breakout_lookback, bias_series=bias_series)
        res_don = agent.donchian_channel(df.copy(), breakout_lookback, bias_series=bias_series)
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
    else:
        print(f"Not enough data for {symbol}")

# NEW: simple equity plotting helper
def plot_equity_curves(results):
    for symbol, res in results.items():
        plt.figure(figsize=(10, 6))
        plotted_any = False
        for strat_name, r in res.items():
            times = r.get('equity_times')
            values = r.get('equity_values')
            if times is not None and values is not None and len(times) and len(times) == len(values):
                plt.step(times, values, where='post', label=strat_name)
                plotted_any = True
        if plotted_any:
            plt.title(f"Equity Curves - {symbol}")
            plt.xlabel("Time")
            plt.ylabel("Balance")
            plt.legend()
            plt.grid(True, alpha=0.3)
        else:
            plt.close()  # nothing to plot for this symbol

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

# NEW: draw charts for the initial run (not during grid search)
plot_equity_curves(results)
plt.show()

# print_summary(results, 'ma_crossover')
# print_summary(results, 'mean_reversion')
# print_summary(results, 'momentum_trend')
# print_summary(results, 'breakout')

# Parameter grids
ma_crossover_grid = [(f, s) for f, s in product([5, 10, 20], [20, 50, 100]) if f < s]
meanrev_grid = [(p, n) for p in [10, 20, 30] for n in [1, 2, 3]]
momentum_grid = [(p, r) for p in [20, 50, 100] for r in [5, 10, 20]]
breakout_grid = [l for l in [10, 20, 50]]
donchian_grid = [l for l in [10, 20, 50]]

# ATR parameter grids for fine tuning
atr_period_grid = [7, 14, 21]
sl_mult_grid = [1.5, 2.0, 2.5, 3.0]
tp_mult_grid = [2.0, 2.5, 3.0, 4.0]
priority_grid = ['SL']

best_params = {}
for strategy, grid in [
    ('ma_crossover', ma_crossover_grid),
    ('mean_reversion', meanrev_grid),
    ('momentum_trend', momentum_grid),
    ('breakout', breakout_grid),
    ('donchian_channel', donchian_grid)
]:
    best_profit = -np.inf
    best_param = None
    for params in grid:
        total_profit = 0
        for symbol in symbols:
            df = get_data(symbol)
            if df is None:
                continue
            if strategy == 'ma_crossover':
                res = agent.ma_crossover(df.copy(), params[0], params[1])
            elif strategy == 'mean_reversion':
                res = agent.mean_reversion(df.copy(), params[0], params[1])
            elif strategy == 'momentum_trend':
                res = agent.momentum_trend(df.copy(), params[0], params[1])
            elif strategy == 'breakout':
                res = agent.breakout(df.copy(), params)
            elif strategy == 'donchian_channel':
                res = agent.donchian_channel(df.copy(), params)
            else:
                continue
            total_profit += res['total_profit']
        if total_profit > best_profit:
            best_profit = total_profit
            best_param = params
    best_params[strategy] = (best_param, best_profit)

print("\nBest parameters for each strategy (by total profit):")
for strat, (params, profit) in best_params.items():
    print(f"{strat}: params={params}, total_profit={profit:.2f}")

# Joint fine-tuning over strategy params plus ATR exit params
print("\nFine-tuning ATR exit parameters jointly with strategies (by total profit):")
best_combo = {}
for strategy, grid in [
    ('ma_crossover', ma_crossover_grid),
    ('mean_reversion', meanrev_grid),
    ('momentum_trend', momentum_grid),
    ('breakout', breakout_grid),
    ('donchian_channel', donchian_grid)
]:
    best_profit = -np.inf
    best_params_combo = None
    for strat_params in grid:
        for ap in atr_period_grid:
            for slm in sl_mult_grid:
                for tpm in tp_mult_grid:
                    for prio in priority_grid:
                        # configure agent
                        agent._atr_period = ap
                        agent._sl_mult = slm
                        agent._tp_mult = tpm
                        agent._intrabar_priority = prio

                        total_profit = 0
                        for symbol in symbols:
                            df = get_data(symbol)
                            if df is None:
                                continue
                            if strategy == 'ma_crossover':
                                res = agent.ma_crossover(df.copy(), strat_params[0], strat_params[1])
                            elif strategy == 'mean_reversion':
                                res = agent.mean_reversion(df.copy(), strat_params[0], strat_params[1])
                            elif strategy == 'momentum_trend':
                                res = agent.momentum_trend(df.copy(), strat_params[0], strat_params[1])
                            elif strategy == 'breakout':
                                res = agent.breakout(df.copy(), strat_params)
                            elif strategy == 'donchian_channel':
                                res = agent.donchian_channel(df.copy(), strat_params)
                            else:
                                continue
                            total_profit += res['total_profit']

                        if total_profit > best_profit:
                            best_profit = total_profit
                            best_params_combo = (strat_params, ap, slm, tpm, prio)

    best_combo[strategy] = (best_params_combo, best_profit)

print("\nBest combined strategy + ATR params (by total profit):")
for strat, (combo, profit) in best_combo.items():
    print(f"{strat}: strategy_params={combo[0]}, ATR_PERIOD={combo[1]}, SLx={combo[2]}, TPx={combo[3]}, priority={combo[4]}, total_profit={profit:.2f}")

mt5.shutdown()