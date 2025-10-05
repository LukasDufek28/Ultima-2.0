import MetaTrader5 as mt5
import pandas as pd
from datetime import datetime, timedelta, timezone
import pytz
from agents import MeanReversionAgent, MACrossoverAgent, MomentumTrendAgent, BreakoutAgent, DonchianChannelAgent
import time
import re
import json
import os
import sys
import atexit
import queue
import threading
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass, field
import math
from statistics import mean, pstdev
from typing import Any, Iterable, Optional
from telegram_control import TelegramController
# Strategy comment registry for MT5-safe order comments
STRATEGY_COMMENT_REGISTRY = {
    "momentum trend": ("MT", "momentum_trend"),
    "ma crossover": ("MA", "ma_crossover"),
    "breakout": ("BO", "breakout"),
    "donchian channel": ("DC", "donchian_channel"),
    "mean reversion": ("MR", "mean_reversion"),
}

COMMENT_CODE_TO_PERF_KEY = {code: perf_key for code, perf_key in STRATEGY_COMMENT_REGISTRY.values()}
COMMENT_CODE_CACHE = dict(COMMENT_CODE_TO_PERF_KEY)


LOG_DIR = Path("logs")
LOG_FILE_PREFIX = "live_run"
MAX_LOG_FILES = 30


class ConsoleTee:
    """Duplicate stream writes to multiple destinations (e.g., console + file)."""

    def __init__(self, *streams):
        self._streams = tuple(stream for stream in streams if stream is not None)
        primary = next((stream for stream in self._streams if hasattr(stream, "encoding")), None)
        self.encoding = getattr(primary, "encoding", "utf-8")

    def write(self, data):
        for stream in list(self._streams):
            try:
                stream.write(data)
                stream.flush()
            except Exception:
                # Streams may be closed during interpreter shutdown; skip failures gracefully.
                continue
        return len(data)

    def flush(self):
        for stream in list(self._streams):
            try:
                stream.flush()
            except Exception:
                continue

    def isatty(self):
        return any(getattr(stream, "isatty", lambda: False)() for stream in self._streams)

    def fileno(self):
        for stream in self._streams:
            if hasattr(stream, "fileno"):
                try:
                    return stream.fileno()
                except OSError:
                    continue
        raise OSError("ConsoleTee has no fileno")


def _prune_old_logs(directory: Path, prefix: str, keep: int) -> None:
    if keep <= 0:
        return
    try:
        log_files = sorted(
            directory.glob(f"{prefix}_*.txt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        for stale_file in log_files[keep:]:
            try:
                stale_file.unlink()
            except FileNotFoundError:
                continue
    except Exception as exc:
        fallback_stream = getattr(sys, "__stderr__", None) or getattr(sys, "__stdout__", None)
        if fallback_stream is not None:
            fallback_stream.write(f"⚠️  Failed to prune old logs: {exc}\n")
            fallback_stream.flush()


def setup_run_logger() -> Path:
    """Mirror stdout and stderr to a timestamped log file for post-run analysis."""

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{LOG_FILE_PREFIX}_{timestamp}.txt"

    log_handle = open(log_path, "a", encoding="utf-8")
    original_stdout = sys.stdout
    original_stderr = sys.stderr

    sys.stdout = ConsoleTee(original_stdout, log_handle)
    sys.stderr = ConsoleTee(original_stderr, log_handle)

    def _close_handle():
        try:
            log_handle.flush()
        finally:
            log_handle.close()

    atexit.register(_close_handle)
    _prune_old_logs(LOG_DIR, LOG_FILE_PREFIX, MAX_LOG_FILES)
    return log_path


RUN_LOG_FILE = setup_run_logger()
print(f"📝 Logging live run to {RUN_LOG_FILE.resolve()} (retaining last {MAX_LOG_FILES} logs)")

run_baseline_snapshot: dict[str, float | str | None] = {
    "balance": None,
    "equity": None,
    "timestamp": None,
}


class EmergencyStop(RuntimeError):
    """Raised when the remote kill switch requests an immediate shutdown."""


SCRIPT_START_TIME = datetime.now(timezone.utc)
TELEGRAM_REPLAY_GRACE_SECONDS = 5
KILL_CONFIRM_WINDOW_SECONDS = 45

kill_switch_state: dict[str, Any] = {
    "pending": False,
    "requested_at": None,
    "expires_at": None,
    "user_id": None,
    "reason": "",
    "engaged": False,
    "engaged_at": None,
    "confirmed_by": None,
    "notified": False,
    "suppress_resume_notify": False,
}

DEAL_ENTRY_IN = getattr(mt5, "DEAL_ENTRY_IN", 0)
DEAL_ENTRY_OUT = getattr(mt5, "DEAL_ENTRY_OUT", 1)
DEAL_ENTRY_INOUT = getattr(mt5, "DEAL_ENTRY_INOUT", 2)
DEAL_REASON_SL = getattr(mt5, "DEAL_REASON_SL", 0)
DEAL_REASON_TP = getattr(mt5, "DEAL_REASON_TP", 0)
DEAL_REASON_CLIENT = getattr(mt5, "DEAL_REASON_CLIENT", 0)
DEAL_REASON_MOBILE = getattr(mt5, "DEAL_REASON_MOBILE", 0)
DEAL_REASON_WEB = getattr(mt5, "DEAL_REASON_WEB", 0)
DEAL_REASON_EXPERT = getattr(mt5, "DEAL_REASON_EXPERT", 0)
DEAL_REASON_SCRIPT = getattr(mt5, "DEAL_REASON_SCRIPT", 0)

MANUAL_CLOSE_REASONS = {
    DEAL_REASON_CLIENT,
    DEAL_REASON_MOBILE,
    DEAL_REASON_WEB,
    DEAL_REASON_EXPERT,
    DEAL_REASON_SCRIPT,
}


def _generate_comment_code(label: str) -> str:
    initials = ''.join(word[0].upper() for word in re.split(r"\s+", label) if word)
    if not initials:
        return "ST"
    return initials[:3]


def build_order_comment(strategy_label: str, signal: str, priority: float) -> str:
    """Create a compact MT5-safe comment encoding strategy, side, and priority."""
    label_key = strategy_label.strip().lower()
    code = STRATEGY_COMMENT_REGISTRY.get(label_key, (None, None))[0]
    perf_key = STRATEGY_COMMENT_REGISTRY.get(label_key, (None, None))[1]
    if code is None:
        code = _generate_comment_code(strategy_label)
        perf_key = label_key.replace(' ', '_') if label_key else "strategy"
    COMMENT_CODE_CACHE[code] = perf_key

    side = 'B' if signal.lower() == 'buy' else 'S'
    priority_tag = max(0, min(99, int(round(priority))))
    return f"{code}{side}{priority_tag:02d}"

# Strategy Performance Tracking
PERFORMANCE_FILE = "strategy_performance.json"
strategy_performance = defaultdict(lambda: {
    'total_trades': 0,
    'winning_trades': 0,
    'total_pnl': 0.0,
    'win_rate': 0.0,
    'avg_win': 0.0,
    'avg_loss': 0.0,
    'profit_factor': 0.0,
    'last_updated': None
})

# Performance tracking settings
PERFORMANCE_LOOKBACK_DAYS = 30  # Track performance over last 30 days
MIN_TRADES_FOR_ADAPTATION = 10  # Minimum trades before adjusting weights

# Risk guard settings
RISK_GUARD_FILE = "risk_guard_state.json"
DAILY_MAX_DRAWDOWN = 0.12  # 12% maximum drop from daily peak
WEEKLY_MAX_DRAWDOWN = 0.20  # 20% maximum drop from weekly peak
RISK_RECOVERY_THRESHOLD = 0.05  # reduce risk once drawdown exceeds 5%
RISK_RECOVERY_MIN_FACTOR = 0.35  # never risk more than 35% of base risk when deep in drawdown
RISK_GUARD_ENABLED = False  # Temporarily disabled per latest instructions

risk_guard_state: dict[str, object] = {}
soft_guard_state: dict[str, float | bool] = {
    "blocked": False,
    "throttle": 1.0,
    "drawdown": 0.0,
}

margin_guard_state: dict[str, float | bool | str] = {
    "allowed": True,
    "throttle": 1.0,
    "status": "clear",
    "usage": 0.0,
    "free_ratio": 1.0,
    "severity": 0.0,
    "last_alert_status": "clear",
    "last_alert_ts": None,
    "last_alert_bucket": -1.0,
}

counter_signal_tracker: dict[str, dict[str, float | int | str]] = {}

# Telegram remote-control infrastructure
TELEGRAM_COMMAND_QUEUE: "queue.Queue[dict[str, object]]" = queue.Queue()
TELEGRAM_CONTROLLER: TelegramController | None = None
TELEGRAM_ALLOWED_IDS: set[int] = set()
TELEGRAM_KNOWN_CHATS: set[int] = set()
TELEGRAM_POLL_SECONDS = 2.5
TELEGRAM_PAUSE_SLEEP_SECONDS = 3.0
DEFAULT_TELEGRAM_ALLOWED_IDS: set[int] = {6401257809}
TELEGRAM_SETTINGS_CANDIDATES: tuple[Path, ...] = (
    Path("telegram_settings.json"),
    Path("live/telegram_settings.json"),
)

control_state_lock = threading.Lock()
kill_state_lock = threading.Lock()
control_state: dict[str, object | None] = {
    "paused": False,
    "pause_reason": "",
    "paused_since": None,
    "pause_notified": False,
    "last_user_id": None,
    "last_source": "startup",
}

risk_profile_state: dict[str, object | None] = {
    "mode": "medium",
    "label": "Balanced growth",
    "applied_at": None,
    "settings": {},
}

def register_telegram_chat(chat_id: int) -> None:
    if not isinstance(chat_id, int):
        return
    TELEGRAM_KNOWN_CHATS.add(chat_id)


def telegram_notify(message: str, chat_ids: Iterable[int] | None = None) -> None:
    if not message:
        return
    targets = list(chat_ids) if chat_ids is not None else list(TELEGRAM_KNOWN_CHATS)
    if not targets:
        if TELEGRAM_ALLOWED_IDS:
            targets = list(TELEGRAM_ALLOWED_IDS)
        else:
            return

    if TELEGRAM_CONTROLLER:
        TELEGRAM_CONTROLLER.broadcast_message(message, targets)


def load_strategy_performance():
    """Load strategy performance from file."""
    global strategy_performance
    try:
        with open(PERFORMANCE_FILE, 'r') as f:
            data = json.load(f)
            strategy_performance.update(data)
        print(f"📊 Loaded strategy performance data from {PERFORMANCE_FILE}")
    except FileNotFoundError:
        print(f"📊 No existing performance file found, starting fresh tracking")
    except Exception as e:
        print(f"Error loading performance data: {e}")

def save_strategy_performance():
    """Save strategy performance to file."""
    try:
        with open(PERFORMANCE_FILE, 'w') as f:
            json.dump(dict(strategy_performance), f, indent=2)
    except Exception as e:
        print(f"Error saving performance data: {e}")


def update_strategy_performance(strategy_name: str, symbol: str, pnl: float):
    """Update performance metrics for a strategy."""
    key = f"{strategy_name}_{symbol}"
    perf = strategy_performance[key]
    
    perf['total_trades'] += 1
    perf['total_pnl'] += pnl
    perf['last_updated'] = datetime.now().isoformat()
    
    if pnl > 0:
        perf['winning_trades'] += 1
    
    # Calculate metrics
    if perf['total_trades'] > 0:
        perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
    
    # For detailed win/loss analysis, we'd need to track individual trades
    # For now, use simplified profit factor approximation
    if perf['total_trades'] >= MIN_TRADES_FOR_ADAPTATION:
        perf['profit_factor'] = max(0.1, perf['total_pnl'] / max(1.0, perf['total_trades']))
    
    save_strategy_performance()
    print(f"📈 Updated performance for {strategy_name} on {symbol}: PnL={pnl:.2f}, WR={perf['win_rate']:.1%}, Trades={perf['total_trades']}")


def summarize_open_positions(positions) -> str:
    """Aggregate MT5 position objects into a compact summary string."""
    if not positions:
        return ""

    aggregated: dict[str, dict[str, float]] = defaultdict(lambda: {
        'count': 0,
        'buy': 0.0,
        'sell': 0.0,
        'pnl': 0.0,
    })

    for pos in positions:
        symbol = getattr(pos, 'symbol', 'UNKNOWN')
        stats = aggregated[symbol]
        stats['count'] += 1
        stats['pnl'] += float(getattr(pos, 'profit', 0.0) or 0.0)
        volume = float(getattr(pos, 'volume', 0.0) or 0.0)
        if getattr(pos, 'type', None) == mt5.POSITION_TYPE_BUY:
            stats['buy'] += volume
        else:
            stats['sell'] += volume

    parts: list[str] = []
    for symbol in sorted(aggregated.keys()):
        stats = aggregated[symbol]
        net_vol = stats['buy'] - stats['sell']
        parts.append(
            f"{symbol}: {int(stats['count'])} pos (buy {stats['buy']:.2f}, sell {stats['sell']:.2f}, net {net_vol:+.2f}, pnl {stats['pnl']:.2f})"
        )
    return " | ".join(parts)


def _format_timedelta_compact(delta: timedelta) -> str:
    total_seconds = int(max(0.0, delta.total_seconds()))
    days, remainder = divmod(total_seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    parts: list[str] = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        if minutes:
            parts.append(f"{minutes}m")
        elif not parts:
            parts.append(f"{seconds}s")
    if not parts:
        parts.append("0s")
    return ''.join(parts)


def _clean_comment(comment: str | None) -> str:
    if not comment:
        return ""
    compact = " ".join(str(comment).strip().split())
    return compact[:48]


def build_position_snapshot(position, now_utc: datetime) -> dict[str, object]:
    symbol = getattr(position, 'symbol', 'UNKNOWN') or 'UNKNOWN'
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None:
        try:
            mt5.symbol_select(symbol, True)
        except Exception:
            pass
        symbol_info = mt5.symbol_info(symbol)

    point = float(getattr(symbol_info, 'point', 0.0) or 0.0) if symbol_info else 0.0
    digits = int(getattr(symbol_info, 'digits', 5) or 5) if symbol_info else 5

    direction = 'BUY' if getattr(position, 'type', None) == mt5.POSITION_TYPE_BUY else 'SELL'
    ticket = int(getattr(position, 'ticket', 0) or 0)
    volume = float(getattr(position, 'volume', 0.0) or 0.0)
    price_open = float(getattr(position, 'price_open', 0.0) or 0.0)
    price_current = float(getattr(position, 'price_current', 0.0) or 0.0)

    def _normalize_price(value):
        try:
            numeric = float(value)
            return numeric if numeric > 0 else None
        except (TypeError, ValueError):
            return None

    sl = _normalize_price(getattr(position, 'sl', None))
    tp = _normalize_price(getattr(position, 'tp', None))

    run_points = None
    if point > 0 and price_open and price_current:
        if direction == 'BUY':
            run_points = (price_current - price_open) / point
        else:
            run_points = (price_open - price_current) / point

    sl_points = None
    if sl is not None and point > 0 and price_current:
        if direction == 'BUY':
            sl_points = max(0.0, (price_current - sl) / point)
        else:
            sl_points = max(0.0, (sl - price_current) / point)

    tp_points = None
    if tp is not None and point > 0 and price_current:
        if direction == 'BUY':
            tp_points = max(0.0, (tp - price_current) / point)
        else:
            tp_points = max(0.0, (price_current - tp) / point)

    profit = float(getattr(position, 'profit', 0.0) or 0.0)
    swap = float(getattr(position, 'swap', 0.0) or 0.0)
    commission = float(getattr(position, 'commission', 0.0) or 0.0)
    margin = getattr(position, 'margin', None)
    try:
        margin_val = float(margin) if margin is not None else None
    except (TypeError, ValueError):
        margin_val = None

    time_open = getattr(position, 'time', None)
    open_time_str = "n/a"
    age_str = "n/a"
    if time_open:
        try:
            open_dt = datetime.fromtimestamp(time_open, tz=timezone.utc)
            open_time_str = open_dt.strftime('%Y-%m-%d %H:%M:%S')
            age_str = _format_timedelta_compact(now_utc - open_dt)
        except Exception:
            open_time_str = "n/a"
            age_str = "n/a"

    comment = _clean_comment(getattr(position, 'comment', ''))
    magic = int(getattr(position, 'magic', 0) or 0)

    return {
        'symbol': symbol,
        'direction': direction,
        'ticket': ticket,
        'volume': volume,
        'price_open': price_open,
        'price_current': price_current,
        'digits': digits,
        'run_points': run_points,
        'sl': sl,
        'tp': tp,
        'sl_points': sl_points,
        'tp_points': tp_points,
        'profit': profit,
        'swap': swap,
        'commission': commission,
        'margin': margin_val,
        'comment': comment,
        'magic': magic,
        'age': age_str,
        'open_time': open_time_str,
    }


def compose_position_line(snapshot: dict[str, object], verbose: bool = False) -> str:
    digits = int(snapshot.get('digits', 5) or 5)
    price_open = float(snapshot.get('price_open') or 0.0)
    price_current = float(snapshot.get('price_current') or 0.0)
    price_open_fmt = f"{price_open:.{digits}f}" if price_open else "0"
    price_current_fmt = f"{price_current:.{digits}f}" if price_current else "0"

    run_points = snapshot.get('run_points')
    run_component = f" ({run_points:+.0f} pts)" if run_points is not None else ""

    sl_value = snapshot.get('sl')
    tp_value = snapshot.get('tp')
    sl_points = snapshot.get('sl_points')
    tp_points = snapshot.get('tp_points')

    def _format_price(value, points):
        if value is None:
            return "—"
        price_fragment = f"{float(value):.{digits}f}"
        if points is not None:
            price_fragment += f" ({points:.0f} pts)"
        return price_fragment

    sl_fragment = _format_price(sl_value, sl_points)
    tp_fragment = _format_price(tp_value, tp_points)

    base = (
        f"#{int(snapshot.get('ticket', 0))} {snapshot.get('direction', '?')} "
        f"{float(snapshot.get('volume', 0.0)):.2f}@{price_open_fmt}→{price_current_fmt}{run_component}"
    )

    profit = float(snapshot.get('profit') or 0.0)
    swap = float(snapshot.get('swap') or 0.0)
    commission = float(snapshot.get('commission') or 0.0)
    margin = snapshot.get('margin')
    comment = snapshot.get('comment') or ""
    magic = int(snapshot.get('magic', 0) or 0)
    age = snapshot.get('age', 'n/a')
    opened = snapshot.get('open_time', 'n/a')

    segments = [
        f"PnL {profit:+.2f}",
        f"SL {sl_fragment}",
        f"TP {tp_fragment}",
        f"Age {age}",
    ]

    if verbose:
        segments.extend([
            f"Swap {swap:+.2f}",
            f"Comm {commission:+.2f}",
        ])
        if margin not in (None, 0):
            segments.append(f"Margin {float(margin):.2f}")
        segments.append(f"Opened {opened}")
        if magic:
            segments.append(f"Magic {magic}")
        if comment:
            segments.append(f"Tag {comment}")
    else:
        if comment:
            segments.append(f"Tag {comment}")

    return base + " | " + " | ".join(segments)


def log_position_details(positions, header: str = "📘 Position detail ledger:") -> None:
    if not positions:
        return
    try:
        now_utc = datetime.now(timezone.utc)
        snapshots = [build_position_snapshot(pos, now_utc) for pos in positions]
        snapshots.sort(key=lambda snap: (snap.get('symbol', ''), snap.get('direction', ''), snap.get('ticket', 0)))
        print(header)
        for snap in snapshots:
            line = compose_position_line(snap, verbose=True)
            print(f"   • {snap.get('symbol', 'UNKNOWN')} {line}")
    except Exception as exc:
        print(f"⚠️  Failed to log detailed position info: {exc}")


def _week_start(date_obj: datetime) -> datetime:
    return (date_obj - timedelta(days=date_obj.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def load_risk_guard_state() -> dict[str, object]:
    global risk_guard_state
    if not RISK_GUARD_ENABLED:
        risk_guard_state = {}
        return risk_guard_state
    try:
        with open(RISK_GUARD_FILE, "r", encoding="utf-8") as handle:
            data = json.load(handle)
            if isinstance(data, dict):
                risk_guard_state = data
            else:
                risk_guard_state = {}
    except FileNotFoundError:
        risk_guard_state = {}
    except Exception as exc:
        print(f"⚠️  Failed to load risk guard state: {exc}")
        risk_guard_state = {}
    return risk_guard_state


def save_risk_guard_state() -> None:
    if not RISK_GUARD_ENABLED:
        return
    try:
        with open(RISK_GUARD_FILE, "w", encoding="utf-8") as handle:
            json.dump(risk_guard_state, handle, indent=2)
    except Exception as exc:
        print(f"⚠️  Failed to save risk guard state: {exc}")


def _reset_daily_state(now: datetime, equity: float) -> None:
    risk_guard_state["day"] = now.date().isoformat()
    risk_guard_state["daily_start_equity"] = equity
    risk_guard_state["daily_peak_equity"] = equity
    risk_guard_state["daily_drawdown"] = 0.0
    risk_guard_state["daily_blocked"] = False


def _reset_weekly_state(now: datetime, equity: float) -> None:
    week_anchor = _week_start(now)
    risk_guard_state["week_start"] = week_anchor.date().isoformat()
    risk_guard_state["weekly_start_equity"] = equity
    risk_guard_state["weekly_peak_equity"] = equity
    risk_guard_state["weekly_drawdown"] = 0.0
    risk_guard_state["weekly_blocked"] = False


def update_risk_guard(equity: float, now: datetime | None = None) -> None:
    if now is None:
        now = datetime.now()

    if not RISK_GUARD_ENABLED:
        risk_guard_state.clear()
        return

    if not risk_guard_state:
        _reset_daily_state(now, equity)
        _reset_weekly_state(now, equity)

    if risk_guard_state.get("day") != now.date().isoformat():
        _reset_daily_state(now, equity)

    current_week = _week_start(now).date().isoformat()
    if risk_guard_state.get("week_start") != current_week:
        _reset_weekly_state(now, equity)

    risk_guard_state["equity"] = equity

    daily_peak = max(risk_guard_state.get("daily_peak_equity", equity), equity)
    risk_guard_state["daily_peak_equity"] = daily_peak
    if daily_peak > 0:
        daily_drawdown = max(0.0, (daily_peak - equity) / daily_peak)
    else:
        daily_drawdown = 0.0
    risk_guard_state["daily_drawdown"] = daily_drawdown
    if daily_drawdown >= DAILY_MAX_DRAWDOWN:
        if not risk_guard_state.get("daily_blocked"):
            print(f"🚨 Daily drawdown {daily_drawdown:.1%} breached {DAILY_MAX_DRAWDOWN:.0%} limit. Pausing trading until next session.")
        risk_guard_state["daily_blocked"] = True

    weekly_peak = max(risk_guard_state.get("weekly_peak_equity", equity), equity)
    risk_guard_state["weekly_peak_equity"] = weekly_peak
    if weekly_peak > 0:
        weekly_drawdown = max(0.0, (weekly_peak - equity) / weekly_peak)
    else:
        weekly_drawdown = 0.0
    risk_guard_state["weekly_drawdown"] = weekly_drawdown
    if weekly_drawdown >= WEEKLY_MAX_DRAWDOWN:
        if not risk_guard_state.get("weekly_blocked"):
            print(f"🚨 Weekly drawdown {weekly_drawdown:.1%} breached {WEEKLY_MAX_DRAWDOWN:.0%} limit. Holding fire until new week.")
        risk_guard_state["weekly_blocked"] = True

    save_risk_guard_state()


def risk_guard_allow_trade(now: datetime | None = None) -> bool:
    if not RISK_GUARD_ENABLED:
        return True
    if now is None:
        now = datetime.now()
    if not risk_guard_state:
        return True

    if risk_guard_state.get("daily_blocked"):
        if risk_guard_state.get("day") != now.date().isoformat():
            risk_guard_state["daily_blocked"] = False
        else:
            return False

    week_anchor = _week_start(now).date().isoformat()
    if risk_guard_state.get("weekly_blocked"):
        if risk_guard_state.get("week_start") != week_anchor:
            risk_guard_state["weekly_blocked"] = False
        else:
            return False

    return True


def risk_guard_drawdown_factor() -> float:
    if not RISK_GUARD_ENABLED:
        return 1.0
    if not risk_guard_state:
        return 1.0
    daily_dd = float(risk_guard_state.get("daily_drawdown", 0.0))
    weekly_dd = float(risk_guard_state.get("weekly_drawdown", 0.0))
    dd = max(daily_dd, weekly_dd)
    if dd <= RISK_RECOVERY_THRESHOLD:
        return 1.0
    span = max(1e-6, max(DAILY_MAX_DRAWDOWN, WEEKLY_MAX_DRAWDOWN) - RISK_RECOVERY_THRESHOLD)
    scaled = min(1.0, (dd - RISK_RECOVERY_THRESHOLD) / span)
    return max(RISK_RECOVERY_MIN_FACTOR, 1.0 - scaled)


def evaluate_soft_guard(balance: float | None, equity: float | None) -> dict[str, float | bool | None]:
    """Assess soft guard state based on current balance/equity."""
    if not SOFT_GUARD_ENABLED or balance is None or balance <= 0 or equity is None:
        soft_guard_state.update({"blocked": False, "throttle": 1.0, "drawdown": 0.0, "status": "clear"})
        return {"blocked": False, "throttle": 1.0, "drawdown": 0.0, "transition": None, "status": "clear"}

    previous_blocked = bool(soft_guard_state.get("blocked", False))
    previous_throttle = float(soft_guard_state.get("throttle", 1.0) or 1.0)
    previous_status = soft_guard_state.get("status", "clear")

    drawdown = max(0.0, (balance - equity) / balance)

    if previous_blocked:
        blocked = drawdown > SOFT_GUARD_RESUME
    else:
        blocked = drawdown >= SOFT_GUARD_LIMIT

    status = "blocked" if blocked else "clear"
    if blocked:
        throttle = 0.0
    else:
        if drawdown <= SOFT_GUARD_RESUME:
            throttle = 1.0
            status = "clear"
        elif drawdown < SOFT_GUARD_CAUTION:
            throttle = 0.95
            status = "soft"
        elif drawdown < SOFT_GUARD_ALERT:
            span = max(1e-6, SOFT_GUARD_ALERT - SOFT_GUARD_CAUTION)
            ratio = (drawdown - SOFT_GUARD_CAUTION) / span
            throttle = max(0.75, 0.9 - ratio * 0.15)
            status = "caution"
        else:
            span = max(1e-6, SOFT_GUARD_LIMIT - SOFT_GUARD_ALERT)
            ratio = min(1.0, max(0.0, (drawdown - SOFT_GUARD_ALERT) / span))
            throttle = max(SOFT_GUARD_MIN_THROTTLE, 0.75 - ratio * (0.75 - SOFT_GUARD_MIN_THROTTLE))
            status = "alert"

    transition: dict[str, float | str] | None = None
    if blocked != previous_blocked:
        transition = {"type": "block" if blocked else "resume"}
    elif status != previous_status:
        transition = {"type": "status", "value": status}
    elif abs(throttle - previous_throttle) >= 0.05:
        transition = {"type": "throttle", "value": throttle}

    soft_guard_state.update({"blocked": blocked, "throttle": throttle, "drawdown": drawdown, "status": status})
    return {"blocked": blocked, "throttle": throttle, "drawdown": drawdown, "transition": transition, "status": status}


def evaluate_margin_guard(account_info) -> dict[str, float | bool | str | None]:
    """Throttle risk when margin usage/free margin ratios get stretched."""
    if not MARGIN_GUARD_ENABLED or not account_info:
        margin_guard_state.update({
            "allowed": True,
            "throttle": 1.0,
            "status": "clear",
            "usage": 0.0,
            "free_ratio": 1.0,
            "severity": 0.0,
        })
        return {
            "allowed": True,
            "throttle": 1.0,
            "usage": 0.0,
            "free_ratio": 1.0,
            "transition": None,
            "status": "clear",
            "severity": 0.0,
        }

    equity = float(getattr(account_info, "equity", 0.0) or 0.0)
    margin = float(getattr(account_info, "margin", 0.0) or 0.0)
    free_margin = float(getattr(account_info, "margin_free", 0.0) or 0.0)

    if equity <= 0.0:
        margin_guard_state.update({
            "allowed": False,
            "throttle": 0.0,
            "status": "blocked",
            "usage": 1.0,
            "free_ratio": 0.0,
            "severity": 1.0,
        })
        return {
            "allowed": False,
            "throttle": 0.0,
            "usage": 1.0,
            "free_ratio": 0.0,
            "transition": {"type": "block"},
            "status": "blocked",
            "severity": 1.0,
        }

    usage = margin / equity if equity > 0 else 0.0
    free_ratio = free_margin / equity if equity > 0 else 1.0

    previous_allowed = bool(margin_guard_state.get("allowed", True))
    previous_throttle = float(margin_guard_state.get("throttle", 1.0) or 1.0)
    previous_status = margin_guard_state.get("status", "clear")

    blocked = usage >= MARGIN_USAGE_BLOCK or free_ratio <= MARGIN_FREE_RATIO_BLOCK
    if not blocked and previous_status == "blocked":
        if usage >= MARGIN_USAGE_RESUME or free_ratio <= MARGIN_FREE_RATIO_RESUME:
            blocked = True

    severity = 0.0
    allowed = not blocked
    throttle = 1.0
    status = "clear"

    if blocked:
        allowed = False
        throttle = 0.0
        status = "blocked"
        severity = 1.0
    else:
        throttle_active = (usage >= MARGIN_USAGE_THROTTLE or free_ratio <= MARGIN_FREE_RATIO_THROTTLE)
        if not throttle_active and previous_status == "throttle":
            if usage >= MARGIN_USAGE_THROTTLE_RESUME or free_ratio <= MARGIN_FREE_RATIO_THROTTLE_RESUME:
                throttle_active = True

        if throttle_active:
            status = "throttle"
            usage_severity = 0.0
            free_severity = 0.0
            if usage >= MARGIN_USAGE_THROTTLE:
                span = max(1e-6, MARGIN_USAGE_BLOCK - MARGIN_USAGE_THROTTLE)
                usage_severity = min(1.0, max(0.0, (usage - MARGIN_USAGE_THROTTLE) / span))
            if free_ratio <= MARGIN_FREE_RATIO_THROTTLE:
                span = max(1e-6, MARGIN_FREE_RATIO_THROTTLE - MARGIN_FREE_RATIO_BLOCK)
                free_severity = min(1.0, max(0.0, (MARGIN_FREE_RATIO_THROTTLE - free_ratio) / span))
            severity = max(usage_severity, free_severity)
            if severity > 0.0:
                severity = min(1.0, pow(severity, MARGIN_GUARD_SEVERITY_EXPONENT))
            throttle = max(MARGIN_GUARD_MIN_THROTTLE, 1.0 - severity * (1.0 - MARGIN_GUARD_MIN_THROTTLE))

            if previous_status == "blocked":
                throttle = min(throttle, max(MARGIN_GUARD_MIN_THROTTLE, previous_throttle))
        else:
            severity = 0.0

    if status == "clear" and previous_status == "throttle":
        if usage >= MARGIN_USAGE_THROTTLE_RESUME or free_ratio <= MARGIN_FREE_RATIO_THROTTLE_RESUME:
            status = "throttle"
            severity = min(1.0, pow(max(0.0, previous_throttle - MARGIN_GUARD_MIN_THROTTLE) / max(1e-6, 1.0 - MARGIN_GUARD_MIN_THROTTLE), MARGIN_GUARD_SEVERITY_EXPONENT))
            throttle = max(MARGIN_GUARD_MIN_THROTTLE, min(previous_throttle, 1.0))

    transition: dict[str, float | str] | None = None
    if allowed != previous_allowed:
        transition = {"type": "resume" if allowed else "block"}
    elif status != previous_status:
        transition = {"type": "status", "value": status}
    elif abs(throttle - previous_throttle) >= 0.05:
        transition = {"type": "throttle", "value": throttle}

    margin_guard_state.update({
        "allowed": allowed,
        "throttle": throttle,
        "status": status,
        "usage": usage,
        "free_ratio": free_ratio,
        "severity": severity,
    })

    payload = {
        "allowed": allowed,
        "throttle": throttle,
        "usage": usage,
        "free_ratio": free_ratio,
        "transition": transition,
        "status": status,
        "severity": severity,
    }

    maybe_notify_margin_guard_event(payload, transition)

    return payload


def _parse_allowed_ids(raw: str | None) -> set[int]:
    ids: set[int] = set()
    if not raw:
        return ids
    for chunk in re.split(r"[;,]\s*", raw):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.add(int(chunk))
        except ValueError:
            print(f"⚠️  Ignoring invalid Telegram ID '{chunk}' from configuration.")
    return ids


def load_telegram_settings() -> tuple[str | None, set[int]]:
    token = os.getenv("ULTIMA_TELEGRAM_BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
    allowed_ids = _parse_allowed_ids(
        os.getenv("ULTIMA_TELEGRAM_ALLOWED_IDS") or os.getenv("TELEGRAM_ALLOWED_IDS")
    )

    for candidate in TELEGRAM_SETTINGS_CANDIDATES:
        if not candidate.exists():
            continue
        try:
            with open(candidate, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"⚠️  Failed to read Telegram settings from {candidate}: {exc}")
            continue

        token = token or payload.get("bot_token") or payload.get("token") or token
        raw_ids = payload.get("allowed_user_ids") or payload.get("allowed_ids") or []
        for entry in raw_ids:
            try:
                allowed_ids.add(int(entry))
            except (TypeError, ValueError):
                print(f"⚠️  Ignoring invalid Telegram ID '{entry}' in {candidate}.")

    if not allowed_ids and DEFAULT_TELEGRAM_ALLOWED_IDS:
        allowed_ids = set(DEFAULT_TELEGRAM_ALLOWED_IDS)

    return token, allowed_ids


def initialize_telegram_control() -> TelegramController | None:
    global TELEGRAM_CONTROLLER, TELEGRAM_ALLOWED_IDS

    if TELEGRAM_CONTROLLER is not None:
        return TELEGRAM_CONTROLLER

    token, allowed_ids = load_telegram_settings()
    TELEGRAM_ALLOWED_IDS = set(allowed_ids)

    if not token:
        print("📨 Telegram control disabled: no bot token configured.")
        return None

    if not allowed_ids:
        print("📨 Telegram control disabled: no authorized user IDs configured.")
        return None

    try:
        controller = TelegramController(
            token,
            allowed_ids,
            TELEGRAM_COMMAND_QUEUE,
            poll_interval=TELEGRAM_POLL_SECONDS,
        )
    except RuntimeError as exc:
        print(f"⚠️  Telegram controller unavailable: {exc}")
        return None

    controller.start()
    atexit.register(controller.stop)
    TELEGRAM_CONTROLLER = controller
    obfuscated = ", ".join(str(uid) for uid in sorted(allowed_ids))
    print(f"📨 Telegram remote control ready for authorized ID(s): {obfuscated}")
    return controller


def set_trading_paused(
    paused: bool,
    *,
    reason: str = "",
    source: str = "manual",
    user_id: int | None = None,
) -> tuple[bool, dict[str, object | None]]:
    with control_state_lock:
        previous = bool(control_state.get("paused", False))
        control_state["paused"] = paused
        control_state["last_source"] = source
        if user_id is not None:
            control_state["last_user_id"] = user_id

        changed = previous != paused

        if paused:
            control_state["pause_reason"] = reason or control_state.get("pause_reason") or "Remote pause"
            control_state["paused_since"] = datetime.now(timezone.utc)
            control_state["pause_notified"] = False
        else:
            control_state["pause_reason"] = ""
            control_state["paused_since"] = None
            control_state["pause_notified"] = False

        snapshot = dict(control_state)

    if changed:
        state_text = "paused" if paused else "resumed"
        actor = f" by {source}" if source else ""
        print(f"🎛️ Trading {state_text}{actor} (reason: {reason or 'n/a'}).")

    return changed, snapshot


def is_trading_paused() -> bool:
    with control_state_lock:
        return bool(control_state.get("paused", False))


def _reset_kill_state() -> None:
    with kill_state_lock:
        kill_switch_state.update(
            {
                "pending": False,
                "requested_at": None,
                "expires_at": None,
                "user_id": None,
                "reason": "",
                "engaged": False,
                "engaged_at": None,
                "confirmed_by": None,
                "notified": False,
                "suppress_resume_notify": False,
            }
        )


def resume_kill_switch(user_id: int | None) -> tuple[bool, dict[str, Any]]:
    with kill_state_lock:
        engaged = bool(kill_switch_state.get("engaged"))
        pending = bool(kill_switch_state.get("pending"))
        if not engaged and not pending:
            return False, dict(kill_switch_state)
        kill_switch_state.update(
            {
                "pending": False,
                "requested_at": None,
                "expires_at": None,
                "user_id": None,
                "reason": "",
                "engaged": False,
                "engaged_at": None,
                "confirmed_by": None,
                "notified": False,
                "suppress_resume_notify": True,
            }
        )
        snapshot = dict(kill_switch_state)
    set_trading_paused(False, reason="Kill switch cleared", source="telegram", user_id=user_id)
    return True, snapshot


def _expire_kill_request_if_needed(now: Optional[datetime] = None) -> None:
    now_ts = now or datetime.now(timezone.utc)
    with kill_state_lock:
        expires_at = kill_switch_state.get("expires_at")
        if not kill_switch_state.get("pending"):
            return
        if isinstance(expires_at, datetime) and expires_at <= now_ts:
            kill_switch_state["pending"] = False
            kill_switch_state["expires_at"] = None
            kill_switch_state["requested_at"] = None
            kill_switch_state["reason"] = ""


def request_kill_switch(user_id: int | None, reason: str | None = None) -> tuple[bool, dict[str, Any]]:
    now_ts = datetime.now(timezone.utc)
    expires = now_ts + timedelta(seconds=KILL_CONFIRM_WINDOW_SECONDS)
    with kill_state_lock:
        if kill_switch_state.get("engaged"):
            return False, dict(kill_switch_state)
        if kill_switch_state.get("pending"):
            # Refresh expiry but keep original requester info
            kill_switch_state["expires_at"] = expires
            kill_switch_state["reason"] = reason or kill_switch_state.get("reason", "")
            return False, dict(kill_switch_state)

        kill_switch_state.update(
            {
                "pending": True,
                "requested_at": now_ts,
                "expires_at": expires,
                "user_id": user_id,
                "reason": reason or "",
            }
        )
        snapshot = dict(kill_switch_state)
    return True, snapshot


def cancel_kill_switch() -> dict[str, Any]:
    with kill_state_lock:
        if kill_switch_state.get("engaged"):
            snapshot = dict(kill_switch_state)
        else:
            _reset_kill_state()
            snapshot = dict(kill_switch_state)
    return snapshot


def confirm_kill_switch(user_id: int | None) -> tuple[bool, dict[str, Any]]:
    now_ts = datetime.now(timezone.utc)
    with kill_state_lock:
        expires_at = kill_switch_state.get("expires_at")
        if not kill_switch_state.get("pending"):
            return False, dict(kill_switch_state)
        if isinstance(expires_at, datetime) and expires_at < now_ts:
            _reset_kill_state()
            return False, dict(kill_switch_state)

        kill_switch_state["pending"] = False
        kill_switch_state["expires_at"] = None
        kill_switch_state["requested_at"] = None
        kill_switch_state["engaged"] = True
        kill_switch_state["engaged_at"] = now_ts
        kill_switch_state["confirmed_by"] = user_id
        kill_switch_state["notified"] = False
        snapshot = dict(kill_switch_state)
    set_trading_paused(True, reason="Emergency kill switch engaged", source="telegram", user_id=user_id)
    return True, snapshot


def is_kill_switch_engaged() -> bool:
    with kill_state_lock:
        return bool(kill_switch_state.get("engaged"))


def enforce_kill_switch() -> None:
    message: str | None = None
    with kill_state_lock:
        engaged = bool(kill_switch_state.get("engaged"))
        if not engaged:
            return
        if not kill_switch_state.get("notified"):
            message = "🛑 Emergency kill switch engaged — preparing to exit loop."
            kill_switch_state["notified"] = True
    if message:
        print(message)
    raise EmergencyStop("Kill switch engaged")


def format_risk_profile_summary(line_prefix: str = "• ") -> str:
    mode = str(risk_profile_state.get("mode") or "unknown").upper()
    label = str(risk_profile_state.get("label") or "")
    lines = [
        f"{line_prefix}Risk mode: {mode} — {label}",
        f"{line_prefix}Account risk/trade: {ACCOUNT_RISK_PER_TRADE:.1%}",
        f"{line_prefix}Risk multiplier cap: x{RISK_MULTIPLIER_MAX:.2f}",
        f"{line_prefix}Soft guard limit: {SOFT_GUARD_LIMIT:.0%} (alert {SOFT_GUARD_ALERT:.0%})",
        f"{line_prefix}Margin block: {MARGIN_USAGE_BLOCK:.0%} (throttle {MARGIN_USAGE_THROTTLE:.0%})",
    ]
    applied = risk_profile_state.get("applied_at")
    if applied:
        lines.append(f"{line_prefix}Last updated: {applied}")
    return "\n".join(lines)


def format_margin_guard_line(
    status: dict[str, object] | None = None,
    line_prefix: str = "• ",
) -> str:
    """Render a single-line summary for the margin guard."""
    state = status or margin_guard_state
    usage = float(state.get("usage", 0.0) or 0.0)
    free_ratio = float(state.get("free_ratio", 1.0) or 1.0)
    throttle = float(state.get("throttle", 1.0) or 1.0)
    severity = float(state.get("severity", 0.0) or 0.0)
    allowed = bool(state.get("allowed", True))
    guard_status = str(state.get("status") or ("blocked" if not allowed else "clear"))

    status_token = {
        "blocked": "⛔ Blocked",
        "throttle": "⚠️ Throttled",
        "clear": "✅ Clear",
    }.get(guard_status, f"ℹ️ {guard_status.title()}")

    if not allowed and guard_status != "blocked":
        status_token += " (orders halted)"

    throttle_display = f"x{max(0.0, throttle):.2f}"
    usage_display = f"{max(0.0, usage):.0%}"
    free_display = f"{max(0.0, free_ratio):.0%}"
    severity_display = f"{max(0.0, min(1.0, severity)):.0%}"

    return (
        f"{line_prefix}Margin guard {status_token} · usage {usage_display} · free {free_display} · "
        f"throttle {throttle_display} · severity {severity_display}"
    )


def format_guard_status_summary(line_prefix: str = "• ") -> str:
    """Produce a compact multi-guard snapshot for Telegram delivery."""
    lines: list[str] = []

    soft_status = str(soft_guard_state.get("status") or "clear")
    soft_blocked = bool(soft_guard_state.get("blocked", False))
    soft_throttle = float(soft_guard_state.get("throttle", 1.0) or 1.0)
    soft_drawdown = float(soft_guard_state.get("drawdown", 0.0) or 0.0)

    soft_token = {
        "blocked": "⛔ Blocked",
        "alert": "⚠️ Alert",
        "throttle": "⚠️ Alert",
        "caution": "🟠 Caution",
        "soft": "🟡 Soft",
        "clear": "✅ Clear",
    }.get(soft_status, soft_status.title())

    if soft_blocked and "Blocked" not in soft_token:
        soft_token = f"⛔ {soft_token}"

    lines.append(
        f"{line_prefix}Soft guard {soft_token} · drawdown {max(0.0, soft_drawdown):.1%} · throttle x{max(0.0, soft_throttle):.2f}"
    )

    lines.append(format_margin_guard_line(line_prefix=line_prefix))

    daily_dd = float(risk_guard_state.get("daily_drawdown", 0.0) or 0.0)
    weekly_dd = float(risk_guard_state.get("weekly_drawdown", 0.0) or 0.0)
    daily_blocked = bool(risk_guard_state.get("daily_blocked", False))
    weekly_blocked = bool(risk_guard_state.get("weekly_blocked", False))

    daily_token = "⛔" if daily_blocked else "📉"
    weekly_token = "⛔" if weekly_blocked else "📉"
    lines.append(
        f"{line_prefix}Daily guard {daily_token} {daily_dd:.1%}/{DAILY_MAX_DRAWDOWN:.0%} · "
        f"Weekly guard {weekly_token} {weekly_dd:.1%}/{WEEKLY_MAX_DRAWDOWN:.0%}"
    )

    recovery_cap = float(
        risk_multiplier_state.get("recovery_cap", RISK_MULTIPLIER_MAX) or RISK_MULTIPLIER_MAX
    )
    recovery_until = risk_multiplier_state.get("recovery_until")
    recovery_note = ""
    if isinstance(recovery_until, str):
        try:
            recovery_dt = datetime.fromisoformat(recovery_until)
            now = datetime.now(timezone.utc)
            if recovery_dt > now:
                recovery_note = f" · recovery until {_format_timedelta_compact(recovery_dt - now)}"
        except ValueError:
            recovery_note = ""

    last_violation = risk_multiplier_state.get("last_violation")
    violation_note = ""
    if isinstance(last_violation, str):
        try:
            violation_dt = datetime.fromisoformat(last_violation)
            now = datetime.now(timezone.utc)
            violation_note = f" (last guard bite {_format_timedelta_compact(now - violation_dt)} ago)"
        except ValueError:
            violation_note = ""

    lines.append(
        f"{line_prefix}Risk multiplier cap x{recovery_cap:.2f}/{RISK_MULTIPLIER_MAX:.2f}{recovery_note}{violation_note}"
    )

    return "\n".join(lines)


def format_active_positions_summary(
    positions: Iterable[Any] | None = None,
    line_prefix: str = "• ",
) -> str:
    """Summarize currently open MT5 positions in a Telegram-friendly string."""
    position_list = list(positions or mt5.positions_get() or [])
    if not position_list:
        return f"{line_prefix}No open positions."

    total_tickets = len(position_list)
    total_volume = sum(max(0.0, float(getattr(pos, "volume", 0.0) or 0.0)) for pos in position_list)
    net_pnl = sum(float(getattr(pos, "profit", 0.0) or 0.0) for pos in position_list)

    summary_line = summarize_open_positions(position_list)
    lines = [
        f"{line_prefix}Open positions: {total_tickets} tickets · {total_volume:.2f} lots · PnL {net_pnl:+.2f}",
    ]
    if summary_line:
        lines.append(f"{line_prefix}Breakdown: {summary_line}")
    return "\n".join(lines)


def format_exposure_summary(
    positions: Iterable[Any] | None = None,
    line_prefix: str = "• ",
) -> str:
    """Summarize symbol and currency exposure against configured guardrails."""
    position_list = list(positions or mt5.positions_get() or [])
    if not position_list:
        return f"{line_prefix}No open positions."

    symbol_stats: dict[str, dict[str, float]] = defaultdict(lambda: {
        "buy": 0.0,
        "sell": 0.0,
        "total": 0.0,
    })

    for pos in position_list:
        symbol = str(getattr(pos, "symbol", "")) or "UNKNOWN"
        volume = float(getattr(pos, "volume", 0.0) or 0.0)
        if volume <= 0:
            continue
        bucket = symbol_stats[_normalize_symbol_key(symbol) or symbol]
        if getattr(pos, "type", None) == mt5.POSITION_TYPE_SELL:
            bucket["sell"] += volume
        else:
            bucket["buy"] += volume
        bucket["total"] += volume

    def _format_notional(value: float) -> str:
        abs_value = abs(value)
        if abs_value >= 1_000_000:
            return f"{value / 1_000_000:.2f}M"
        if abs_value >= 1_000:
            return f"{value / 1_000:.1f}K"
        return f"{value:.0f}"

    symbol_chunks: list[str] = []
    for key in sorted(symbol_stats.keys()):
        stats = symbol_stats[key]
        same_side = max(stats["buy"], stats["sell"])
        limits = _get_symbol_exposure_limits(key)
        max_total = float(limits.get("max_total") or 0.0)
        max_same = float(limits.get("max_same") or 0.0)
        total_part = f"total {stats['total']:.2f}{f'/{max_total:.2f}' if max_total else ''}"
        same_part = f"same {same_side:.2f}{f'/{max_same:.2f}' if max_same else ''}"
        symbol_chunks.append(f"{key}: {total_part}, {same_part}")

    exposures_by_currency = _collect_currency_notional(position_list)
    equity = get_account_equity_quiet() or get_account_balance()
    currency_chunks: list[str] = []
    for currency in sorted(exposures_by_currency.keys()):
        notional = exposures_by_currency[currency]
        multiplier = float(
            EXPOSURE_CURRENCY_OVERRIDES.get(currency, EXPOSURE_CURRENCY_EQUITY_MULTIPLIER)
        )
        limit = equity * multiplier if equity else 0.0
        if limit > 0:
            ratio = min(9.99, max(0.0, notional / limit))
            currency_chunks.append(
                f"{currency} {_format_notional(notional)}/{_format_notional(limit)} ({ratio:.0%})"
            )
        else:
            currency_chunks.append(f"{currency} {_format_notional(notional)}")

    lines: list[str] = []
    if symbol_chunks:
        lines.append(f"{line_prefix}Symbols: " + " | ".join(symbol_chunks))
    if currency_chunks:
        lines.append(f"{line_prefix}Currencies: " + " | ".join(currency_chunks))

    total_volume = sum(stats["total"] for stats in symbol_stats.values())
    lines.append(f"{line_prefix}Total lots: {total_volume:.2f} across {len(position_list)} tickets")

    return "\n".join(lines)


def maybe_notify_margin_guard_event(
    status: dict[str, float | bool | str | None],
    transition: dict[str, float | str] | None,
) -> None:
    """Broadcast meaningful margin guard state changes via Telegram."""
    try:
        new_status = str(status.get("status") or "clear")
        severity = float(status.get("severity", 0.0) or 0.0)
        throttle = float(status.get("throttle", 1.0) or 1.0)
        usage = float(status.get("usage", 0.0) or 0.0)
        free_ratio = float(status.get("free_ratio", 1.0) or 1.0)

        last_status = str(margin_guard_state.get("last_alert_status") or "clear")
        last_bucket = int(float(margin_guard_state.get("last_alert_bucket", -1.0) or -1.0))

        event: str | None = None

        if transition:
            transition_type = str(transition.get("type") or "")
            if transition_type == "block":
                event = "status:blocked"
            elif transition_type == "resume":
                event = "status:clear"
            elif transition_type == "status":
                value = str(transition.get("value") or new_status)
                event = f"status:{value}"
            elif transition_type == "throttle":
                event = "throttle"

        if event is None and new_status != last_status:
            event = f"status:{new_status}"

        bucket_idx = -1
        if event in {None, "throttle"}:
            if new_status in {"throttle", "blocked"} and severity >= MARGIN_ALERT_SEVERITY_THRESHOLD:
                bucket_span = max(MARGIN_ALERT_REPEAT_BUCKET, 1e-6)
                bucket_idx = int(math.floor((severity - MARGIN_ALERT_SEVERITY_THRESHOLD) / bucket_span))
                if bucket_idx > last_bucket:
                    event = "severity"
            elif new_status in {"throttle", "blocked"}:
                margin_guard_state["last_alert_bucket"] = -1.0
            elif new_status not in {"throttle", "blocked"}:
                margin_guard_state["last_alert_bucket"] = -1.0

        if event is None:
            return

        status_value = new_status
        if event.startswith("status:"):
            status_value = event.split(":", 1)[1] or new_status
            if status_value == "block":
                status_value = "blocked"
            if status_value == "resume":
                status_value = "clear"

        header = ""
        usage_pct = f"{max(0.0, usage):.0%}"
        free_pct = f"{max(0.0, free_ratio):.0%}"

        if event == "severity":
            header = (
                f"⚠️ Margin guard severity {min(1.0, max(0.0, severity)):.0%} — usage {usage_pct}, free {free_pct}, throttle x{max(0.0, throttle):.2f}"
            )
        elif status_value == "blocked":
            header = (
                f"⛔ Margin guard BLOCKED — entries halted (usage {usage_pct}, free {free_pct})."
            )
        elif status_value == "throttle":
            header = (
                f"⚠️ Margin guard throttling exposure — usage {usage_pct}, free {free_pct}."
            )
        elif status_value == "clear":
            header = "✅ Margin guard recovered — entries re-enabled."
        else:
            header = f"ℹ️ Margin guard status {status_value}."

        details = format_margin_guard_line(status, line_prefix="• ")
        message_lines = [header, details]

        if status_value != "clear" or event == "severity":
            message_lines.append(format_active_positions_summary(line_prefix="• "))

        telegram_notify("\n".join(message_lines))

        margin_guard_state["last_alert_status"] = status_value
        margin_guard_state["last_alert_ts"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if status_value in {"throttle", "blocked"} and bucket_idx >= 0:
            margin_guard_state["last_alert_bucket"] = float(bucket_idx)
        elif status_value == "clear":
            margin_guard_state["last_alert_bucket"] = -1.0
    except Exception as exc:
        print(f"⚠️ Failed to dispatch margin guard alert: {exc}")


def truncate_for_telegram(text: str, limit: int = 3600) -> str:
    if len(text) <= limit:
        return text
    ellipsis = "…"
    return text[: limit - len(ellipsis)].rstrip() + ellipsis


def _get_session_start_time(default_window_hours: int = 36) -> datetime:
    timestamp = run_baseline_snapshot.get("timestamp")
    if isinstance(timestamp, str):
        try:
            return datetime.fromisoformat(timestamp) - timedelta(minutes=1)
        except ValueError:
            pass
    return datetime.now() - timedelta(hours=default_window_hours)


def _safe_deal_time(deal) -> datetime:
    try:
        ts = float(getattr(deal, "time", 0) or 0)
    except (TypeError, ValueError):
        ts = 0.0
    if ts <= 0:
        return datetime.now(timezone.utc)
    return datetime.fromtimestamp(ts, tz=timezone.utc)


def _session_base_equity() -> float:
    baseline = run_baseline_snapshot.get("equity")
    if baseline is not None:
        try:
            return float(baseline)
        except (TypeError, ValueError):
            pass
    account_info = mt5.account_info()
    if account_info:
        equity = getattr(account_info, "equity", None)
        balance = getattr(account_info, "balance", None)
        for candidate in (equity, balance):
            if candidate is not None:
                try:
                    return float(candidate)
                except (TypeError, ValueError):
                    continue
    return 0.0


def collect_session_performance(now: Optional[datetime] = None) -> dict[str, Any]:
    start = _get_session_start_time()
    end = now or datetime.now()
    try:
        deals = mt5.history_deals_get(start, end)
    except Exception:
        deals = None

    if not deals:
        return {
            "gross_profit": 0.0,
            "gross_loss": 0.0,
            "net_profit": 0.0,
            "partial_realized": 0.0,
            "partial_events": 0,
            "trades": 0,
            "wins": 0,
            "losses": 0,
            "breakeven": 0,
            "stop_loss_hits": 0,
            "take_profit_hits": 0,
            "manual_closes": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "largest_win": 0.0,
            "largest_loss": 0.0,
            "max_drawdown": 0.0,
            "equity_high": 0.0,
            "equity_low": 0.0,
            "sharpe": 0.0,
            "duration_minutes": 0.0,
            "avg_trade_duration": 0.0,
            "first_time": None,
            "last_time": None,
            "base_equity": _session_base_equity(),
        }

    ordered_deals = sorted(deals, key=_safe_deal_time)
    positions: dict[int, dict[str, Any]] = {}
    gross_profit = 0.0
    gross_loss = 0.0
    net_profit = 0.0
    partial_realized = 0.0
    partial_events = 0
    trades = 0
    wins = 0
    losses = 0
    breakeven = 0
    stop_loss_hits = 0
    take_profit_hits = 0
    manual_closes = 0
    largest_win = float("-inf")
    largest_loss = float("inf")
    equity = 0.0
    equity_high = 0.0
    equity_low = 0.0
    max_drawdown = 0.0
    positive_closes: list[float] = []
    negative_closes: list[float] = []
    trade_returns: list[float] = []
    trade_durations: list[float] = []
    first_time: datetime | None = None
    last_time: datetime | None = None
    base_equity = _session_base_equity()

    for deal in ordered_deals:
        deal_time = _safe_deal_time(deal)
        if first_time is None or deal_time < first_time:
            first_time = deal_time
        if last_time is None or deal_time > last_time:
            last_time = deal_time

        position_id = int(getattr(deal, "position_id", getattr(deal, "position", 0)) or 0)
        volume = float(getattr(deal, "volume", 0.0) or 0.0)
        profit = float(getattr(deal, "profit", 0.0) or 0.0)

        container = positions.setdefault(
            position_id,
            {
                "entry_volume": 0.0,
                "exit_volume": 0.0,
                "realized": 0.0,
                "first": deal_time,
                "direction": getattr(deal, "type", 0),
                "last_reason": None,
            },
        )

        entry_type = getattr(deal, "entry", DEAL_ENTRY_OUT)
        reason = getattr(deal, "reason", None)
        if entry_type == DEAL_ENTRY_IN:
            container["entry_volume"] += volume
            container.setdefault("first", deal_time)
            continue

        if container["entry_volume"] <= 0.0:
            container["entry_volume"] = volume

        container["exit_volume"] += volume
        container["realized"] += profit
        container["last"] = deal_time
        container["last_reason"] = reason

        net_profit += profit
        if profit >= 0:
            gross_profit += profit
        else:
            gross_loss += profit

        equity += profit
        equity_high = max(equity_high, equity)
        equity_low = min(equity_low, equity)
        drawdown = equity_high - equity
        max_drawdown = max(max_drawdown, drawdown)

        if entry_type in (DEAL_ENTRY_OUT, DEAL_ENTRY_INOUT):
            entry_volume = container.get("entry_volume", 0.0)
            exit_volume = container.get("exit_volume", 0.0)
            residual = entry_volume - exit_volume
            if residual > 1e-6:
                partial_realized += profit
                partial_events += 1

            if residual <= 1e-6:
                trade_profit = container.get("realized", 0.0)
                trades += 1
                if trade_profit > 1e-6:
                    wins += 1
                    positive_closes.append(trade_profit)
                elif trade_profit < -1e-6:
                    losses += 1
                    negative_closes.append(trade_profit)
                else:
                    breakeven += 1

                largest_win = max(largest_win, trade_profit)
                largest_loss = min(largest_loss, trade_profit)

                if base_equity > 0:
                    trade_returns.append(trade_profit / base_equity)

                opened_at = container.get("first")
                closed_at = container.get("last")
                if isinstance(opened_at, datetime) and isinstance(closed_at, datetime):
                    trade_durations.append(max(0.0, (closed_at - opened_at).total_seconds() / 60.0))

                final_reason = container.get("last_reason", reason)
                if final_reason == DEAL_REASON_SL:
                    stop_loss_hits += 1
                elif final_reason == DEAL_REASON_TP:
                    take_profit_hits += 1
                elif final_reason in MANUAL_CLOSE_REASONS:
                    manual_closes += 1

                container["entry_volume"] = 0.0
                container["exit_volume"] = 0.0
                container["realized"] = 0.0
                container["last_reason"] = None

    win_rate = (wins / trades) if trades else 0.0
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < -1e-6 else (gross_profit if gross_profit > 0 else 0.0)
    largest_win = largest_win if largest_win != float("-inf") else 0.0
    largest_loss = largest_loss if largest_loss != float("inf") else 0.0
    duration_minutes = 0.0
    if isinstance(first_time, datetime) and isinstance(last_time, datetime):
        duration_minutes = max(0.0, (last_time - first_time).total_seconds() / 60.0)
    avg_trade_duration = sum(trade_durations) / len(trade_durations) if trade_durations else 0.0

    sharpe = 0.0
    if len(trade_returns) >= 2:
        mean_return = mean(trade_returns)
        std_return = pstdev(trade_returns)
        if std_return > 0:
            sharpe = (mean_return / std_return) * math.sqrt(len(trade_returns))

    return {
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "net_profit": net_profit,
        "partial_realized": partial_realized,
        "partial_events": partial_events,
        "trades": trades,
        "wins": wins,
        "losses": losses,
        "breakeven": breakeven,
        "stop_loss_hits": stop_loss_hits,
        "take_profit_hits": take_profit_hits,
        "manual_closes": manual_closes,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "max_drawdown": max_drawdown,
        "equity_high": equity_high,
        "equity_low": equity_low,
        "sharpe": sharpe,
        "duration_minutes": duration_minutes,
        "avg_trade_duration": avg_trade_duration,
        "first_time": first_time,
        "last_time": last_time,
        "base_equity": base_equity,
    }


def compose_performance_snapshot() -> str:
    account_info = mt5.account_info()
    lines: list[str] = ["📊 Ultima performance snapshot"]

    with control_state_lock:
        paused = bool(control_state.get("paused", False))
        pause_reason = str(control_state.get("pause_reason") or "")
        paused_since = control_state.get("paused_since")

    if paused:
        duration_text = ""
        if isinstance(paused_since, datetime):
            duration = datetime.now(timezone.utc) - paused_since
            duration_text = f" for {_format_timedelta_compact(duration)}"
        reason_text = f" — {pause_reason}" if pause_reason else ""
        lines.append(f"• Run state: Paused{duration_text}{reason_text}")
    else:
        lines.append("• Run state: Live")

    if account_info:
        equity = float(getattr(account_info, "equity", 0.0) or 0.0)
        balance = float(getattr(account_info, "balance", 0.0) or 0.0)
        profit = float(getattr(account_info, "profit", 0.0) or 0.0)
        lines.append(
            f"• Balance {balance:.2f} | Equity {equity:.2f} | Open PnL {profit:+.2f}"
        )
    else:
        lines.append("• Account info unavailable (mt5.account_info() returned None)")

    performance = collect_session_performance()
    perf_trades = int(performance.get("trades", 0) or 0)
    perf_partials = int(performance.get("partial_events", 0) or 0)
    perf_net = float(performance.get("net_profit", 0.0) or 0.0)

    if perf_trades or perf_partials or abs(perf_net) > 0.01:
        gross_profit = float(performance.get("gross_profit", 0.0) or 0.0)
        gross_loss = float(performance.get("gross_loss", 0.0) or 0.0)
        win_rate = float(performance.get("win_rate", 0.0) or 0.0)
        profit_factor = performance.get("profit_factor", 0.0)
        if isinstance(profit_factor, float) and profit_factor > 1e9:
            profit_factor_display = "∞"
        elif perf_trades > 0 and float(performance.get("losses", 0) or 0) == 0 and gross_profit > 0:
            profit_factor_display = "∞"
        else:
            profit_factor_display = f"{float(profit_factor or 0.0):.2f}"

        sharpe = float(performance.get("sharpe", 0.0) or 0.0)
        max_dd = float(performance.get("max_drawdown", 0.0) or 0.0)
        equity_high = float(performance.get("equity_high", 0.0) or 0.0)
        equity_low = float(performance.get("equity_low", 0.0) or 0.0)
        base_equity = float(performance.get("base_equity", 0.0) or 0.0)
        equity_peak = base_equity + equity_high
        equity_trough = base_equity + equity_low
        stop_loss_hits = int(performance.get("stop_loss_hits", 0) or 0)
        take_profit_hits = int(performance.get("take_profit_hits", 0) or 0)
        breakeven_hits = int(performance.get("breakeven", 0) or 0)
        manual_closes = int(performance.get("manual_closes", 0) or 0)

        lines.append(
            f"• Closed PnL {perf_net:+.2f} (gross +{gross_profit:.2f} / -{abs(gross_loss):.2f})"
        )
        lines.append(
            f"• Equity peak {equity_peak:.2f} | Equity low {equity_trough:.2f} | Max drawdown {max_dd:.2f}"
        )
        lines.append(
            f"• Stops — TP {take_profit_hits}, SL {stop_loss_hits}, BE {breakeven_hits}"
        )

        partial_realized = float(performance.get("partial_realized", 0.0) or 0.0)
        lines.append(
            f"• Partials {perf_partials} ({partial_realized:+.2f}) | Manual closes {manual_closes}"
        )

        lines.append(
            f"• Win rate {win_rate:.1%} | Profit factor {profit_factor_display} | Sharpe {sharpe:.2f}"
        )

        first_time = performance.get("first_time")
        last_time = performance.get("last_time")
        if isinstance(first_time, datetime) and isinstance(last_time, datetime):
            window_label = (
                f"{first_time.strftime('%Y-%m-%d %H:%M')} → {last_time.strftime('%H:%M')} UTC"
            )
            lines.append(f"• Closed-trade window {window_label}")
    else:
        lines.append("• No closed trades recorded in this session window yet")

    lines.append(format_risk_profile_summary())

    snapshot = "\n".join(lines)
    return truncate_for_telegram(snapshot)


def apply_risk_preset(preset_name: str) -> tuple[bool, str]:
    preset_key = preset_name.lower()
    preset = RISK_PRESETS.get(preset_key)
    if not preset:
        options = ", ".join(sorted(RISK_PRESETS.keys()))
        return False, f"Unknown risk preset '{preset_name}'. Choose one of: {options}."

    settings: dict[str, float] = preset.get("settings", {})  # type: ignore[arg-type]
    previous_mode = risk_profile_state.get("mode")

    for key, value in settings.items():
        globals()[key] = value

    risk_profile_state["mode"] = preset_key
    risk_profile_state["label"] = preset.get("label")
    risk_profile_state["applied_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    risk_profile_state["settings"] = dict(settings)

    account_info = mt5.account_info()
    if account_info:
        evaluate_soft_guard(
            getattr(account_info, "balance", None),
            getattr(account_info, "equity", None),
        )
        evaluate_margin_guard(account_info)

    summary_lines = [
        f"Risk mode set to {preset_key.upper()} — {preset.get('label')}",
        preset.get("description", ""),
        format_risk_profile_summary(line_prefix="• "),
    ]
    summary = "\n".join(line for line in summary_lines if line)
    changed = previous_mode != preset_key
    return changed, summary


def format_telegram_help() -> str:
    return (
        "🤖 Ultima remote commands:\n"
        "/start — welcome message and resume trading if safe\n"
        "/pause [reason] — pause trading\n"
        "/resume — resume trading\n"
        "/performance — current account + guard snapshot\n"
        "/status — quick guard + position overview\n"
        "/guards — detailed guard state\n"
        "/risklist — available risk presets\n"
        "/risk <low|medium|high|status> — adjust risk or show current preset\n"
        "/exposure — symbol + currency exposure view\n"
        "/flatten — close all open positions\n"
        "/scaleout [fraction] — trim every open ticket (default 0.5)\n"
        "/kill [reason] — arm emergency stop (requires confirmation)\n"
        "/confirmkill — confirm kill within 45s to halt trading\n"
        "/cancelkill — cancel a pending kill request\n"
        "/startbot — clear kill state and restart trading\n"
        "/help — show this overview"
    )


def compose_start_response(start_message: str, *, include_help_hint: bool = True) -> str:
    """Build a concise status snapshot suitable for replying to /start."""

    with control_state_lock:
        paused = bool(control_state.get("paused", False))
        pause_reason = str(control_state.get("pause_reason") or "")
        paused_since = control_state.get("paused_since")

    if paused:
        duration_text = ""
        if isinstance(paused_since, datetime):
            duration = datetime.now(timezone.utc) - paused_since
            duration_text = f" for {_format_timedelta_compact(duration)}"
        reason_text = f" — {pause_reason}" if pause_reason else ""
        run_state_line = f"• Run state: Paused{duration_text}{reason_text}"
    else:
        run_state_line = "• Run state: Live"

    account_info = mt5.account_info()
    if account_info:
        equity = float(getattr(account_info, "equity", 0.0) or 0.0)
        balance = float(getattr(account_info, "balance", 0.0) or 0.0)
        profit = float(getattr(account_info, "profit", 0.0) or 0.0)
        account_line = f"• Balance {balance:.2f} | Equity {equity:.2f} | Open PnL {profit:+.2f}"
    else:
        account_line = "• Account info unavailable (mt5.account_info() returned None)"

    detail_lines: list[str] = [run_state_line, account_line]
    detail_lines.extend(format_guard_status_summary(line_prefix="• ").splitlines())
    detail_lines.extend(format_active_positions_summary(line_prefix="• ").splitlines())
    if include_help_hint:
        detail_lines.append("• Commands: /help for the full list of remote controls.")

    detail_block = "\n".join(detail_lines)
    if detail_block:
        composed = f"{start_message}\n\n🤖 Ultima live status:\n{detail_block}"
    else:
        composed = start_message
    return truncate_for_telegram(composed)


def process_telegram_commands(controller: TelegramController | None) -> bool:
    handled = False
    while True:
        try:
            command = TELEGRAM_COMMAND_QUEUE.get_nowait()
        except queue.Empty:
            break

        handled = True
        name = str(command.get("name") or "").lower()
        chat_id = int(command.get("chat_id") or 0)
        user_id = command.get("user_id")
        args = command.get("args") or []
        message_ts = command.get("timestamp") or command.get("date")

        if message_ts is not None:
            try:
                message_dt = datetime.fromtimestamp(int(message_ts), tz=timezone.utc)
            except (TypeError, ValueError, OSError, OverflowError):
                message_dt = None
            if message_dt is not None:
                cutoff = SCRIPT_START_TIME - timedelta(seconds=TELEGRAM_REPLAY_GRACE_SECONDS)
                if message_dt < cutoff:
                    print(
                        f"⏩ Skipping stale Telegram command /{name} from {message_dt.isoformat()} (before current session)."
                    )
                    TELEGRAM_COMMAND_QUEUE.task_done()
                    continue

        _expire_kill_request_if_needed()
        print(f"📨 Telegram command /{name} from {user_id}")

        if chat_id:
            register_telegram_chat(chat_id)
        if user_id is not None:
            with control_state_lock:
                control_state["last_user_id"] = user_id
                control_state["last_source"] = f"telegram:{user_id}"

        reply: str | None = None

        if name == "pause":
            reason = " ".join(args).strip()
            changed, state = set_trading_paused(True, reason=reason, source="telegram", user_id=user_id)
            paused_since = state.get("paused_since")
            duration_note = ""
            if isinstance(paused_since, datetime):
                duration_note = f" (since {paused_since})"
            prefix = "✅" if changed else "ℹ️"
            message = f"{prefix} Trading paused{duration_note}."
            if reason:
                message += f" Reason: {reason}"
            reply = message

        elif name == "resume":
            changed, _ = set_trading_paused(False, source="telegram", user_id=user_id)
            prefix = "▶️" if changed else "ℹ️"
            message = f"{prefix} Trading resumed." if changed else "ℹ️ Trading was already running."
            reply = message

        elif name == "performance":
            snapshot = compose_performance_snapshot()
            reply = snapshot

        elif name == "status":
            lines = [
                "📊 Status snapshot:",
                format_guard_status_summary(line_prefix="• "),
                format_active_positions_summary(line_prefix="• "),
            ]
            reply = truncate_for_telegram("\n".join(lines))

        elif name == "guards":
            reply = truncate_for_telegram(
                "🛡️ Guard status:\n" + format_guard_status_summary(line_prefix="• ")
            )

        elif name == "risklist":
            lines = ["📐 Risk presets:"]
            for key, preset in sorted(RISK_PRESETS.items()):
                settings: dict[str, float] = preset.get("settings", {})  # type: ignore[arg-type]
                risk = float(settings.get("ACCOUNT_RISK_PER_TRADE", ACCOUNT_RISK_PER_TRADE) or 0.0)
                cap = float(settings.get("RISK_MULTIPLIER_MAX", RISK_MULTIPLIER_MAX) or RISK_MULTIPLIER_MAX)
                label = preset.get("label", "")
                description = preset.get("description", "")
                lines.append(
                    f"• {key.upper()}: {label} — risk/trade {risk:.1%}, multiplier cap x{cap:.2f}. {description}"
                )
            reply = truncate_for_telegram("\n".join(lines))

        elif name == "risk":
            if not args or args[0].lower() in {"status", "show"}:
                summary = "📐 Current risk settings:\n" + format_risk_profile_summary(line_prefix="• ")
                reply = summary
            else:
                mode = args[0].lower()
                changed, summary = apply_risk_preset(mode)
                prefix = "✅" if changed else "ℹ️"
                reply = f"{prefix} {summary}"

        elif name == "exposure":
            lines = [
                "💼 Exposure summary:",
                format_exposure_summary(line_prefix="• "),
            ]
            if MARGIN_GUARD_ENABLED:
                lines.append(format_margin_guard_line(line_prefix="• "))
            reply = truncate_for_telegram("\n".join(lines))

        elif name == "flatten":
            if not ensure_connection_ready():
                reply = "⚠️ Unable to reach MT5 terminal; flatten aborted."
            else:
                positions = list(mt5.positions_get() or [])
                volume_positions = [p for p in positions if float(getattr(p, "volume", 0.0) or 0.0) > 0]
                if not volume_positions:
                    reply = "ℹ️ No open positions to flatten."
                else:
                    symbols = sorted({str(getattr(p, "symbol", "")) or "UNKNOWN" for p in volume_positions})
                    closed = 0
                    for symbol in symbols:
                        closed += close_symbol_positions(symbol, reason="Telegram flatten")
                    if closed > 0:
                        post_positions = mt5.positions_get()
                        status_line = format_active_positions_summary(post_positions, line_prefix="• ")
                        reply = (
                            f"🧹 Flattened {closed} ticket{'s' if closed != 1 else ''} across {len(symbols)} symbol"
                            f"{'s' if len(symbols) != 1 else ''}.\n{status_line}"
                        )
                    else:
                        reply = "⚠️ Flatten request executed but no tickets were closed."

        elif name == "scaleout":
            if not ensure_connection_ready():
                reply = "⚠️ Unable to reach MT5 terminal; scale-out aborted."
            else:
                positions = list(mt5.positions_get() or [])
                volume_positions = [p for p in positions if float(getattr(p, "volume", 0.0) or 0.0) > 0]
                if not volume_positions:
                    reply = "ℹ️ No open positions to scale out."
                else:
                    fraction = 0.5
                    if args:
                        try:
                            parsed = float(args[0])
                            if math.isfinite(parsed):
                                fraction = max(0.1, min(0.9, parsed))
                        except ValueError:
                            pass

                    total_closed = 0.0
                    successes = 0
                    failures = 0
                    reason = f"Telegram scale-out {fraction:.0%}"
                    for pos in volume_positions:
                        ticket = int(getattr(pos, "ticket", 0))
                        volume = float(getattr(pos, "volume", 0.0) or 0.0)
                        if ticket <= 0 or volume <= 0:
                            continue
                        close_amount = max(MIN_LOT_SIZE, min(volume, volume * fraction))
                        if close_partial_position(ticket, close_amount, reason=reason):
                            successes += 1
                            total_closed += close_amount
                        else:
                            failures += 1
                    if successes:
                        post_positions = mt5.positions_get()
                        status_line = format_active_positions_summary(post_positions, line_prefix="• ")
                        extra = f"\n• {failures} ticket{'s' if failures != 1 else ''} failed to trim." if failures else ""
                        reply = (
                            f"✂️ Scaled out {total_closed:.2f} lots across {successes} ticket{'s' if successes != 1 else ''}"
                            f" (~{fraction:.0%} per ticket).\n{status_line}{extra}"
                        )
                    else:
                        reply = "⚠️ Scale-out request failed to trim any tickets."

        elif name == "help":
            reply = format_telegram_help()

        elif name == "kill":
            reason = " ".join(args).strip()
            created, state = request_kill_switch(user_id, reason)
            expires_at = state.get("expires_at")
            if isinstance(expires_at, datetime):
                remaining = max(0.0, (expires_at - datetime.now(timezone.utc)).total_seconds())
                countdown = _format_timedelta_compact(timedelta(seconds=remaining))
            else:
                countdown = _format_timedelta_compact(timedelta(seconds=KILL_CONFIRM_WINDOW_SECONDS))
            if state.get("engaged"):
                reply = "🛑 Kill switch already engaged — trading halted."
            else:
                prefix = "🆘" if created else "ℹ️"
                reason_text = f" Reason: {reason}" if reason else ""
                reply = (
                    f"{prefix} Kill switch armed.{reason_text} Confirm within {countdown} using /confirmkill, /cancelkill to abort, then /startbot when ready to resume."
                )

        elif name == "confirmkill":
            success, state = confirm_kill_switch(user_id)
            if success:
                reply = (
                    "🛑 Kill switch engaged. Trading halted; send /startbot when it’s safe to resume."
                )
            else:
                if state.get("pending"):
                    reply = "⚠️ Kill request pending but confirmation expired. Issue /kill again."
                else:
                    reply = "ℹ️ No kill request awaiting confirmation."

        elif name == "cancelkill":
            cancel_kill_switch()
            reply = "✅ Pending kill request cancelled."

        elif name in {"startbot", "start"}:
            resumed, state = resume_kill_switch(user_id)
            if resumed:
                base_reply = "▶️ Kill switch cleared. Trading loop restarting."
            else:
                if state.get("pending"):
                    base_reply = "⚠️ Kill request pending confirmation. Use /confirmkill or /cancelkill first."
                else:
                    changed, _ = set_trading_paused(
                        False,
                        reason="Start command",
                        source="telegram",
                        user_id=user_id,
                    )
                    base_reply = "▶️ Trading resumed." if changed else "ℹ️ Trading already running."

            if name == "start" and not state.get("pending"):
                reply = compose_start_response(base_reply)
            else:
                reply = base_reply

        else:
            unknown = command.get("command") or name
            reply = f"❓ Unknown command '/{unknown}'. Try /help for available commands."

        if reply and controller and chat_id:
            controller.send_message(chat_id, truncate_for_telegram(reply))

        TELEGRAM_COMMAND_QUEUE.task_done()

    return handled


def wait_if_paused(controller: TelegramController | None) -> bool:
    entered_pause = False
    while True:
        with control_state_lock:
            paused = bool(control_state.get("paused", False))
            pause_notified = bool(control_state.get("pause_notified", False))
            pause_reason = str(control_state.get("pause_reason") or "")
            paused_since = control_state.get("paused_since")

        if not paused:
            if entered_pause:
                print("▶️ Trading resume signal received; resuming live loop.")
            with control_state_lock:
                control_state["pause_notified"] = False
            return entered_pause

        entered_pause = True

        if not pause_notified:
            duration_text = ""
            if isinstance(paused_since, datetime):
                duration = datetime.now(timezone.utc) - paused_since
                duration_text = f" (paused for {_format_timedelta_compact(duration)})"
            reason_text = f" — {pause_reason}" if pause_reason else ""
            print(f"⏸️ Trading paused via remote command{duration_text}{reason_text}.")
            with control_state_lock:
                control_state["pause_notified"] = True

        process_telegram_commands(controller)
        enforce_kill_switch()
        time.sleep(max(TELEGRAM_POLL_SECONDS, TELEGRAM_PAUSE_SLEEP_SECONDS))


def wait_for_kill_clear(controller: TelegramController | None) -> None:
    notified = False
    while True:
        _expire_kill_request_if_needed()
        process_telegram_commands(controller)
        suppress_resume_notify = False
        with kill_state_lock:
            engaged = bool(kill_switch_state.get("engaged"))
            suppress_resume_notify = bool(kill_switch_state.get("suppress_resume_notify", False))
            if not engaged and suppress_resume_notify:
                kill_switch_state["suppress_resume_notify"] = False
        if not engaged:
            if notified:
                print("▶️ Kill switch cleared — restarting trading loop.")
                if not suppress_resume_notify:
                    telegram_notify("▶️ Kill switch cleared. Trading loop restarting.")
            return
        if not notified:
            print("🔒 Kill switch engaged — trading halted. Awaiting /startbot to resume.")
            telegram_notify("🛑 Kill switch engaged. Send /startbot when ready to restart trading.")
            notified = True
        time.sleep(max(TELEGRAM_POLL_SECONDS, TELEGRAM_PAUSE_SLEEP_SECONDS))


def enforce_kill_switch_safe(controller: TelegramController | None, scan_counter: int) -> bool:
    try:
        enforce_kill_switch()
        return True
    except EmergencyStop:
        log_manual_stop_summary(scan_counter, trigger="Kill switch engaged")
        wait_for_kill_clear(controller)
        return False


def apply_risk_guard_to_multiplier(multiplier: float) -> float:
    now = datetime.now(timezone.utc)

    guard_factor = risk_guard_drawdown_factor() if RISK_GUARD_ENABLED else 1.0
    soft_factor = float(soft_guard_state.get("throttle", 1.0) or 1.0) if SOFT_GUARD_ENABLED else 1.0
    margin_factor = float(margin_guard_state.get("throttle", 1.0) or 1.0) if MARGIN_GUARD_ENABLED else 1.0
    combined_factor = guard_factor * soft_factor
    combined_factor *= margin_factor

    adjusted = multiplier * combined_factor

    min_bound = RISK_MULTIPLIER_MIN
    if RISK_GUARD_ENABLED and guard_factor < 1.0:
        min_bound = min(min_bound, RISK_RECOVERY_MIN_FACTOR)
    if SOFT_GUARD_ENABLED and soft_factor < 1.0:
        min_bound = min(min_bound, max(SOFT_GUARD_MIN_THROTTLE, soft_factor))
    if MARGIN_GUARD_ENABLED and margin_factor < 1.0:
        min_bound = min(min_bound, max(MARGIN_GUARD_MIN_THROTTLE, margin_factor))

    recovery_until = None
    try:
        recovery_until = risk_multiplier_state.get("recovery_until")
        if isinstance(recovery_until, str):
            recovery_until = datetime.fromisoformat(recovery_until)
    except Exception:
        recovery_until = None

    is_recovering = bool(recovery_until and isinstance(recovery_until, datetime) and now <= recovery_until)
    if guard_factor < 1.0 or soft_factor < 1.0 or margin_factor < 1.0:
        risk_multiplier_state["last_violation"] = now.isoformat()
        recovery_until = now + RISK_MULTIPLIER_RECOVERY_WINDOW
        risk_multiplier_state["recovery_until"] = recovery_until.isoformat()
        risk_multiplier_state["recovery_cap"] = RISK_MULTIPLIER_RELIEF_MAX
        is_recovering = True

    if not is_recovering:
        last_violation = risk_multiplier_state.get("last_violation")
        if isinstance(last_violation, str):
            try:
                last_violation_dt = datetime.fromisoformat(last_violation)
                if now - last_violation_dt > RISK_MULTIPLIER_RECOVERY_WINDOW:
                    risk_multiplier_state["recovery_cap"] = RISK_MULTIPLIER_MAX
                    risk_multiplier_state["recovery_until"] = None
            except ValueError:
                risk_multiplier_state["recovery_cap"] = RISK_MULTIPLIER_MAX
                risk_multiplier_state["recovery_until"] = None

    recovery_cap = float(risk_multiplier_state.get("recovery_cap", RISK_MULTIPLIER_MAX) or RISK_MULTIPLIER_MAX)

    target = min(RISK_MULTIPLIER_MAX, recovery_cap)
    if is_recovering and multiplier <= RISK_MULTIPLIER_RECOVERY_TARGET:
        target = min(target, RISK_MULTIPLIER_RECOVERY_TARGET)

    adjusted = max(min_bound, min(target, adjusted))
    return round(adjusted, 2)


def monitor_closed_positions():
    """Monitor recently closed positions and update strategy performance."""
    try:
        # Get deals from last 24 hours
        from_date = datetime.now() - timedelta(days=1)
        to_date = datetime.now()
        
        deals = mt5.history_deals_get(from_date, to_date)
        if not deals:
            return
        
        # Track processed deals to avoid double counting
        processed_deals_file = "processed_deals.txt"
        try:
            with open(processed_deals_file, 'r') as f:
                processed_deals = set(line.strip() for line in f)
        except FileNotFoundError:
            processed_deals = set()
        
        new_processed = []
        
        for deal in deals:
            deal_id = str(deal.ticket)
            if deal_id in processed_deals:
                continue
            
            # Only process exit deals (position close)
            if deal.entry != 1:  # 1 = entry, 0 = exit
                continue
            
            symbol = deal.symbol
            profit = deal.profit
            comment = (deal.comment or "").strip()
            performance_key = None

            # New compact comment format: CODE + side + priority (e.g., MTB16)
            match = re.match(r"^([A-Z]{1,3})([BS])(\d{2})", comment)
            if match:
                code = match.group(1)
                performance_key = COMMENT_CODE_CACHE.get(code)
            elif " P" in comment:
                # Legacy format: "Strategy Name P123"
                strategy_name = comment.split(" P")[0].replace('-', ' ').strip()
                performance_key = strategy_name.lower().replace(' ', '_')

            if performance_key and symbol:
                update_strategy_performance(performance_key, symbol, profit)
                new_processed.append(deal_id)
        
        # Save processed deals
        if new_processed:
            with open(processed_deals_file, 'a') as f:
                for deal_id in new_processed:
                    f.write(f"{deal_id}\n")
                    
    except Exception as e:
        print(f"Error monitoring closed positions: {e}")


def get_strategy_performance_weight(strategy_name: str, symbol: str) -> float:
    """Get performance-based weight multiplier for strategy."""
    key = f"{strategy_name}_{symbol}"
    perf = strategy_performance[key]
    
    if perf['total_trades'] < MIN_TRADES_FOR_ADAPTATION:
        return 1.0  # Use default weight until enough data
    
    # Base weight on profit factor and win rate
    profit_factor = perf.get('profit_factor', 0.0)
    win_rate = perf.get('win_rate', 0.0)
    
    # Conservative adaptation: good performance gets slight boost, poor performance gets reduced
    if profit_factor > 0.1 and win_rate > 0.4:
        return min(1.5, 1.0 + (profit_factor * 0.5) + (win_rate * 0.3))  # Max 1.5x boost
    elif profit_factor < -0.05 or win_rate < 0.3:
        return max(0.3, 1.0 + profit_factor)  # Reduce to min 0.3x
    else:
        return 1.0  # Neutral performance

# Settings
WEEKDAY_SYMBOLS: list[str] = ["EURUSD.i", "USDJPY.i", "GBPUSD.i", "GBPJPY.i", "XAUUSD.i"]
WEEKEND_SYMBOLS: list[str] = ["BTCUSD"]
ALL_SYMBOLS: tuple[str, ...] = tuple(dict.fromkeys(WEEKDAY_SYMBOLS + WEEKEND_SYMBOLS))
timeframe = mt5.TIMEFRAME_M15
bars = 100  # Number of bars to fetch

# Risk/target settings (ATR-based) - OPTIMIZED FROM BACKTESTING
ATR_PERIOD = 5  # Base period from latest optimization (strategies currently share this value)
SL_ATR_MULTIPLIER = 2.0  # Default fallback; individual strategies can override
TP_ATR_MULTIPLIER = 2.0  # Default fallback; individual strategies can override
RESPECT_STOPS_LEVEL = True  # Enforce broker minimal stop distance if provided
ALLOW_HEDGING = False  # When False, wait for existing positions to close before taking opposite trades

# Counter-signal exit tuning (avoid sitting in drawdown when strong opposite signal persists)
COUNTER_SIGNAL_EXIT_ENABLED = True
COUNTER_SIGNAL_EXIT_MIN_CONFIDENCE = 0.7
COUNTER_SIGNAL_EXIT_MIN_ATR_LOSS = 0.9
COUNTER_SIGNAL_EXIT_MIN_STREAK = 2
COUNTER_SIGNAL_EXIT_RESET_SCANS = 6
COUNTER_SIGNAL_EXIT_REQUIRE_MICRO = True
COUNTER_SIGNAL_EXIT_MIN_NET_LOSS = -1.0  # Require at least -$1 unrealized on the conflicted leg
COUNTER_SIGNAL_EXIT_COMMENT = "CounterExit"

# Enhanced risk management
ACCOUNT_RISK_PER_TRADE = 0.06  # Balanced growth profile default risk per trade
MIN_LOT_SIZE = 0.01  # Minimum position size
MAX_LOT_SIZE = 5.0   # Maximum position size cap

# Broker-specific cost assumptions (Vantage RAW)
BROKER_COMMISSION_PER_LOT = 6.0  # Round-turn commission per 1.0 lot in account currency
SYMBOL_COMMISSION_OVERRIDES: dict[str, float] = {
    # e.g. "XAUUSD+": 12.0,  # Override if the broker charges different commission for metals
    "BTCUSD": 0.0,
}

# Broker execution preferences (Vantage RAW)
BROKER_PREFERRED_FILLINGS: tuple[int, ...] = tuple(
    filter(
        None,
        (
            getattr(mt5, "ORDER_FILLING_IOC", None),
            getattr(mt5, "ORDER_FILLING_FOK", None),
            getattr(mt5, "ORDER_FILLING_RETURN", None),
        ),
    )
)
BROKER_ORDER_DEVIATION = 8  # Allowed price deviation in points for raw-spread fills
LOG_SPREAD_TELEMETRY = True

# Weekend crypto trading configuration
ENABLE_WEEKEND_CRYPTO = True
CRYPTO_WEEKEND_SESSION_NAME = "Crypto Weekend"

# Dynamic risk multiplier bounds
RISK_MULTIPLIER_MIN = 1.0
RISK_MULTIPLIER_MAX = 2.8
RISK_MULTIPLIER_RELIEF_MAX = 2.2       # Ceiling while recovering from drawdown/guard throttles
RISK_MULTIPLIER_RECOVERY_WINDOW = timedelta(hours=6)
RISK_MULTIPLIER_RECOVERY_TARGET = 1.15

# Liquidity-aware spread gating (spread-to-ATR heuristics)
LIQUIDITY_SPREAD_ATR_BONUS = 0.18      # Tight spreads (<18% of ATR) earn a risk boost
LIQUIDITY_TIGHT_BONUS_MAG = 0.35       # Maximum proportional boost applied for ultra-tight spreads
LIQUIDITY_MAX_BONUS = 1.25             # Hard ceiling on liquidity-based boost factor
LIQUIDITY_SPREAD_ATR_SOFT_CAP = 0.32   # Begin throttling once spread/ATR exceeds 32%
LIQUIDITY_SPREAD_ATR_HARD_CAP = 0.55   # Block new entries once spread/ATR crosses 55%
LIQUIDITY_MAX_PENALTY = 0.55           # Maximum proportional reduction applied before blocking
LIQUIDITY_MIN_SCALE = 0.45             # Never scale risk below 45% unless guard blocks entirely

# Alpha surge (short-lived opportunity) detection
ALPHA_SURGE_ENABLED = True
ALPHA_SURGE_MIN_SESSION_PRIORITY = 4
ALPHA_SURGE_MIN_VOL_PRESSURE = 0.65
ALPHA_SURGE_MIN_CANDLE_CONVICTION = 0.60
ALPHA_SURGE_MAX_SPREAD_RATIO = 0.25
ALPHA_SURGE_RISK_MULTIPLIER = 1.18
ALPHA_SURGE_CONFIDENCE_RELIEF = 0.05
ALPHA_SURGE_MIN_THRESHOLD = 0.55

# Scan cadence configuration
SCAN_INTERVAL_SECONDS = 60        # Time between scan loops (seconds)

# Soft drawdown guard (still aggressive but avoids death spirals)
SOFT_GUARD_ENABLED = True
SOFT_GUARD_LIMIT = 0.28           # Block new trades if unrealized DD exceeds 28% of balance
SOFT_GUARD_ALERT = 0.23           # Strongly throttle between 23%-28%
SOFT_GUARD_CAUTION = 0.16         # Begin soft throttling after 16%
SOFT_GUARD_RESUME = 0.12          # Resume trading once DD recovers below 12%
SOFT_GUARD_MIN_THROTTLE = 0.45    # Never risk below 45% sizing unless fully blocked

# Margin usage guard (prevent over-leveraging during stack builds)
MARGIN_GUARD_ENABLED = True
MARGIN_USAGE_THROTTLE = 0.52      # Begin throttling once >52% of equity is tied up in margin
MARGIN_USAGE_BLOCK = 0.70         # Block new entries once margin usage crosses 70%
MARGIN_FREE_RATIO_THROTTLE = 0.40 # Throttle if free margin drops below 40% of equity
MARGIN_FREE_RATIO_BLOCK = 0.24    # Block new entries if free margin <24% of equity
MARGIN_GUARD_MIN_THROTTLE = 0.40  # Lowest risk factor applied before blocking
MARGIN_USAGE_RESUME = 0.60        # Usage must fall below 60% before resuming after a block
MARGIN_FREE_RATIO_RESUME = 0.36   # Free margin must recover above 36% of equity to unblock
MARGIN_USAGE_THROTTLE_RESUME = 0.48  # Usage must stay below 48% before clearing throttle state
MARGIN_FREE_RATIO_THROTTLE_RESUME = 0.44  # Free margin resume threshold for throttle recovery
MARGIN_GUARD_SEVERITY_EXPONENT = 1.35  # Curve severity to bite harder as usage worsens
MARGIN_ENTRY_GUARD_MIN_FACTOR = 0.55  # Do not open fresh exposure when guard factor falls below 55%
PYRAMID_GUARD_MIN_FACTOR = 0.65       # Require at least 65% guard factor before stacking positions
MARGIN_ALERT_SEVERITY_THRESHOLD = 0.55  # Alert Telegram once severity crosses 55%
MARGIN_ALERT_REPEAT_BUCKET = 0.10       # Re-alert every +10% severity when throttled

# Exposure caps (per-symbol lots + per-currency notional)
EXPOSURE_CAPS_ENABLED = True
EXPOSURE_SYMBOL_DEFAULT_MAX_TOTAL = 1.8        # Hard ceiling of aggregate lots per symbol
EXPOSURE_SYMBOL_DEFAULT_MAX_SAME = 1.2         # Max same-direction lots before forcing partials
EXPOSURE_SYMBOL_OVERRIDES: dict[str, dict[str, float]] = {
    "XAUUSD": {"max_total": 1.2, "max_same": 1.0},
    "GBPJPY": {"max_total": 1.3, "max_same": 0.9},
    "BTCUSD": {"max_total": 0.6, "max_same": 0.4},
}
EXPOSURE_SYMBOL_MIN_REMAINDER = 0.02           # Require at least this lot space to open another trade
EXPOSURE_CURRENCY_EQUITY_MULTIPLIER = 15.0     # Allow up to 15× equity notional per currency
EXPOSURE_CURRENCY_OVERRIDES: dict[str, float] = {
    "JPY": 12.0,
    "XAU": 8.0,
    "BTC": 4.0,
}
EXPOSURE_CURRENCY_WARN_RATIO = 0.85            # Emit warning once >85% of a currency band is consumed

# Automated margin unwind safeguards
MARGIN_UNWIND_ENABLED = True
MARGIN_UNWIND_TRIGGER_THROTTLE = 0.72       # Start shedding exposure once guard throttles below 72%
MARGIN_UNWIND_TRIGGER_SEVERITY = 0.45       # Or when severity climbs above 45%
MARGIN_UNWIND_BLOCK_REDUCTION = 0.55        # Close at least 55% of lots when margin guard blocks entries
MARGIN_UNWIND_MIN_REDUCTION = 0.18          # Minimum fraction of lots to trim during a throttle event
MARGIN_UNWIND_MAX_REDUCTION = 0.70          # Never close more than 70% in one pass (avoid full flatten)
MARGIN_UNWIND_MIN_VOLUME = MIN_LOT_SIZE     # Guarantee we trim at least the broker minimum lot size
MARGIN_UNWIND_COOLDOWN_SECONDS = 120        # Wait two minutes before attempting another unwind pass

# Order retry tuning for margin shortfalls
ORDER_NO_MONEY_RETRY_ENABLED = True
ORDER_NO_MONEY_MAX_RETRIES = 2
ORDER_NO_MONEY_REDUCTION = 0.7     # Reduce lot size by 30% on each retry

# Drawdown-aware micro confirmation tuning
DRAWDOWN_RELAXATION_PER_ATR = 0.25   # Relax micro alignment 25% per ATR of drawdown
DRAWDOWN_OVERRIDE_ATR = 1.2          # Allow fallback overrides when drawdown exceeds 1.2 ATR

# Aggressive mode tuning
AGGRESSIVE_MODE = True
AGGRESSIVE_REGIME_THRESHOLD = 0.05  # Minimal regime weight still allowed when aggressive
AGGRESSIVE_MICRO_TOLERANCE = 0.25   # Allow counter-momentum within 25% of threshold
AGGRESSIVE_MICRO_MIN_RATIO = 0.25   # Require at least 25% of base threshold in the trade direction
HIGH_CONFIDENCE_BOOST_CAP = 1.5     # Max boost applied when confidence is exceptional

# Pyramiding (position stacking) configuration
PYRAMID_ENABLED = True
PYRAMID_MAX_ENTRIES_PER_SIDE = 3          # Total entries (initial + adds) allowed per direction
PYRAMID_MIN_CONFIDENCE = 0.9              # Require strong conviction to stack
PYRAMID_MAX_LOSING_POSITIONS = 3          # Allow deeper averaging while managing size
PYRAMID_WINNING_RISK_BOOST = 1.2          # Risk multiplier boost when scaling into profitable legs
PYRAMID_LOSING_RISK_BOOST = 1.0           # Neutral sizing when averaging into drawdown

# Mean reversion stack safety
MEAN_REVERSION_MAX_LOSING_POSITIONS = 2
MEAN_REVERSION_STACK_DRAW_LIMIT_ATR = 1.6
MEAN_REVERSION_STACK_HARD_CAP_ATR = 2.3
MEAN_REVERSION_STACK_RESCUE_CONFIDENCE = 1.25
MEAN_REVERSION_STACK_RESCUE_SCALE = 0.6

# Trade quality filters
SPREAD_POINTS_LIMIT = 10            # Maximum raw spread in points before skipping
SPREAD_ATR_RATIO_LIMIT = 0.45       # Spread must remain under 45% of current ATR

# Micro timeframe momentum confirmation (sniper entry enhancer)
ENABLE_MICRO_MOMENTUM_CONFIRMATION = True
MICRO_MOMENTUM_TIMEFRAME = mt5.TIMEFRAME_M5
MICRO_MOMENTUM_LOOKBACK = 6
MICRO_MOMENTUM_BASE_THRESHOLD = 0.0006
MICRO_MOMENTUM_MIN_THRESHOLD = 0.0002
MICRO_MOMENTUM_MAX_THRESHOLD = 0.0025
MICRO_MOMENTUM_DYNAMIC_MULTIPLIER = 0.55
MICRO_MOMENTUM_SOFT_PASS_RATIO = 0.45
MICRO_MOMENTUM_NEAR_MISS_RATIO = 0.8
MICRO_MOMENTUM_GUARD_MIN = 0.85
MICRO_MOMENTUM_NEAR_MISS_MAX_DD = 0.8

# Signal confidence gating
CONFIDENCE_EXECUTION_THRESHOLD = 0.65  # Slightly looser gate for higher trade frequency


RISK_PRESETS: dict[str, dict[str, object]] = {
    "low": {
        "label": "Capital preservation",
        "description": "Dial risk back sharply to ride out drawdowns or when trading unattended.",
        "settings": {
            "ACCOUNT_RISK_PER_TRADE": 0.04,
            "RISK_MULTIPLIER_MAX": 2.0,
            "SOFT_GUARD_LIMIT": 0.22,
            "SOFT_GUARD_ALERT": 0.18,
            "SOFT_GUARD_CAUTION": 0.12,
            "SOFT_GUARD_RESUME": 0.08,
            "SOFT_GUARD_MIN_THROTTLE": 0.50,
            "MARGIN_USAGE_THROTTLE": 0.45,
            "MARGIN_USAGE_BLOCK": 0.62,
            "MARGIN_USAGE_RESUME": 0.54,
            "MARGIN_USAGE_THROTTLE_RESUME": 0.40,
            "MARGIN_FREE_RATIO_THROTTLE": 0.48,
            "MARGIN_FREE_RATIO_BLOCK": 0.32,
            "MARGIN_FREE_RATIO_RESUME": 0.42,
            "MARGIN_FREE_RATIO_THROTTLE_RESUME": 0.50,
            "MARGIN_GUARD_MIN_THROTTLE": 0.45,
        },
    },
    "medium": {
        "label": "Balanced growth",
        "description": "Default profile tuned for live trading with strong guardrails.",
        "settings": {
            "ACCOUNT_RISK_PER_TRADE": 0.06,
            "RISK_MULTIPLIER_MAX": 2.8,
            "SOFT_GUARD_LIMIT": 0.28,
            "SOFT_GUARD_ALERT": 0.23,
            "SOFT_GUARD_CAUTION": 0.16,
            "SOFT_GUARD_RESUME": 0.12,
            "SOFT_GUARD_MIN_THROTTLE": 0.45,
            "MARGIN_USAGE_THROTTLE": 0.52,
            "MARGIN_USAGE_BLOCK": 0.70,
            "MARGIN_USAGE_RESUME": 0.60,
            "MARGIN_USAGE_THROTTLE_RESUME": 0.48,
            "MARGIN_FREE_RATIO_THROTTLE": 0.40,
            "MARGIN_FREE_RATIO_BLOCK": 0.24,
            "MARGIN_FREE_RATIO_RESUME": 0.36,
            "MARGIN_FREE_RATIO_THROTTLE_RESUME": 0.44,
            "MARGIN_GUARD_MIN_THROTTLE": 0.40,
        },
    },
    "high": {
        "label": "Aggressive compounding",
        "description": "Open the throttle when market conditions are exceptional and you can monitor closely.",
        "settings": {
            "ACCOUNT_RISK_PER_TRADE": 0.09,
            "RISK_MULTIPLIER_MAX": 3.4,
            "SOFT_GUARD_LIMIT": 0.33,
            "SOFT_GUARD_ALERT": 0.28,
            "SOFT_GUARD_CAUTION": 0.20,
            "SOFT_GUARD_RESUME": 0.16,
            "SOFT_GUARD_MIN_THROTTLE": 0.42,
            "MARGIN_USAGE_THROTTLE": 0.58,
            "MARGIN_USAGE_BLOCK": 0.74,
            "MARGIN_USAGE_RESUME": 0.66,
            "MARGIN_USAGE_THROTTLE_RESUME": 0.52,
            "MARGIN_FREE_RATIO_THROTTLE": 0.36,
            "MARGIN_FREE_RATIO_BLOCK": 0.22,
            "MARGIN_FREE_RATIO_RESUME": 0.32,
            "MARGIN_FREE_RATIO_THROTTLE_RESUME": 0.40,
            "MARGIN_GUARD_MIN_THROTTLE": 0.38,
        },
    },
}

risk_profile_state["settings"] = dict(RISK_PRESETS["medium"]["settings"])  # type: ignore[index]
risk_profile_state["applied_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

risk_multiplier_state: dict[str, object] = {
    "last_violation": None,
    "recovery_until": None,
    "recovery_cap": RISK_MULTIPLIER_RELIEF_MAX,
}


def get_account_balance() -> float:
    """Get the current account balance from MT5."""
    try:
        account_info = mt5.account_info()
        if account_info is None:
            print("Failed to get account info, using fallback balance of $1000")
            return 1000.0
        
        equity = getattr(account_info, 'equity', None)
        balance = float(account_info.balance)
        effective_balance = float(equity) if equity is not None else balance
        print(f"Current account equity: ${effective_balance:.2f} (balance: ${balance:.2f})")
        return effective_balance
    except Exception as e:
        print(f"Error getting account balance: {e}, using fallback balance of $1000")
        return 1000.0


def get_account_equity_quiet() -> float:
    """Fetch account equity without emitting log noise."""
    try:
        account_info = mt5.account_info()
        if account_info is None:
            return 0.0
        equity_value = getattr(account_info, "equity", None)
        if equity_value is None:
            return float(account_info.balance)
        return float(equity_value)
    except Exception as exc:
        print(f"⚠️  Failed to fetch account equity: {exc}")
        return 0.0


def is_market_session_active(active_symbols: list[str] | None = None) -> bool:
    """Check if we're in an active trading session (avoid low-liquidity periods)."""
    try:
        # Get current UTC time
        now_utc = datetime.now(pytz.UTC)
        current_hour = now_utc.hour
        current_weekday = now_utc.weekday()  # 0=Monday, 6=Sunday

        if is_weekend_trading_window(now_utc):
            if not ENABLE_WEEKEND_CRYPTO:
                return False
            if not active_symbols:
                return False
            crypto_only = all(symbol in WEEKEND_SYMBOLS for symbol in active_symbols)
            return crypto_only
        
        # Active trading sessions (UTC times):
        # Sydney: 22:00-07:00 UTC
        # Tokyo: 00:00-09:00 UTC  
        # London: 08:00-17:00 UTC
        # New York: 13:00-22:00 UTC
        
        # Avoid dead zones: 07:00-08:00 UTC (between Sydney/Tokyo and London)
        if 7 <= current_hour < 8:
            return False
        
        # Most active periods (overlaps):
        # London-New York overlap: 13:00-17:00 UTC (Best)
        # Tokyo-London overlap: 08:00-09:00 UTC (Good)
        # Always active: 22:00-07:00, 08:00-22:00 (avoid only 07:00-08:00)
        
        return True
        
    except Exception as e:
        print(f"Error checking market session: {e}, defaulting to allow trading")
        return True


def get_session_info() -> str:
    """Get current trading session name for logging."""
    try:
        now_utc = datetime.now(pytz.UTC)
        hour = now_utc.hour
        if ENABLE_WEEKEND_CRYPTO and is_weekend_trading_window(now_utc):
            return CRYPTO_WEEKEND_SESSION_NAME
        
        if 22 <= hour or hour < 7:
            return "Sydney/Tokyo"
        elif 8 <= hour < 13:
            return "London"
        elif 13 <= hour < 17:
            return "London-NY Overlap"
        elif 17 <= hour < 22:
            return "New York"
        else:
            return "Dead Zone"
    except Exception:
        return "Unknown"


def is_weekend_trading_window(now_utc: datetime | None = None) -> bool:
    if now_utc is None:
        now_utc = datetime.now(pytz.UTC)
    weekday = now_utc.weekday()
    hour = now_utc.hour
    if weekday == 5:
        return True
    if weekday == 6:
        return hour < 22
    if weekday == 4:
        return hour >= 22
    return False


def get_active_symbols(now_utc: datetime | None = None) -> list[str]:
    if ENABLE_WEEKEND_CRYPTO and is_weekend_trading_window(now_utc):
        return WEEKEND_SYMBOLS.copy()
    return WEEKDAY_SYMBOLS.copy()


def log_cycle_summary(active_symbols: list[str], cycle_stats: dict[str, int], scan_counter: int) -> None:
    total_symbols = len(active_symbols)
    print(
        f"📒 Scan #{scan_counter} summary -> "
        f"symbols processed {cycle_stats['symbols_total']}/{total_symbols} | "
        f"open-symbols {cycle_stats['symbols_with_open_positions']} | "
        f"candidates {cycle_stats['candidates_total']} | executed {cycle_stats['signals_executed']} | pyramids {cycle_stats['signals_executed_pyramid']} | "
        f"confidence-filtered {cycle_stats['signals_filtered_confidence']} | micro-filtered {cycle_stats['signals_filtered_micro']} | "
        f"position-blocked {cycle_stats['signals_skipped_position_conflict']} | duplicates {cycle_stats['signals_skipped_duplicate_side']} | guard-blocked {cycle_stats['signals_skipped_guard']} | "
        f"counter-exits {cycle_stats['counter_signal_exits']}"
    )
    print(
        "🛠️ Stop management -> "
        f"breakeven {cycle_stats['stops_breakeven']} | trailing {cycle_stats['stops_trailing']} | "
        f"partials {cycle_stats['partials_taken']} | partial-missed {cycle_stats['partials_unavailable']} | "
        f"adjustment-fails {cycle_stats['stops_failed'] + cycle_stats['partials_failed']}"
    )
    print(
        "📊 Strategy diagnostics -> "
        f"disabled {cycle_stats['strategies_disabled_regime']} | no-signal {cycle_stats['strategies_no_signal']} | "
        f"ATR-missing {cycle_stats['strategies_skipped_atr']} | micro-overrides {cycle_stats['micro_overrides']} | micro-confirms {cycle_stats['micro_confirms']}"
    )
    print(
        "💧 Liquidity gating -> "
        f"boosts {cycle_stats['liquidity_boosted']} | throttles {cycle_stats['liquidity_scaled']} | blocks {cycle_stats['liquidity_skipped']}"
    )
    print(
        "⚡ Alpha surge -> "
        f"detected {cycle_stats['alpha_surge_active']} | scaled {cycle_stats['alpha_surge_scaled']}"
    )
    print(f"⏱️ Next scan in {SCAN_INTERVAL_SECONDS} seconds...")


# Instrument-specific session priorities (based on underlying market activity)
INSTRUMENT_SESSION_PRIORITY = {
    "EURUSD.i": {
        "London": 5,           # Best: EUR/USD most active during European session
        "London-NY Overlap": 5, # Best: Highest liquidity period for majors
        "New York": 4,         # Good: Still active during US session
        "Sydney/Tokyo": 2,     # Poor: Low EUR activity during Asian session
    },
    "GBPUSD.i": {
        "London": 5,           # Best: GBP home session, highest volatility
        "London-NY Overlap": 5, # Best: Peak liquidity for GBP/USD
        "New York": 4,         # Good: Continued activity in US session
        "Sydney/Tokyo": 1,     # Very poor: GBP barely trades in Asia
    },
    "USDJPY.i": {
        "Sydney/Tokyo": 5,     # Best: JPY home session, highest volatility
        "London": 4,           # Good: European traders active
        "London-NY Overlap": 4, # Good: Still decent liquidity
        "New York": 3,         # Fair: End of USD day, lower JPY activity
    },
    "GBPJPY.i": {
        "Sydney/Tokyo": 4,     # Good: JPY side active
        "London": 5,           # Best: GBP side most active + some JPY overlap
        "London-NY Overlap": 4, # Good: High volatility cross
        "New York": 2,         # Poor: Both sides winding down
    },
    "XAUUSD.i": {
        "London": 5,           # Best: Major gold trading center (London)
        "London-NY Overlap": 5, # Best: Peak institutional activity
        "New York": 4,         # Good: US market participation
        "Sydney/Tokyo": 3,     # Fair: Asian demand, but lower liquidity
    },
    "BTCUSD": {
        CRYPTO_WEEKEND_SESSION_NAME: 5,  # Dedicated weekend session
        "London": 1,                   # Deprioritize forex sessions during week
        "London-NY Overlap": 1,
        "New York": 1,
        "Sydney/Tokyo": 2,             # Overnight liquidity acceptable if needed
    }
}

# Minimum session priority to allow trading (1-5 scale)
MIN_SESSION_PRIORITY = 3  # Only trade when session priority >= 3

# Correlation management - prevent over-concentration in similar moves
CORRELATION_GROUPS = {
    "EUR_PAIRS": ["EURUSD.i"],                    # EUR strength/weakness
    "GBP_PAIRS": ["GBPUSD.i", "GBPJPY.i"],        # GBP strength/weakness
    "JPY_PAIRS": ["USDJPY.i", "GBPJPY.i"],        # JPY strength/weakness
    "USD_MAJORS": ["EURUSD.i", "GBPUSD.i", "USDJPY.i"], # USD strength/weakness
    "SAFE_HAVEN": ["USDJPY.i", "XAUUSD.i"],       # Risk-off correlation
}

DEFAULT_CORRELATION_LIMITS = {
    "max_same_direction": 2,
    "max_total": 3,
    "size_multiplier": 0.6,
    "max_same_direction_volume": 0.0,
    "volume_relief_scale": 0.55,
}

CORRELATION_GROUP_LIMITS = {
    "EUR_PAIRS": {"max_same_direction": 3, "max_total": 4, "size_multiplier": 0.65, "max_same_direction_volume": 0.24, "volume_relief_scale": 0.6},
    "GBP_PAIRS": {"max_same_direction": 3, "max_total": 4, "size_multiplier": 0.6, "max_same_direction_volume": 0.26, "volume_relief_scale": 0.6},
    "JPY_PAIRS": {"max_same_direction": 2, "max_total": 3, "size_multiplier": 0.5, "max_same_direction_volume": 0.18, "volume_relief_scale": 0.55},
    "USD_MAJORS": {"max_same_direction": 3, "max_total": 4, "size_multiplier": 0.65, "max_same_direction_volume": 0.28, "volume_relief_scale": 0.6},
    "SAFE_HAVEN": {"max_same_direction": 2, "max_total": 2, "size_multiplier": 0.5, "max_same_direction_volume": 0.12, "volume_relief_scale": 0.5},
}

# Adaptive correlation guard tuning
CORRELATION_HIGH_CONFIDENCE_THRESHOLD = 1.25
CORRELATION_GUARD_SUPPORT_THRESHOLD = 0.8
CORRELATION_OVERRIDE_SCALE = 0.65
CORRELATION_OVERRIDE_MIN_SCALE = 0.45
CORRELATION_HEDGE_RELIEF_SCALE = 0.8
CORRELATION_HEDGE_BIAS_SLACK = 1
CORRELATION_DRAWDOWN_TIGHTEN_ATR = 1.2
CORRELATION_DRAWDOWN_CUTOFF_ATR = 1.8

# Market Regime Detection - Dynamic strategy adaptation
REGIME_LOOKBACK_PERIODS = 50  # Bars to analyze for regime detection
TREND_THRESHOLD = 0.3         # ADX threshold for trending markets
VOLATILITY_THRESHOLD = 1.5    # ATR multiplier for volatile markets

# Strategy weights by market regime (sum should = 1.0 for each regime)
REGIME_STRATEGY_WEIGHTS = {
    'TRENDING': {
        'ma_crossover': 0.35,      # Best in trends
        'momentum_trend': 0.30,    # Excellent trend following
        'breakout': 0.25,          # Good for trend continuation
        'mean_reversion': 0.05,    # Poor in strong trends
        'donchian_channel': 0.05,  # Poor in strong trends
    },
    'RANGING': {
        'mean_reversion': 0.40,    # Excellent in sideways markets
        'donchian_channel': 0.25,  # Good for range trading
        'ma_crossover': 0.15,      # Moderate in ranges
        'momentum_trend': 0.10,    # Poor in ranges
        'breakout': 0.10,          # Poor in ranges (many false breakouts)
    },
    'VOLATILE': {
        'breakout': 0.35,          # Good for explosive moves
        'momentum_trend': 0.25,    # Good for volatility expansion
        'ma_crossover': 0.20,      # Moderate in volatile conditions
        'donchian_channel': 0.15,  # Moderate adaptability
                'mean_reversion': 0.05,    # Poor in high volatility
    }
}

# News/Economic Event Filtering - avoid trading around major announcements
NEWS_BLACKOUT_TIMES = {
    # Major USD events (all times in UTC)
    'USD_MAJOR': [
        (8, 30),   # 8:30 UTC - US Economic data (NFP, CPI, etc.)
        (14, 0),   # 2:00 PM UTC - FOMC meetings  
        (18, 0),   # 6:00 PM UTC - Fed speeches
    ],
    'EUR_MAJOR': [
        (7, 0),    # 7:00 UTC - ECB interest rate decisions
        (8, 0),    # 8:00 UTC - EU inflation data
        (9, 0),    # 9:00 UTC - German/EU economic data
    ],
    'GBP_MAJOR': [
        (6, 0),    # 6:00 UTC - BOE interest rate decisions
        (8, 30),   # 8:30 UTC - UK employment/inflation data
        (9, 30),   # 9:30 UTC - UK GDP data
    ],
    'JPY_MAJOR': [
        (23, 50),  # 11:50 PM UTC - BOJ interest rate decisions  
        (0, 30),   # 12:30 AM UTC - Japanese economic data
    ]
}

# News/Economic Event Filtering - DISABLED (too restrictive for active trading)
ENABLE_NEWS_FILTERING = False  # Disable news filtering to allow more trading opportunities
NEWS_BLACKOUT_MINUTES = 30  # Minutes before/after news to avoid trading
NEWS_BOOST_MINUTES = 90     # Minutes after major news for increased activity

# Multi-timeframe confirmation settings
ENABLE_MTF_CONFIRMATION = True  # Enable multi-timeframe signal confirmation
MTF_TIMEFRAMES = [
    mt5.TIMEFRAME_M15,  # Primary timeframe (current)
    mt5.TIMEFRAME_H1,   # Higher timeframe for trend confirmation
]
MTF_LOOKBACK_BARS = 20  # Bars to fetch for higher timeframe analysis

# Advanced Stop Management - HIGH IMPACT ON RETURNS
ENABLE_ADVANCED_STOPS = True    # Enable advanced stop management
BREAKEVEN_TRIGGER = 1.0         # Move SL to breakeven after 1x ATR profit
TRAILING_START = 1.5            # Start trailing after 1.5x ATR profit  
TRAILING_DISTANCE = 1.0         # Trail SL 1x ATR behind price
PARTIAL_TAKE_PROFIT = 0.5       # Take 50% profit at 1:1 risk/reward
PARTIAL_TP_RATIO = 1.0          # Take partial profit at 1x risk
TRAILING_TIGHT_PROFIT = 2.5     # Tighten trailing once profit reaches 2.5x ATR
TRAILING_DISTANCE_TIGHT = 0.8   # Tighter trailing distance after strong move
TRAILING_ULTRA_PROFIT = 4.0     # Aggressively trail once profit exceeds 4x ATR
TRAILING_DISTANCE_ULTRA = 0.5   # Very tight trailing distance for deep winners
TRAILING_MICRO_DISTANCE = 0.65  # Micro-lot positions trail slightly tighter
PARTIAL_MICRO_FORCE_ENABLED = True
PARTIAL_MICRO_FORCE_ATR = 1.35          # Fully exit micro lots once profit exceeds this ATR ratio
PARTIAL_MICRO_FORCE_VOLUME_FACTOR = 1.6 # Treat positions <= factor * min volume as micro lots
PARTIAL_MICRO_FORCE_TRAIL = 0.45        # Use tighter trailing distance on micro lots waiting for force exit
PARTIAL_MICRO_FORCE_COMMENT = "Micro force exit"


def get_symbol_currency_exposure(symbol: str) -> list[str]:
    """Infer the primary currencies for a tradable instrument."""
    if not symbol:
        return ["USD"]

    normalized = symbol.upper().replace(".", "").replace("_", "").rstrip("+")

    custom_map = {
        "EURUSD": ["EUR", "USD"],
        "GBPUSD": ["GBP", "USD"],
        "USDJPY": ["USD", "JPY"],
        "GBPJPY": ["GBP", "JPY"],
        "XAUUSD": ["XAU", "USD"],
        "XAGUSD": ["XAG", "USD"],
        "BTCUSD": ["BTC", "USD"],
        "ETHUSD": ["ETH", "USD"],
    }

    if normalized in custom_map:
        return custom_map[normalized]

    if len(normalized) >= 6:
        base = normalized[:3]
        quote = normalized[3:6]
        return [base, quote]

    return ["USD"]


def is_news_blackout_period(symbol: str) -> tuple[bool, str]:
    """Check if current time is within news blackout period for symbol's currencies."""
    # Skip news filtering if disabled
    if not ENABLE_NEWS_FILTERING:
        return False, "News filtering disabled"
    
    try:
        now_utc = datetime.now(pytz.UTC)
        current_hour = now_utc.hour
        current_minute = now_utc.minute
        current_time = current_hour * 60 + current_minute  # Convert to minutes
        
        print(f"🕐 Current UTC time: {current_hour:02d}:{current_minute:02d} (checking {symbol})")
        
        currencies = get_symbol_currency_exposure(symbol)
        
        for currency in currencies:
            news_times = NEWS_BLACKOUT_TIMES.get(f"{currency}_MAJOR", [])
            
            for news_hour, news_minute in news_times:
                news_time = news_hour * 60 + news_minute
                
                # Calculate time difference (handle same day only for now)
                time_diff = abs(current_time - news_time)
                
                print(f"  📰 {currency} news at {news_hour:02d}:{news_minute:02d} - time diff: {time_diff} min (limit: {NEWS_BLACKOUT_MINUTES})")
                
                # Only consider blackout if we're within the same day and close to news time
                if time_diff <= NEWS_BLACKOUT_MINUTES:
                    return True, f"{currency} news at {news_hour:02d}:{news_minute:02d}"
        
        return False, ""
    except Exception as e:
        print(f"Error checking news blackout: {e}")
        return False, ""


def manage_position_stops(symbol: str, atr_value: float) -> dict[str, int]:
    """Advanced stop management: breakeven, trailing, partial profits."""
    stats = {
        "stops_breakeven": 0,
        "stops_trailing": 0,
        "stops_failed": 0,
        "partials_taken": 0,
        "partials_failed": 0,
        "partials_unavailable": 0,
        "partials_forced": 0,
    }

    if not ENABLE_ADVANCED_STOPS or not atr_value:
        return stats

    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            mt5.symbol_select(symbol, True)
            symbol_info = mt5.symbol_info(symbol)

        point = getattr(symbol_info, "point", 0.00001) or 1e-5
        digits = getattr(symbol_info, "digits", 5)
        min_volume = getattr(symbol_info, "volume_min", 0.01) or 0.01
        min_stop_distance = 0.0
        if RESPECT_STOPS_LEVEL:
            stop_level_points = float(getattr(symbol_info, "trade_stops_level", 0) or 0.0)
            step_points = float(getattr(symbol_info, "trade_stops_step", 0) or 0.0)
            freeze_points = float(getattr(symbol_info, "trade_freeze_level", 0) or 0.0)
            min_stop_distance = max(stop_level_points, step_points, freeze_points) * point

        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return stats

        for position in positions:
            ticket = position.ticket
            position_type = position.type  # 0=buy, 1=sell
            open_price = float(getattr(position, "price_open", 0.0) or 0.0)
            current_price = float(getattr(position, "price_current", 0.0) or 0.0)
            volume = float(getattr(position, "volume", 0.0) or 0.0)
            raw_sl = float(getattr(position, "sl", 0.0) or 0.0)
            sl = raw_sl if raw_sl not in (0.0, None) else None

            if volume <= 0:
                continue

            micro_lot_threshold = min_volume * PARTIAL_MICRO_FORCE_VOLUME_FACTOR
            micro_lot = volume <= (micro_lot_threshold + 1e-9)

            if position_type == mt5.POSITION_TYPE_BUY:
                profit_distance = current_price - open_price
                is_profitable = profit_distance > 0
            else:
                profit_distance = open_price - current_price
                is_profitable = profit_distance > 0

            if not is_profitable:
                continue

            profit_atr = profit_distance / atr_value if atr_value > 0 else 0.0
            breakeven_target = round(open_price, digits)

            # 1. Move to breakeven after threshold profit
            if profit_atr >= BREAKEVEN_TRIGGER and (sl is None or abs(sl - breakeven_target) > point):
                distance_to_price = (
                    (current_price - breakeven_target)
                    if position_type == mt5.POSITION_TYPE_BUY
                    else (breakeven_target - current_price)
                )
                if min_stop_distance > 0 and distance_to_price < min_stop_distance:
                    shortfall_pts = (min_stop_distance - distance_to_price) / point if point else 0.0
                    print(
                        f"ℹ️ {symbol} ticket {ticket}: skipping breakeven lock (short by {shortfall_pts:.1f} pts to broker minimum)."
                    )
                elif modify_position_sl(ticket, breakeven_target):
                    stats["stops_breakeven"] += 1
                    old_sl_display = sl if sl is not None else 0.0
                    print(
                        f"💰 {symbol} ticket {ticket}: SL {old_sl_display:.5f}→{breakeven_target:.5f} "
                        f"(breakeven lock, profit {profit_atr:.1f}x ATR)"
                    )
                    sl = breakeven_target
                else:
                    stats["stops_failed"] += 1
                    print(
                        f"⚠️ {symbol} ticket {ticket}: breakeven SL update rejected (target {breakeven_target:.5f})."
                    )

            # 2. Dynamic trailing stop once position runs
            if profit_atr >= TRAILING_START:
                trail_distance = TRAILING_DISTANCE
                if profit_atr >= TRAILING_TIGHT_PROFIT:
                    trail_distance = min(trail_distance, TRAILING_DISTANCE_TIGHT)
                if profit_atr >= TRAILING_ULTRA_PROFIT:
                    trail_distance = min(trail_distance, TRAILING_DISTANCE_ULTRA)
                if micro_lot:
                    tight_micro_trail = PARTIAL_MICRO_FORCE_TRAIL if profit_atr >= PARTIAL_TP_RATIO else TRAILING_MICRO_DISTANCE
                    trail_distance = min(trail_distance, tight_micro_trail)

                trail_distance = max(0.1, trail_distance)

                if position_type == mt5.POSITION_TYPE_BUY:
                    trail_sl = current_price - (trail_distance * atr_value)
                    should_move = sl is None or trail_sl > (sl + point)
                else:
                    trail_sl = current_price + (trail_distance * atr_value)
                    should_move = sl is None or trail_sl < (sl - point)

                if min_stop_distance > 0:
                    if position_type == mt5.POSITION_TYPE_BUY:
                        trail_sl = min(trail_sl, current_price - min_stop_distance)
                    else:
                        trail_sl = max(trail_sl, current_price + min_stop_distance)

                trail_sl = round(trail_sl, digits)

                distance_to_price = (
                    (current_price - trail_sl)
                    if position_type == mt5.POSITION_TYPE_BUY
                    else (trail_sl - current_price)
                )

                if min_stop_distance > 0 and distance_to_price < min_stop_distance:
                    print(
                        f"ℹ️ {symbol} ticket {ticket}: trailing SL candidate too tight (gap {distance_to_price/point:.1f} pts < broker minimum)."
                    )
                    should_move = False

                if position_type == mt5.POSITION_TYPE_BUY:
                    should_move = should_move and (sl is None or trail_sl > (sl + point))
                else:
                    should_move = should_move and (sl is None or trail_sl < (sl - point))

                if should_move and trail_sl > 0:
                    if modify_position_sl(ticket, trail_sl):
                        stats["stops_trailing"] += 1
                        old_sl_display = sl if sl is not None else breakeven_target
                        direction_icon = "📈" if position_type == mt5.POSITION_TYPE_BUY else "📉"
                        print(
                            f"{direction_icon} {symbol} ticket {ticket}: trailing SL {old_sl_display:.5f}→{trail_sl:.5f} "
                            f"({trail_distance:.2f} ATR span, profit {profit_atr:.1f}x ATR)"
                        )
                        sl = trail_sl
                    else:
                        stats["stops_failed"] += 1
                        print(
                            f"⚠️ {symbol} ticket {ticket}: trailing SL update rejected (target {trail_sl:.5f})."
                        )

            # 3. Partial profit taking when contract size permits
            if profit_atr >= PARTIAL_TP_RATIO:
                partial_volume = volume * PARTIAL_TAKE_PROFIT
                if partial_volume < min_volume - 1e-9:
                    stats["partials_unavailable"] += 1
                    if micro_lot:
                        print(
                            f"ℹ️ {symbol} ticket {ticket}: micro lot {volume:.2f} can't partial (min {min_volume:.2f}). "
                            f"Using tight trail {PARTIAL_MICRO_FORCE_TRAIL:.2f} ATR while awaiting force exit."
                        )
                        if profit_atr >= PARTIAL_MICRO_FORCE_ATR and PARTIAL_MICRO_FORCE_ENABLED:
                            if close_partial_position(ticket, volume, reason=PARTIAL_MICRO_FORCE_COMMENT):
                                stats["partials_forced"] += 1
                                print(
                                    f"✅ {symbol} ticket {ticket}: forced full exit at {profit_atr:.1f}x ATR (micro lot)."
                                )
                                continue
                            else:
                                stats["partials_failed"] += 1
                                print(
                                    f"⚠️ {symbol} ticket {ticket}: forced exit request failed for {volume:.2f} lots."
                                )
                    else:
                        print(
                            f"ℹ️ {symbol} ticket {ticket}: partial target {partial_volume:.2f} below min lot "
                            f"{min_volume:.2f}; relying on trailing instead."
                        )
                else:
                    if close_partial_position(ticket, partial_volume):
                        stats["partials_taken"] += 1
                        remaining = max(0.0, volume - partial_volume)
                        print(
                            f"💵 {symbol} ticket {ticket}: partial {partial_volume:.2f} lots secured at {profit_atr:.1f}x ATR "
                            f"(remaining {remaining:.2f})."
                        )
                    else:
                        stats["partials_failed"] += 1
                        print(
                            f"⚠️ {symbol} ticket {ticket}: partial close request failed for {partial_volume:.2f} lots."
                        )

        return stats

    except Exception as e:
        print(f"Error managing stops for {symbol}: {e}")
        return stats


def modify_position_sl(ticket: int, new_sl: float, attempts: int = 3, backoff: float = 0.6) -> bool:
    """Modify stop loss for an existing position with basic retry handling."""
    try:
        retryable_codes = {
            getattr(mt5, "TRADE_RETCODE_REQUOTE", None),
            getattr(mt5, "TRADE_RETCODE_REJECT", None),
            getattr(mt5, "TRADE_RETCODE_INVALID_PRICE", None),
            getattr(mt5, "TRADE_RETCODE_PRICE_CHANGED", None),
            getattr(mt5, "TRADE_RETCODE_MARKET_CLOSED", None),
            getattr(mt5, "TRADE_RETCODE_NO_CONNECTION", None),
        }

        for attempt in range(1, max(1, attempts) + 1):
            position = mt5.positions_get(ticket=ticket)
            if not position:
                print(f"⚠️  Unable to modify SL for ticket {ticket}: position not found.")
                return False

            pos = position[0]
            request = {
                "action": mt5.TRADE_ACTION_SLTP,
                "position": ticket,
                "sl": new_sl,
                "tp": pos.tp,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                if attempt > 1:
                    print(f"✅ Ticket {ticket}: SL update succeeded on retry #{attempt}.")
                return True

            reason = result.comment if result else "No result"
            retcode = result.retcode if result else None
            is_retryable = retcode in retryable_codes

            if attempt < attempts and is_retryable:
                delay = backoff * attempt
                print(
                    f"⏳ Ticket {ticket}: SL update retry #{attempt} failed ({reason}); retrying in {delay:.1f}s..."
                )
                time.sleep(delay)
                continue

            print(f"Failed to modify SL for ticket {ticket}: {reason}")
            return False

        return False

    except Exception as e:
        print(f"Error modifying SL for ticket {ticket}: {e}")
        return False


def close_partial_position(ticket: int, volume: float, reason: str = "Partial profit") -> bool:
    """Close a portion of an open position for profit taking or guard-driven scaling."""
    try:
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        position = position[0]
        symbol = position.symbol

        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            mt5.symbol_select(symbol, True)
            symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"Unable to retrieve symbol info for {symbol}; skipping partial close.")
            return False

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"No tick data for {symbol}; cannot execute partial close.")
            return False
        
        # Determine order type (opposite of position type)
        if position.type == 0:  # Close buy position with sell order
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:  # Close sell position with buy order
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        max_closeable = float(getattr(position, 'volume', 0.0) or 0.0)
        if max_closeable <= 0:
            return False

        requested_volume = min(volume, max_closeable)
        normalized_volume = normalize_volume(requested_volume, symbol_info)

        if normalized_volume <= 0:
            print(f"Partial close volume normalized to zero for {symbol}; aborting.")
            return False

        if normalized_volume > max_closeable:
            normalized_volume = max_closeable

        residual = max_closeable - normalized_volume
        min_volume = getattr(symbol_info, 'volume_min', 0.01) or 0.01
        step = getattr(symbol_info, 'volume_step', 0.01) or 0.01

        if residual > 0 and residual < min_volume:
            # Leaving less than the minimum causes rejection; adjust to close entire position
            normalized_volume = max_closeable

        filling_type = resolve_filling_type(symbol_info)
        time_type = getattr(mt5, 'ORDER_TIME_GTC', 0)
        
        comment_text = sanitize_comment(reason or "Partial profit")

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": normalized_volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "comment": comment_text,
            "type_filling": filling_type,
            "type_time": time_type,
            "deviation": 10,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            return True
        elif result and result.retcode == getattr(mt5, 'TRADE_RETCODE_PLACED', -1):
            print(f"Partial close for ticket {ticket} placed (pending).")
            return True
        else:
            print(f"Failed to close partial position for ticket {ticket}: {result.comment if result else 'No result'}")
            return False
            
    except Exception as e:
        print(f"Error closing partial position for ticket {ticket}: {e}")
        return False


def close_symbol_positions(
    symbol: str,
    side: str | None = None,
    reason: str | None = None,
    deviation: int = 10,
) -> int:
    """Close all matching positions for a symbol, optionally filtered by side ('buy' or 'sell')."""
    try:
        if not ensure_connection_ready():
            return 0

        positions = mt5.positions_get(symbol=symbol)
        if not positions:
            return 0

        symbol_info = prepare_symbol(symbol)
        if symbol_info is None:
            return 0

        filling = resolve_filling_type(symbol_info)
        closed = 0
        base_comment = sanitize_comment(reason or COUNTER_SIGNAL_EXIT_COMMENT)

        for position in positions:
            pos_side = 'buy' if position.type == mt5.POSITION_TYPE_BUY else 'sell'
            if side and pos_side != side:
                continue

            volume = float(getattr(position, 'volume', 0.0) or 0.0)
            if volume <= 0:
                continue

            tick = mt5.symbol_info_tick(symbol)
            if tick is None:
                print(f"⚠️ {symbol}: Missing tick data while attempting to close positions.")
                break

            order_type = mt5.ORDER_TYPE_SELL if pos_side == 'buy' else mt5.ORDER_TYPE_BUY
            price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
            normalized_volume = normalize_volume(volume, symbol_info)
            request_comment = sanitize_comment(f"{base_comment}-{pos_side[:1].upper()}")

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "position": position.ticket,
                "volume": normalized_volume,
                "type": order_type,
                "price": price,
                "deviation": deviation,
                "type_filling": filling,
                "comment": request_comment,
            }

            result = mt5.order_send(request)
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                closed += 1
                print(
                    f"✅ {symbol}: Closed {pos_side} ticket {position.ticket} ({normalized_volume:.2f} lots) via {request_comment}."
                )
            else:
                reason_text = result.comment if result else "No result"
                print(f"⚠️ {symbol}: Failed to close ticket {position.ticket} ({pos_side}) -> {reason_text}")

        return closed
    except Exception as exc:
        print(f"Error while closing positions for {symbol}: {exc}")
        return 0


def execute_margin_unwind(
    margin_status: dict[str, float | bool | str | None],
    positions: Iterable[Any],
) -> dict[str, float | int | str | bool]:
    """Automatically trim exposure when the margin guard begins throttling or blocks entries."""
    stats: dict[str, float | int | str | bool] = {
        "attempted": False,
        "closed_tickets": 0,
        "closed_volume": 0.0,
        "failures": 0,
        "reason": "",
    }

    if not MARGIN_UNWIND_ENABLED:
        return stats

    position_list = list(positions or [])
    if not position_list:
        return stats

    throttle = float(margin_status.get("throttle", 1.0) or 1.0)
    severity = float(margin_status.get("severity", 0.0) or 0.0)
    allowed = bool(margin_status.get("allowed", True))
    status = str(margin_status.get("status") or "clear")
    usage = float(margin_status.get("usage", 0.0) or 0.0)
    free_ratio = float(margin_status.get("free_ratio", 1.0) or 1.0)

    should_unwind = False
    trigger_reason = ""

    if not allowed or status == "blocked":
        should_unwind = True
        trigger_reason = "blocked"
    elif status == "throttle" or throttle < MARGIN_UNWIND_TRIGGER_THROTTLE or severity >= MARGIN_UNWIND_TRIGGER_SEVERITY:
        should_unwind = True
        trigger_reason = "throttle"

    if not should_unwind:
        return stats

    now = datetime.now(timezone.utc)
    last_unwind_iso = margin_guard_state.get("last_unwind")
    if isinstance(last_unwind_iso, str):
        try:
            last_unwind = datetime.fromisoformat(last_unwind_iso)
            if (now - last_unwind).total_seconds() < MARGIN_UNWIND_COOLDOWN_SECONDS:
                stats["reason"] = "cooldown"
                return stats
        except ValueError:
            pass

    total_volume = sum(max(0.0, float(getattr(pos, "volume", 0.0) or 0.0)) for pos in position_list)
    if total_volume <= 0.0:
        return stats

    if not allowed or status == "blocked":
        target_fraction = MARGIN_UNWIND_BLOCK_REDUCTION + severity * 0.25
    else:
        throttle_deficit = max(0.0, MARGIN_UNWIND_TRIGGER_THROTTLE - throttle)
        target_fraction = MARGIN_UNWIND_MIN_REDUCTION + (throttle_deficit * 0.8) + (severity * 0.4)

    target_fraction = max(MARGIN_UNWIND_MIN_REDUCTION, min(MARGIN_UNWIND_MAX_REDUCTION, target_fraction))
    target_volume = max(MARGIN_UNWIND_MIN_VOLUME, total_volume * target_fraction)
    target_volume = min(target_volume, total_volume)

    stats["attempted"] = True
    stats["reason"] = trigger_reason or status

    sorted_positions: list[dict[str, float | int | str]] = []
    for pos in position_list:
        volume = float(getattr(pos, "volume", 0.0) or 0.0)
        if volume <= 0:
            continue
        profit = float(getattr(pos, "profit", 0.0) or 0.0)
        margin = float(getattr(pos, "margin", 0.0) or 0.0)
        symbol = str(getattr(pos, "symbol", ""))
        ticket = int(getattr(pos, "ticket", 0))
        sorted_positions.append(
            {
                "ticket": ticket,
                "volume": volume,
                "profit": profit,
                "margin": margin,
                "symbol": symbol,
            }
        )

    if not sorted_positions:
        return stats

    sorted_positions.sort(
        key=lambda entry: (
            0 if float(entry.get("profit", 0.0) or 0.0) < 0 else 1,
            -float(entry.get("margin", 0.0) or 0.0),
            -float(entry.get("volume", 0.0) or 0.0),
        )
    )

    remaining = target_volume
    for entry in sorted_positions:
        if remaining <= MARGIN_UNWIND_MIN_VOLUME * 0.5:
            break

        ticket = int(entry["ticket"])
        volume = float(entry["volume"])
        if volume <= 0:
            continue

        close_amount = min(volume, remaining)
        if close_amount < MIN_LOT_SIZE:
            if remaining >= volume - (MIN_LOT_SIZE * 0.4):
                close_amount = volume
            else:
                continue

        success = close_partial_position(ticket, close_amount, reason="Margin unwind")
        if success:
            stats["closed_tickets"] += 1
            stats["closed_volume"] += close_amount
            remaining = max(0.0, remaining - close_amount)
            time.sleep(0.2)
        else:
            stats["failures"] += 1

        if remaining <= MIN_LOT_SIZE * 0.4:
            break

    if stats["closed_volume"] > 0:
        margin_guard_state["last_unwind"] = now.isoformat()
        print(
            "🧹 Margin guard unwind: trimmed "
            f"{stats['closed_volume']:.2f} lots across {stats['closed_tickets']} tickets "
            f"(status {status}, throttle {throttle:.0%}, severity {severity:.0%}, "
            f"usage {usage:.0%}, free {free_ratio:.0%})."
        )
    elif stats["attempted"]:
        print(
            "⚠️ Margin guard unwind attempt yielded no reductions "
            f"(status {status}, throttle {throttle:.0%}, severity {severity:.0%})."
        )

    return stats


def clear_counter_signal_state(symbol: str) -> None:
    """Reset accumulated counter-signal tracking for the given symbol."""
    counter_signal_tracker.pop(symbol, None)


def should_force_counter_exit(
    symbol: str,
    incoming_signal: str,
    candidate_meta: dict[str, object],
    positions: list | None,
    atr_value: float | None,
    scan_number: int,
) -> tuple[bool, str | None]:
    """Evaluate whether to exit opposing exposure to support a strong counter signal."""
    if not COUNTER_SIGNAL_EXIT_ENABLED or not positions:
        return False, None

    if incoming_signal not in {"buy", "sell"}:
        return False, None

    direction_type = mt5.POSITION_TYPE_SELL if incoming_signal == 'buy' else mt5.POSITION_TYPE_BUY
    relevant_positions = [p for p in positions if getattr(p, 'type', None) == direction_type]
    if not relevant_positions:
        clear_counter_signal_state(symbol)
        return False, None

    net_profit = sum(float(getattr(p, 'profit', 0.0) or 0.0) for p in relevant_positions)
    if net_profit > COUNTER_SIGNAL_EXIT_MIN_NET_LOSS:
        clear_counter_signal_state(symbol)
        return False, None

    effective_atr = float(atr_value) if atr_value else None
    if effective_atr is None or effective_atr <= 0:
        effective_atr = calculate_atr(symbol, timeframe=mt5.TIMEFRAME_M15, period=ATR_PERIOD)
    if effective_atr is None or effective_atr <= 0:
        return False, None

    same_dd, opposite_dd, worst_dd, total_dd = _calculate_drawdown_pressure_atr(
        relevant_positions,
        effective_atr,
        incoming_signal,
    )
    loss_atr = max(opposite_dd, worst_dd, total_dd)
    if loss_atr < COUNTER_SIGNAL_EXIT_MIN_ATR_LOSS:
        tracker = counter_signal_tracker.setdefault(
            symbol,
            {"direction": incoming_signal, "streak": 0, "last_scan": 0},
        )
        tracker["streak"] = 0
        tracker["direction"] = incoming_signal
        tracker["last_scan"] = scan_number
        return False, None

    confidence = float(candidate_meta.get('confidence') or 0.0)
    micro_score = candidate_meta.get('micro_momentum')
    micro_soft = bool(candidate_meta.get('micro_soft_pass'))
    micro_alignment = False
    if micro_score is not None:
        score_val = float(micro_score)
        micro_alignment = score_val >= 0 if incoming_signal == 'buy' else score_val <= 0
    if micro_soft and micro_score is not None:
        micro_alignment = True

    if COUNTER_SIGNAL_EXIT_REQUIRE_MICRO and not micro_alignment:
        tracker = counter_signal_tracker.setdefault(
            symbol,
            {"direction": incoming_signal, "streak": 0, "last_scan": 0},
        )
        tracker["streak"] = 0
        tracker["direction"] = incoming_signal
        tracker["last_scan"] = scan_number
        return False, None

    tracker = counter_signal_tracker.setdefault(
        symbol,
        {"direction": incoming_signal, "streak": 0, "last_scan": 0},
    )
    if tracker.get("direction") != incoming_signal:
        tracker["streak"] = 0
    last_scan = int(tracker.get("last_scan", 0) or 0)
    if scan_number - last_scan > COUNTER_SIGNAL_EXIT_RESET_SCANS:
        tracker["streak"] = 0

    tracker["direction"] = incoming_signal
    tracker["last_scan"] = scan_number

    if confidence >= COUNTER_SIGNAL_EXIT_MIN_CONFIDENCE and net_profit <= COUNTER_SIGNAL_EXIT_MIN_NET_LOSS:
        tracker["streak"] = int(tracker.get("streak", 0)) + 1
        if tracker["streak"] >= COUNTER_SIGNAL_EXIT_MIN_STREAK:
            tracker["streak"] = 0
            reason = (
                f"confidence {confidence:.2f}, drawdown {loss_atr:.2f} ATR, net {net_profit:.2f}"
            )
            return True, reason
    else:
        tracker["streak"] = 0

    return False, None


def get_mtf_trend_bias(symbol: str) -> str:
    """Get higher timeframe trend bias for signal confirmation."""
    try:
        if not ENABLE_MTF_CONFIRMATION:
            return 'neutral'  # No filtering if MTF disabled
        
        # Get higher timeframe data (H1)
        today = datetime.now()
        from_date = today - timedelta(hours=MTF_LOOKBACK_BARS)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_H1, from_date, today)
        
        if rates is None or len(rates) < 10:
            return 'neutral'  # Insufficient data
        
        df_h1 = pd.DataFrame(rates)
        df_h1['time'] = pd.to_datetime(df_h1['time'], unit='s')
        
        # Simple trend detection using MA
        close = df_h1['close']
        ma_short = close.rolling(5).mean()
        ma_long = close.rolling(10).mean()
        
        current_price = close.iloc[-1]
        ma_short_current = ma_short.iloc[-1]
        ma_long_current = ma_long.iloc[-1]
        
        if pd.isna(ma_short_current) or pd.isna(ma_long_current):
            return 'neutral'
        
        # Determine trend bias
        if ma_short_current > ma_long_current and current_price > ma_short_current:
            return 'bullish'
        elif ma_short_current < ma_long_current and current_price < ma_short_current:
            return 'bearish'
        else:
            return 'neutral'
            
    except Exception as e:
        print(f"Error getting MTF bias for {symbol}: {e}")
        return 'neutral'


def confirm_signal_with_mtf(signal: str, symbol: str) -> bool:
    """Confirm M15 signal with higher timeframe bias."""
    if not ENABLE_MTF_CONFIRMATION:
        return True  # No filtering if disabled
    
    mtf_bias = get_mtf_trend_bias(symbol)
    
    # Allow signal if:
    # 1. MTF is neutral (no strong bias)
    # 2. Signal aligns with MTF bias
    # 3. MTF bias detection failed (neutral)
    
    if mtf_bias == 'neutral':
        return True
    elif signal == 'buy' and mtf_bias == 'bullish':
        return True
    elif signal == 'sell' and mtf_bias == 'bearish':
        return True
    else:
        return False


def _calculate_drawdown_pressure_atr(
    positions: list | None,
    atr_value: float | None,
    signal: str,
) -> tuple[float, float, float, float]:
    """Estimate drawdown pressure in ATR units for the given signal direction.

    Returns tuple of (same_direction, opposite_direction, worst_single, total_loss)
    expressed in ATR multiples. Values are 0.0 when unavailable.
    """
    if not positions or atr_value is None or atr_value <= 0:
        return 0.0, 0.0, 0.0, 0.0

    same_direction_dd = 0.0
    opposite_direction_dd = 0.0
    worst_dd = 0.0
    total_dd = 0.0

    for pos in positions:
        price_open = float(getattr(pos, 'price_open', 0.0) or 0.0)
        price_current = float(getattr(pos, 'price_current', 0.0) or 0.0)
        if price_open <= 0 or price_current <= 0:
            continue

        if getattr(pos, 'type', None) == mt5.POSITION_TYPE_BUY:
            adverse_move = price_open - price_current
            pos_side = 'buy'
        else:
            adverse_move = price_current - price_open
            pos_side = 'sell'

        if adverse_move <= 0:
            continue  # Position is not under water

        drawdown_atr = adverse_move / atr_value
        worst_dd = max(worst_dd, drawdown_atr)
        total_dd += drawdown_atr

        if (signal == 'buy' and pos_side == 'buy') or (signal == 'sell' and pos_side == 'sell'):
            same_direction_dd = max(same_direction_dd, drawdown_atr)
        else:
            opposite_direction_dd = max(opposite_direction_dd, drawdown_atr)

    return same_direction_dd, opposite_direction_dd, worst_dd, total_dd


def confirm_with_micro_momentum(
    symbol: str,
    signal: str,
    regime: str,
    atr_value: float | None,
    reference_price: float | None,
    open_positions: list | None = None,
    guard_factor: float | None = None,
) -> tuple[bool, float, float, bool, bool]:
    """Use micro timeframe momentum bias to refine entries."""
    if not ENABLE_MICRO_MOMENTUM_CONFIRMATION:
        return True, 0.0, 0.0, False, False

    try:
        override_used = False
        guard_factor_val = float(guard_factor) if guard_factor is not None else 1.0
        guard_factor_val = max(0.0, min(guard_factor_val, 1.5))
        request_count = MICRO_MOMENTUM_LOOKBACK + 3
        rates = mt5.copy_rates_from_pos(symbol, MICRO_MOMENTUM_TIMEFRAME, 0, request_count)
        if rates is None or len(rates) < MICRO_MOMENTUM_LOOKBACK + 2:
            return True, 0.0, 0.0, False, False  # Skip if insufficient data

        df = pd.DataFrame(rates)
        close = df['close'].astype(float)
        momentum_series = close.pct_change().rolling(MICRO_MOMENTUM_LOOKBACK).sum()
        momentum_score = momentum_series.iloc[-1]

        if pd.isna(momentum_score):
            return True, 0.0, 0.0, False, False

        # Derive adaptive threshold based on ATR and market regime
        threshold = MICRO_MOMENTUM_BASE_THRESHOLD
        if atr_value and reference_price:
            atr_pct = max(1e-9, atr_value / reference_price)
            dynamic = atr_pct * MICRO_MOMENTUM_DYNAMIC_MULTIPLIER
            threshold = max(MICRO_MOMENTUM_MIN_THRESHOLD, min(MICRO_MOMENTUM_MAX_THRESHOLD, dynamic))

        if regime == 'TRENDING':
            threshold *= 0.75
        elif regime == 'VOLATILE':
            threshold *= 0.85
        threshold = max(MICRO_MOMENTUM_MIN_THRESHOLD, threshold)

        same_dir_dd_atr, opposite_dir_dd_atr, worst_dd_atr, total_dd_atr = _calculate_drawdown_pressure_atr(
            open_positions,
            atr_value,
            signal,
        )
        drawdown_pressure_atr = same_dir_dd_atr if same_dir_dd_atr > 0 else opposite_dir_dd_atr

        if drawdown_pressure_atr > 0:
            relaxation_multiplier = max(0.25, 1.0 - min(2.0, drawdown_pressure_atr) * DRAWDOWN_RELAXATION_PER_ATR)
            if relaxation_multiplier < 1.0:
                threshold *= relaxation_multiplier

        soft_pass = False
        aligned_momentum = momentum_score if signal == 'buy' else -momentum_score
        tolerance = threshold * AGGRESSIVE_MICRO_TOLERANCE
        if drawdown_pressure_atr > 0:
            tolerance *= 1.0 + min(0.6, drawdown_pressure_atr * 0.3)
        alignment_ratio = AGGRESSIVE_MICRO_MIN_RATIO
        if regime == 'VOLATILE':
            alignment_ratio *= 0.5
        elif regime == 'RANGING':
            alignment_ratio *= 0.75
        if drawdown_pressure_atr > 0:
            alignment_ratio *= max(0.4, 1.0 - min(1.5, drawdown_pressure_atr) * 0.2)
        alignment_ratio = max(0.05, alignment_ratio)
        required_alignment = threshold * alignment_ratio

        if signal == 'buy':
            if momentum_score >= threshold:
                return True, float(momentum_score), threshold, soft_pass, override_used
            elif momentum_score >= threshold * MICRO_MOMENTUM_SOFT_PASS_RATIO:
                soft_pass = True
                return True, float(momentum_score), threshold, soft_pass, override_used
        else:
            if momentum_score <= -threshold:
                return True, float(momentum_score), threshold, soft_pass, override_used
            elif momentum_score <= -threshold * MICRO_MOMENTUM_SOFT_PASS_RATIO:
                soft_pass = True
                return True, float(momentum_score), threshold, soft_pass, override_used

        if drawdown_pressure_atr >= DRAWDOWN_OVERRIDE_ATR and aligned_momentum >= -tolerance:
            soft_pass = True
            override_used = True
            print(
                f"🩹 {symbol}: Drawdown override for {signal.upper()} (pressure {drawdown_pressure_atr:.2f} ATR, momentum {aligned_momentum:+.2%} within tolerance)."
            )
            return True, float(momentum_score), threshold, soft_pass, override_used

        if AGGRESSIVE_MODE and regime in ('TRENDING', 'VOLATILE'):
            if aligned_momentum >= required_alignment:
                soft_pass = True
                override_used = True
                tolerance_pct = f"±{tolerance:.2%}"
                required_symbol = "≥" if signal == 'buy' else "≤"
                required_value = required_alignment if signal == 'buy' else -required_alignment
                required_pct = f"{required_symbol} {required_value:.2%}"
                print(
                    f"⚡ {symbol}: Aggressive micro override for {signal.upper()} (aligned {aligned_momentum:+.2%} within tolerance {tolerance_pct}, required {required_pct})."
                )
                return True, float(momentum_score), threshold, soft_pass, override_used

        guard_relaxed = guard_factor_val >= MICRO_MOMENTUM_GUARD_MIN
        drawdown_ok = drawdown_pressure_atr <= MICRO_MOMENTUM_NEAR_MISS_MAX_DD
        near_miss_ratio = MICRO_MOMENTUM_NEAR_MISS_RATIO

        if guard_relaxed and drawdown_ok:
            if signal == 'buy' and momentum_score >= threshold * near_miss_ratio:
                soft_pass = True
                override_used = True
                print(
                    f"🪄 {symbol}: Guard-approved micro near-miss for BUY (score {momentum_score:+.2%} vs threshold {threshold:.2%}, guard {guard_factor_val:.2f})."
                )
                return True, float(momentum_score), threshold, soft_pass, override_used
            if signal == 'sell' and momentum_score <= -threshold * near_miss_ratio:
                soft_pass = True
                override_used = True
                print(
                    f"🪄 {symbol}: Guard-approved micro near-miss for SELL (score {momentum_score:+.2%} vs threshold {-threshold:.2%}, guard {guard_factor_val:.2f})."
                )
                return True, float(momentum_score), threshold, soft_pass, override_used

        return False, float(momentum_score), threshold, soft_pass, override_used
    except Exception as e:
        print(f"Error computing micro momentum for {symbol}: {e}")
        return True, 0.0, MICRO_MOMENTUM_MIN_THRESHOLD, False, False


def get_instrument_session_priority(symbol: str, session: str) -> int:
    """Get the trading priority for a symbol during a specific session."""
    return INSTRUMENT_SESSION_PRIORITY.get(symbol, {}).get(session, 3)


def should_trade_instrument_in_session(symbol: str) -> bool:
    """Check if we should trade this instrument in the current session."""
    try:
        session = get_session_info()
        if session == "Dead Zone":
            return False
        
        priority = get_instrument_session_priority(symbol, session)
        return priority >= MIN_SESSION_PRIORITY
    except Exception:
        return True  # Default to allow if error


def get_correlated_open_positions(symbol: str) -> list:
    """Get list of open positions in symbols correlated with the given symbol."""
    try:
        correlated_positions = []
        
        # Find which correlation groups this symbol belongs to
        symbol_groups = []
        for group_name, group_symbols in CORRELATION_GROUPS.items():
            if symbol in group_symbols:
                symbol_groups.append(group_name)
        
        # Get all open positions
        all_positions = mt5.positions_get() or []
        
        # Check each position to see if it's in a correlated group
        for position in all_positions:
            pos_symbol = position.symbol
            for group_name in symbol_groups:
                if pos_symbol in CORRELATION_GROUPS[group_name] and pos_symbol != symbol:
                    correlated_positions.append({
                        'symbol': pos_symbol,
                        'type': 'buy' if position.type == 0 else 'sell',
                        'volume': position.volume,
                        'group': group_name
                    })
        
        return correlated_positions
    except Exception as e:
        print(f"Error checking correlated positions: {e}")
        return []


def should_limit_correlation_exposure(
    symbol: str,
    signal: str,
    confidence: float | None = None,
    guard_factor: float | None = None,
    drawdown_pressure: float | None = None,
) -> tuple[bool, float]:
    """Check if we should limit position size due to correlation exposure.
    
    Returns:
        (should_limit: bool, size_multiplier: float)
    """
    try:
        correlated_positions = get_correlated_open_positions(symbol)
        
        if not correlated_positions:
            return False, 1.0  # No correlation limits

        confidence = float(confidence or 0.0)
        guard_factor = float(guard_factor) if guard_factor is not None else 1.0
        drawdown_pressure = float(drawdown_pressure or 0.0)
        high_confidence = confidence >= CORRELATION_HIGH_CONFIDENCE_THRESHOLD
        guard_supportive = guard_factor >= CORRELATION_GUARD_SUPPORT_THRESHOLD
        drawdown_tight = drawdown_pressure >= CORRELATION_DRAWDOWN_TIGHTEN_ATR
        drawdown_hard_stop = drawdown_pressure >= CORRELATION_DRAWDOWN_CUTOFF_ATR
        
        group_stats: dict[str, dict[str, float]] = {}
        for pos in correlated_positions:
            group = pos.get('group')
            if not group:
                continue
            stats = group_stats.setdefault(
                group,
                {
                    'same': 0,
                    'opposite': 0,
                    'total': 0,
                    'same_volume': 0.0,
                    'opposite_volume': 0.0,
                    'total_volume': 0.0,
                },
            )
            stats['total'] += 1
            volume = float(pos.get('volume') or 0.0)
            stats['total_volume'] += volume
            if pos['type'] == signal:
                stats['same'] += 1
                stats['same_volume'] += volume
            else:
                stats['opposite'] += 1
                stats['opposite_volume'] += volume

        size_multiplier = 1.0
        scale_records: list[dict[str, object]] = []

        for group, stats in group_stats.items():
            limits = CORRELATION_GROUP_LIMITS.get(group, DEFAULT_CORRELATION_LIMITS)
            base_multiplier = float(limits.get('size_multiplier', DEFAULT_CORRELATION_LIMITS['size_multiplier']))
            net_bias = stats['same'] - stats['opposite']
            hedged = stats['opposite'] >= max(1, stats['same'] - CORRELATION_HEDGE_BIAS_SLACK)
            group_multiplier = 1.0
            reasons: list[str] = []

            if stats['same'] >= limits['max_same_direction']:
                override_scale: float | None = None
                override_reason: str | None = None
                max_same_volume = float(limits.get('max_same_direction_volume') or 0.0)
                if (
                    max_same_volume > 0
                    and stats.get('same_volume', 0.0) <= max_same_volume
                    and not drawdown_hard_stop
                ):
                    override_scale = float(limits.get('volume_relief_scale') or base_multiplier)
                    override_reason = (
                        f"volume relief {stats.get('same_volume', 0.0):.2f}/{max_same_volume:.2f} lots"
                    )
                elif hedged and not drawdown_hard_stop:
                    override_scale = base_multiplier * CORRELATION_HEDGE_RELIEF_SCALE
                    override_reason = f"hedged exposure ({stats['opposite']} opposite)"
                elif high_confidence and guard_supportive and not drawdown_hard_stop:
                    override_scale = base_multiplier * CORRELATION_OVERRIDE_SCALE
                    override_reason = f"high confidence {confidence:.2f}"

                if override_scale is not None:
                    if drawdown_tight:
                        override_scale *= 0.9
                    override_scale = min(0.95, max(CORRELATION_OVERRIDE_MIN_SCALE, override_scale))
                    group_multiplier = min(group_multiplier, override_scale)
                    reasons.append(f"override: {override_reason}")
                else:
                    print(
                        f"🔗 {symbol}: Blocking trade - {stats['same']} same-direction positions in {group} "
                        f"(limit {limits['max_same_direction']})."
                    )
                    return True, 0.0

            if stats['total'] >= limits['max_total']:
                adjusted_multiplier = base_multiplier
                if drawdown_tight:
                    adjusted_multiplier *= 0.85
                if drawdown_hard_stop:
                    adjusted_multiplier *= 0.8
                adjusted_multiplier = min(0.95, max(CORRELATION_OVERRIDE_MIN_SCALE, adjusted_multiplier))
                if adjusted_multiplier < group_multiplier:
                    group_multiplier = adjusted_multiplier
                reasons.append(f"total exposure {stats['total']}/{limits['max_total']}")
            elif stats['same'] > 0:
                adjusted_multiplier = base_multiplier
                if hedged:
                    adjusted_multiplier = min(0.95, adjusted_multiplier + 0.1 * min(stats['opposite'], 2))
                if drawdown_tight:
                    adjusted_multiplier *= 0.9
                if drawdown_hard_stop:
                    adjusted_multiplier *= 0.85
                adjusted_multiplier = min(0.95, max(CORRELATION_OVERRIDE_MIN_SCALE, adjusted_multiplier))
                if adjusted_multiplier < group_multiplier:
                    group_multiplier = adjusted_multiplier
                reason_text = f"existing aligned exposure {stats['same']}"
                if stats.get('same_volume'):
                    reason_text += f" ({stats['same_volume']:.2f} lots)"
                reasons.append(reason_text)

            if reasons and group_multiplier < 1.0:
                size_multiplier = min(size_multiplier, group_multiplier)
                scale_records.append({
                    "group": group,
                    "stats": stats,
                    "limits": limits,
                    "multiplier": group_multiplier,
                    "reasons": reasons,
                    "net_bias": net_bias,
                })

        global_adjustments: list[dict[str, object]] = []
        if drawdown_hard_stop:
            dd_scale = max(CORRELATION_OVERRIDE_MIN_SCALE, 0.7)
            size_multiplier = min(size_multiplier, dd_scale)
            global_adjustments.append({"label": f"drawdown {drawdown_pressure:.2f} ATR", "multiplier": dd_scale})
        elif drawdown_tight:
            dd_scale = max(CORRELATION_OVERRIDE_MIN_SCALE, 0.85)
            size_multiplier = min(size_multiplier, dd_scale)
            global_adjustments.append({"label": f"drawdown {drawdown_pressure:.2f} ATR", "multiplier": dd_scale})

        if guard_factor < 1.0:
            guard_scale = max(CORRELATION_OVERRIDE_MIN_SCALE, 0.75 + 0.25 * guard_factor)
            size_multiplier = min(size_multiplier, guard_scale)
            global_adjustments.append({"label": f"guard factor {guard_factor:.2f}", "multiplier": guard_scale})

        if size_multiplier < 1.0:
            dominant_multiplier = size_multiplier
            dominant_reason = None

            if scale_records:
                strongest_group = min(scale_records, key=lambda r: r["multiplier"])
                if abs(strongest_group["multiplier"] - dominant_multiplier) <= 1e-6:
                    dominant_reason = (
                        strongest_group["group"],
                        strongest_group["stats"],
                        strongest_group["limits"],
                        strongest_group["reasons"],
                        strongest_group["net_bias"],
                    )

            if dominant_reason is None and global_adjustments:
                strongest_global = min(global_adjustments, key=lambda r: r["multiplier"])
                print(
                    f"🔗 {symbol}: Correlation exposure tempered by {strongest_global['label']}. "
                    f"Scaling position by {size_multiplier:.0%}."
                )
            elif dominant_reason is not None:
                group, stats, limits, reasons, net_bias = dominant_reason
                reason_text = "; ".join(str(reason) for reason in reasons)
                print(
                    f"🔗 {symbol}: Correlation exposure in {group} ({stats['total']}/{limits['max_total']} total, "
                    f"{stats['same']} same, {stats['opposite']} opposite, net {net_bias:+.0f}, "
                    f"vol {stats.get('total_volume', 0.0):.2f}). Scaling position by {size_multiplier:.0%} "
                    f"({reason_text})."
                )
            else:
                print(
                    f"🔗 {symbol}: Correlation exposure active; scaling position by {size_multiplier:.0%}."
                )
            return False, size_multiplier

        print(f"🔗 {symbol}: Correlation check OK - diversified exposure across correlated groups")
        return False, 1.0  # Normal position size
            
    except Exception as e:
        print(f"Error in correlation check: {e}")
        return False, 1.0  # Default to allow if error


def evaluate_pyramiding(
    symbol: str,
    signal: str,
    positions: list,
    candidate_confidence: float,
    atr_value: float | None = None,
    guard_factor: float | None = None,
    strategy_label: str | None = None,
) -> tuple[bool, float, str]:
    """Determine whether we can stack another position in the same direction."""
    if not PYRAMID_ENABLED:
        return False, 1.0, "pyramiding disabled"

    if signal not in ("buy", "sell"):
        return False, 1.0, "invalid signal"

    direction_type = mt5.POSITION_TYPE_BUY if signal == "buy" else mt5.POSITION_TYPE_SELL
    same_direction_positions = [p for p in positions if p.type == direction_type]

    if not same_direction_positions:
        return False, 1.0, "no base position"

    if len(same_direction_positions) >= PYRAMID_MAX_ENTRIES_PER_SIDE:
        return False, 1.0, "max stacks reached"

    if candidate_confidence < PYRAMID_MIN_CONFIDENCE:
        return False, 1.0, "confidence below pyramid threshold"

    if guard_factor is not None and guard_factor < PYRAMID_GUARD_MIN_FACTOR:
        return False, 1.0, f"guard factor {guard_factor:.2f} below pyramid floor"

    total_profit = sum(float(getattr(p, "profit", 0.0) or 0.0) for p in same_direction_positions)
    losing_leg = total_profit < 0

    strategy_label_lower = (strategy_label or "").lower()
    is_mean_reversion = "mean" in strategy_label_lower and "reversion" in strategy_label_lower

    if losing_leg and len(same_direction_positions) >= PYRAMID_MAX_LOSING_POSITIONS:
        return False, 1.0, "losing stack limit reached"

    if is_mean_reversion and losing_leg and len(same_direction_positions) >= MEAN_REVERSION_MAX_LOSING_POSITIONS:
        return False, 1.0, "mean reversion losing stack cap"

    drawdown_same, drawdown_opposite, worst_drawdown_atr, total_drawdown_atr = _calculate_drawdown_pressure_atr(
        positions,
        atr_value,
        signal,
    )
    drawdown_pressure = drawdown_same if drawdown_same > 0 else max(drawdown_opposite, worst_drawdown_atr * 0.5)
    combined_draw_pressure = max(drawdown_pressure, total_drawdown_atr)

    rescue_mode = False
    if is_mean_reversion and losing_leg:
        if combined_draw_pressure >= MEAN_REVERSION_STACK_HARD_CAP_ATR:
            return False, 1.0, f"mean reversion guard: adverse {combined_draw_pressure:.2f} ATR"
        if combined_draw_pressure >= MEAN_REVERSION_STACK_DRAW_LIMIT_ATR:
            if candidate_confidence >= MEAN_REVERSION_STACK_RESCUE_CONFIDENCE and len(same_direction_positions) < MEAN_REVERSION_MAX_LOSING_POSITIONS:
                rescue_mode = True
            else:
                return False, 1.0, "mean reversion guard: waiting for stabilization"

    confidence_bonus = max(0.0, candidate_confidence - PYRAMID_MIN_CONFIDENCE)
    stack_load_ratio = len(same_direction_positions) / max(1, PYRAMID_MAX_ENTRIES_PER_SIDE)

    if losing_leg:
        base_scale = PYRAMID_LOSING_RISK_BOOST
        drawdown_scale = max(0.55, 1.0 - min(2.5, drawdown_pressure) * 0.2)
        risk_boost = base_scale * drawdown_scale
        rationale = "averaging into drawdown"
        if rescue_mode:
            risk_boost *= MEAN_REVERSION_STACK_RESCUE_SCALE
            rationale = "mean reversion rescue add"
        if drawdown_pressure > 0:
            rationale += f" ({drawdown_pressure:.2f} ATR adverse)"
    else:
        base_scale = PYRAMID_WINNING_RISK_BOOST
        confidence_scale = 1.0 + min(0.35, confidence_bonus * 0.5)
        risk_boost = base_scale * confidence_scale
        rationale = "scaling a winner"
        if confidence_bonus > 0:
            rationale += f" (confidence +{confidence_bonus:.2f})"

    stack_scale = max(0.6, 1.0 - stack_load_ratio * 0.25)
    if is_mean_reversion and losing_leg:
        stack_scale = min(stack_scale, 0.8)
    risk_boost *= stack_scale
    if stack_load_ratio > 0:
        rationale += f", stack {len(same_direction_positions)}/{PYRAMID_MAX_ENTRIES_PER_SIDE}"

    correlation_block, correlation_multiplier = should_limit_correlation_exposure(
        symbol,
        signal,
        confidence=candidate_confidence,
        guard_factor=guard_factor,
        drawdown_pressure=drawdown_pressure,
    )
    if correlation_block:
        return False, 1.0, "correlation limit"

    if correlation_multiplier < 1.0:
        risk_boost *= correlation_multiplier
        rationale += f", correlation {correlation_multiplier:.0%}"

    if drawdown_pressure > 0 and not losing_leg:
        mitigation_scale = max(0.7, 1.0 - min(2.0, drawdown_pressure) * 0.15)
        risk_boost *= mitigation_scale
        rationale += f", drawdown guard {mitigation_scale:.0%}"

    risk_boost = max(0.5, min(risk_boost, RISK_MULTIPLIER_MAX))
    return True, risk_boost, rationale


def calculate_risk_multiplier(session_priority: int, regime_weight: float, performance_weight: float) -> float:
    """Blend session quality, regime alignment, and performance into a risk multiplier."""
    multiplier = 1.0

    # Session-based boost
    if session_priority >= 5:
        multiplier += 0.15
    elif session_priority == 4:
        multiplier += 0.05

    # Regime alignment: baseline weight ~0.2. Encourage strong fits, penalize weak.
    baseline_regime = 0.2
    multiplier += (regime_weight - baseline_regime) * 0.4

    # Performance-based adjustment (win-rate/profit factor derived weight around 1.0 - 1.5)
    multiplier += (performance_weight - 1.0) * 0.3

    # Clamp to configured bounds
    multiplier = max(RISK_MULTIPLIER_MIN, min(RISK_MULTIPLIER_MAX, multiplier))
    return round(multiplier, 2)


def evaluate_liquidity_scale(spread_ratio: float | None) -> tuple[bool, float, str]:
    """Determine liquidity-based risk adjustments using spread/ATR ratio."""
    if spread_ratio is None or spread_ratio <= 0:
        return False, 1.0, ""

    # Hard block: avoid entries when spreads consume the majority of ATR
    if spread_ratio >= LIQUIDITY_SPREAD_ATR_HARD_CAP:
        return True, 0.0, (
            f"spread/ATR {spread_ratio:.2f} ≥ {LIQUIDITY_SPREAD_ATR_HARD_CAP:.2f} cap"
        )

    # Soft throttle: progressively scale down risk as spreads widen beyond comfort
    if spread_ratio > LIQUIDITY_SPREAD_ATR_SOFT_CAP:
        span = max(1e-6, LIQUIDITY_SPREAD_ATR_HARD_CAP - LIQUIDITY_SPREAD_ATR_SOFT_CAP)
        penalty = min(1.0, (spread_ratio - LIQUIDITY_SPREAD_ATR_SOFT_CAP) / span)
        scale = max(
            LIQUIDITY_MIN_SCALE,
            1.0 - penalty * LIQUIDITY_MAX_PENALTY,
        )
        return False, round(scale, 2), (
            f"scaled to x{scale:.2f} (spread/ATR {spread_ratio:.2f})"
        )

    # Bonus: when spreads are exceptionally tight relative to volatility, lean in
    if spread_ratio < LIQUIDITY_SPREAD_ATR_BONUS:
        headroom = LIQUIDITY_SPREAD_ATR_BONUS
        bonus_ratio = min(1.0, (headroom - spread_ratio) / max(headroom, 1e-6))
        scale = min(
            LIQUIDITY_MAX_BONUS,
            1.0 + bonus_ratio * LIQUIDITY_TIGHT_BONUS_MAG,
        )
        return False, round(scale, 2), (
            f"boost to x{scale:.2f} (spread/ATR {spread_ratio:.2f})"
        )

    return False, 1.0, ""


def evaluate_alpha_surge(
    session_priority: int,
    spread_ratio: float | None,
    volatility_pressure: float | None,
    candle_conviction: float | None,
) -> tuple[bool, float, float, str]:
    """Detect high-quality bursts and return risk/confidence adjustments."""

    if not ALPHA_SURGE_ENABLED:
        return False, 1.0, 0.0, ""

    if session_priority < ALPHA_SURGE_MIN_SESSION_PRIORITY:
        return False, 1.0, 0.0, ""

    if (
        volatility_pressure is None
        or candle_conviction is None
        or volatility_pressure < ALPHA_SURGE_MIN_VOL_PRESSURE
        or candle_conviction < ALPHA_SURGE_MIN_CANDLE_CONVICTION
    ):
        return False, 1.0, 0.0, ""

    if spread_ratio is None or spread_ratio <= 0 or spread_ratio > ALPHA_SURGE_MAX_SPREAD_RATIO:
        return False, 1.0, 0.0, ""

    # Passed all gates -> offer capped risk boost and confidence relief
    note = (
        f"session {session_priority}, vol {volatility_pressure:.2f}, candle {candle_conviction:.2f}, spread/ATR {spread_ratio:.2f}"
    )
    risk_scale = ALPHA_SURGE_RISK_MULTIPLIER
    confidence_relief = ALPHA_SURGE_CONFIDENCE_RELIEF
    return True, risk_scale, confidence_relief, note


def calculate_adx(df: pd.DataFrame, period: int = 14) -> float:
    """Calculate Average Directional Index (ADX) for trend strength."""
    try:
        if len(df) < period + 1:
            return None
            
        high = df['high']
        low = df['low'] 
        close = df['close']
        
        # Calculate True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        dm_plus = ((high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)).fillna(0)
        dm_minus = ((low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)).fillna(0)
        
        # Smooth the values
        tr_smooth = tr.rolling(period).mean()
        dm_plus_smooth = dm_plus.rolling(period).mean()
        dm_minus_smooth = dm_minus.rolling(period).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX and ADX
        dx = 100 * (abs(di_plus - di_minus) / (di_plus + di_minus))
        adx = dx.rolling(period).mean()
        
        return float(adx.iloc[-1]) if not pd.isna(adx.iloc[-1]) else None
    except Exception:
        return None


def detect_market_regime(df: pd.DataFrame) -> str:
    """Detect market regime: TRENDING, RANGING, or VOLATILE."""
    try:
        if len(df) < REGIME_LOOKBACK_PERIODS:
            return 'RANGING'  # Default to ranging if insufficient data
        
        # Calculate trend strength (ADX)
        adx = calculate_adx(df)
        
        # Calculate volatility (current ATR vs historical average)
        current_atr = compute_atr(df, period=14)
        historical_atr = compute_atr(df.iloc[:-10], period=14)  # ATR from 10 bars ago
        
        if current_atr is None or historical_atr is None or historical_atr == 0:
            return 'RANGING'
        
        volatility_ratio = current_atr / historical_atr
        
        # Regime classification logic
        if adx is not None and adx > TREND_THRESHOLD * 100:  # ADX > 30
            if volatility_ratio > VOLATILITY_THRESHOLD:
                return 'VOLATILE'  # High trend + high volatility = volatile trending
            else:
                return 'TRENDING'  # High trend + normal volatility = clean trend
        else:
            if volatility_ratio > VOLATILITY_THRESHOLD:
                return 'VOLATILE'  # Low trend + high volatility = choppy/volatile
            else:
                return 'RANGING'   # Low trend + normal volatility = sideways
                
    except Exception as e:
        print(f"Error detecting market regime: {e}")
        return 'RANGING'  # Safe default


def get_regime_strategy_weight(strategy_name: str, regime: str) -> float:
    """Get the weight for a strategy in the current market regime."""
    regime_weights = REGIME_STRATEGY_WEIGHTS.get(regime, REGIME_STRATEGY_WEIGHTS['RANGING'])
    return regime_weights.get(strategy_name, 0.2)  # Default 20% if strategy not found


def should_trade_strategy_in_regime(strategy_name: str, regime: str) -> bool:
    """Check if a strategy should be active in the current regime."""
    weight = get_regime_strategy_weight(strategy_name, regime)
    if weight >= 0.15:
        return True
    if AGGRESSIVE_MODE and weight >= AGGRESSIVE_REGIME_THRESHOLD:
        return True
    return False  # Default: disable very low-weight strategies


# Connect to MT5
if not mt5.initialize():
    print("initialize() failed")
    quit()

# Load strategy performance data
load_strategy_performance()
load_risk_guard_state()

initial_equity = get_account_equity_quiet()
if initial_equity > 0:
    update_risk_guard(initial_equity)
else:
    print("⚠️  Unable to prime risk guard with account equity (0 received).")

# Agent settings for each strategy - OPTIMIZED FROM BACKTESTING
agent_definitions = [
    {
        'label': 'Breakout',
        'cls': BreakoutAgent,
        'params': {'lookback': 10},
        'sl_mult': 2.5,
        'tp_mult': 3.0,
        'priority': 3,
        'atr_period': 14,
    },
    {
        'label': 'Donchian Channel',
        'cls': DonchianChannelAgent,
        'params': {'channel_length': 10},
        'sl_mult': 3.0,
        'tp_mult': 3.5,
        'priority': 4,
        'atr_period': 5,
    },
    {
        'label': 'MA Crossover',
        'cls': MACrossoverAgent,
        'params': {'fast_period': 5, 'slow_period': 40},
        'sl_mult': 2.0,
        'tp_mult': 2.0,
        'priority': 2,
        'atr_period': 5,
    },
    {
        'label': 'Momentum Trend',
        'cls': MomentumTrendAgent,
        'params': {'ma_period': 30, 'roc_period': 5},
        'sl_mult': 2.75,
        'tp_mult': 2.0,
        'priority': 5,
        'atr_period': 5,
    },
    {
        'label': 'Mean Reversion',
        'cls': MeanReversionAgent,
        'params': {'ma_period': 10, 'num_std': 1.0},
        'sl_mult': 2.75,
        'tp_mult': 4.0,
        'priority': 1,
        'atr_period': 7,
    },
]

# For each symbol, create a dict of agents
agent_dict = {
    symbol: [
        {
            'label': definition['label'],
            'agent': definition['cls'](**definition['params']),
            'sl_mult': definition.get('sl_mult', SL_ATR_MULTIPLIER),
            'tp_mult': definition.get('tp_mult', TP_ATR_MULTIPLIER),
            'priority': definition.get('priority', 999),
        }
        for definition in agent_definitions
    ]
    for symbol in ALL_SYMBOLS
}

def ensure_connection_ready() -> bool:
    """Verify the terminal connection and attempt a reinitialize if needed."""
    terminal_info = mt5.terminal_info()
    if terminal_info is None or not getattr(terminal_info, 'connected', False):
        print("MT5 terminal not connected. Attempting to reinitialize...")
        if not mt5.initialize():
            code, message = mt5.last_error()
            print(f"Reinitialize failed: {code} | {message}")
            return False
        terminal_info = mt5.terminal_info()
        if terminal_info is None or not getattr(terminal_info, 'connected', False):
            print("Unable to establish MT5 connection after reinitialize attempt.")
            return False
    return True


def prepare_symbol(symbol: str):
    """Ensure a symbol is visible/tradable and return its updated info."""
    symbol_info = mt5.symbol_info(symbol)
    if symbol_info is None or not getattr(symbol_info, 'visible', True):
        if not mt5.symbol_select(symbol, True):
            code, message = mt5.last_error()
            print(f"Unable to select {symbol}: {code} | {message}")
            return None
        symbol_info = mt5.symbol_info(symbol)

    if symbol_info is None:
        print(f"Symbol info still unavailable for {symbol} after selection.")
        return None

    trade_mode = getattr(symbol_info, 'trade_mode', None)
    if trade_mode in (mt5.SYMBOL_TRADE_MODE_DISABLED, mt5.SYMBOL_TRADE_MODE_CLOSEONLY):
        print(f"Trading disabled for {symbol} (trade_mode={trade_mode}).")
        return None

    return symbol_info


def _resolve_commission(symbol_name: str | None) -> float:
    if not symbol_name:
        return BROKER_COMMISSION_PER_LOT

    symbol_key = symbol_name
    commission = SYMBOL_COMMISSION_OVERRIDES.get(symbol_key)
    if commission is not None:
        return commission

    trimmed = symbol_key.rstrip("+")
    if trimmed and trimmed != symbol_key:
        commission = SYMBOL_COMMISSION_OVERRIDES.get(trimmed)
        if commission is not None:
            return commission

    return BROKER_COMMISSION_PER_LOT


def calculate_position_size(
    symbol_info,
    atr_value: float,
    sl_mult: float,
    risk_multiplier: float = 1.0,
    account_balance: float | None = None,
    symbol: str | None = None,
) -> float:
    """Calculate position size based on volatility and risk management."""
    if account_balance is None or account_balance <= 0:
        account_balance = get_account_balance()

    if not atr_value or atr_value <= 0:
        # Fallback to conservative fixed sizing if no ATR
        name = getattr(symbol_info, "name", "") if symbol_info is not None else ""
        if isinstance(name, str) and "XAUUSD" in name.upper():
            return 0.1
        return 0.5

    # Risk amount in account currency (bounded multiplier)
    bounded_multiplier = max(RISK_MULTIPLIER_MIN, min(RISK_MULTIPLIER_MAX, risk_multiplier))
    risk_amount = account_balance * ACCOUNT_RISK_PER_TRADE * bounded_multiplier

    # Expected loss per unit if SL is hit
    base_tick_size = getattr(symbol_info, 'point', 0.01) if symbol_info is not None else 0.01
    tick_size = getattr(symbol_info, 'trade_tick_size', base_tick_size) or base_tick_size
    tick_value = getattr(symbol_info, 'trade_tick_value', 1.0) or 1.0

    sl_distance = atr_value * sl_mult
    ticks_at_risk = sl_distance / tick_size if tick_size else 0.0
    loss_per_unit = ticks_at_risk * tick_value

    symbol_name = symbol or getattr(symbol_info, 'name', None)
    commission_per_lot = _resolve_commission(symbol_name if isinstance(symbol_name, str) else None)
    per_lot_cost = loss_per_unit + commission_per_lot

    if per_lot_cost <= 0:
        return MIN_LOT_SIZE

    # Calculate optimal position size
    optimal_size = risk_amount / per_lot_cost

    # Apply limits
    position_size = max(MIN_LOT_SIZE, min(optimal_size, MAX_LOT_SIZE))

    return round(position_size, 2)


def normalize_volume(volume: float, symbol_info) -> float:
    """Clamp and snap the requested volume to the broker's allowed step."""
    step = getattr(symbol_info, 'volume_step', 0.01) or 0.01
    min_volume = getattr(symbol_info, 'volume_min', step) or step
    max_volume = getattr(symbol_info, 'volume_max', volume) or volume
    normalized = max(min_volume, min(volume, max_volume))
    steps = round(normalized / step)
    normalized = steps * step
    decimals = len(str(step).split('.')[-1]) if '.' in str(step) else 0
    return round(normalized, decimals)


def _normalize_symbol_key(symbol: str | None) -> str:
    if not symbol:
        return ""
    return symbol.upper().replace(".", "").replace("_", "").rstrip("+")


def _get_symbol_exposure_limits(symbol: str) -> dict[str, float | None]:
    limits: dict[str, float | None] = {
        "max_total": EXPOSURE_SYMBOL_DEFAULT_MAX_TOTAL,
        "max_same": EXPOSURE_SYMBOL_DEFAULT_MAX_SAME,
    }
    override = EXPOSURE_SYMBOL_OVERRIDES.get(symbol)
    if override:
        limits.update(override)
    return limits


def _estimate_position_notional(
    symbol: str,
    volume: float,
    price: float | None = None,
    symbol_info=None,
) -> float:
    if volume <= 0:
        return 0.0

    normalized_symbol = _normalize_symbol_key(symbol)
    info = symbol_info or mt5.symbol_info(symbol)
    contract_size = float(getattr(info, "trade_contract_size", 0.0) or 0.0) if info else 0.0
    if contract_size <= 0:
        contract_size = 100000.0 if len(normalized_symbol) >= 6 else 1.0

    px = float(price or 0.0)
    if px <= 0:
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            px = float(tick.ask or tick.bid or tick.last or 0.0)
    if px <= 0 and info:
        px = float(getattr(info, "last", 0.0) or 0.0)
    if px <= 0:
        px = 1.0

    notional = abs(volume) * max(contract_size, 1.0)
    notional *= max(px, 0.0001)
    return float(notional)


def _collect_currency_notional(positions: Iterable[Any]) -> dict[str, float]:
    exposures: defaultdict[str, float] = defaultdict(float)
    info_cache: dict[str, Any] = {}

    for pos in positions or []:
        symbol = getattr(pos, "symbol", None)
        volume = float(getattr(pos, "volume", 0.0) or 0.0)
        if not symbol or volume <= 0:
            continue

        price = float(getattr(pos, "price_current", 0.0) or getattr(pos, "price_open", 0.0) or 0.0)
        symbol_key = _normalize_symbol_key(symbol)
        info = info_cache.get(symbol_key)
        if info is None:
            info = mt5.symbol_info(symbol)
            info_cache[symbol_key] = info

        notional = _estimate_position_notional(symbol, volume, price, info)
        for currency in get_symbol_currency_exposure(symbol):
            exposures[currency] += notional

    return dict(exposures)


def apply_exposure_caps(
    symbol: str,
    signal: str,
    requested_volume: float,
    price: float,
    symbol_info,
    positions: Iterable[Any],
    account_equity: float,
) -> dict[str, object]:
    result: dict[str, object] = {
        "allowed": True,
        "volume": max(0.0, float(requested_volume)),
        "reason": "",
        "warnings": [],
    }

    if not EXPOSURE_CAPS_ENABLED:
        return result

    volume = float(result["volume"])
    if volume <= 0:
        return result

    positions_list = list(positions or [])
    symbol_key = _normalize_symbol_key(symbol)
    direction_type = mt5.POSITION_TYPE_BUY if signal == "buy" else mt5.POSITION_TYPE_SELL

    total_volume = 0.0
    same_volume = 0.0
    for pos in positions_list:
        pos_symbol = getattr(pos, "symbol", None)
        if _normalize_symbol_key(pos_symbol) != symbol_key:
            continue
        pos_volume = float(getattr(pos, "volume", 0.0) or 0.0)
        if pos_volume <= 0:
            continue
        total_volume += pos_volume
        if getattr(pos, "type", None) == direction_type:
            same_volume += pos_volume

    limits = _get_symbol_exposure_limits(symbol_key)
    reasons: list[str] = []

    max_same = limits.get("max_same")
    if max_same is not None:
        available_same = max(0.0, max_same - same_volume)
        if available_same + 1e-9 < volume:
            volume = min(volume, available_same)
            reasons.append(
                f"symbol {symbol_key or symbol} directional cap {same_volume + volume:.2f}/{max_same:.2f} lots"
            )

    max_total = limits.get("max_total")
    if max_total is not None:
        available_total = max(0.0, max_total - total_volume)
        if available_total + 1e-9 < volume:
            volume = min(volume, available_total)
            reasons.append(
                f"symbol {symbol_key or symbol} total cap {total_volume + volume:.2f}/{max_total:.2f} lots"
            )

        if available_total < max(EXPOSURE_SYMBOL_MIN_REMAINDER, MIN_LOT_SIZE):
            result.update({
                "allowed": False,
                "volume": 0.0,
                "reason": f"symbol cap reached ({total_volume:.2f}/{max_total:.2f} lots)",
            })
            return result

    if volume < MIN_LOT_SIZE - 1e-6:
        block_reason = reasons[-1] if reasons else f"symbol exposure already {total_volume:.2f} lots"
        result.update({"allowed": False, "volume": 0.0, "reason": block_reason})
        return result

    equity = float(account_equity or 0.0)
    if equity <= 0:
        equity = get_account_equity_quiet()
    if equity <= 0:
        equity = 10000.0

    per_lot_notional = _estimate_position_notional(symbol, 1.0, price, symbol_info)
    per_lot_notional = max(per_lot_notional, 1.0)

    exposures_by_currency = _collect_currency_notional(positions_list)
    warnings: list[str] = []
    currency_reasons: list[str] = []
    candidate_volume = volume

    for currency in get_symbol_currency_exposure(symbol):
        multiplier = EXPOSURE_CURRENCY_OVERRIDES.get(currency, EXPOSURE_CURRENCY_EQUITY_MULTIPLIER)
        multiplier = max(float(multiplier), 1.0)
        limit = equity * multiplier
        if limit <= 0:
            continue

        existing_notional = exposures_by_currency.get(currency, 0.0)
        available_notional = max(0.0, limit - existing_notional)
        allowed_volume = available_notional / max(per_lot_notional, 1e-6)
        if allowed_volume + 1e-9 < candidate_volume:
            candidate_volume = min(candidate_volume, allowed_volume)
            currency_reasons.append(
                f"{currency} cap {min(limit, existing_notional + per_lot_notional * candidate_volume):.0f}/{limit:.0f}"
            )

        projected_notional = existing_notional + per_lot_notional * candidate_volume
        utilization = projected_notional / limit if limit else 0.0
        if utilization >= 1.0 - 1e-6:
            warnings.append(
                f"⚠️ {currency} exposure fully allocated ({projected_notional:.0f}/{limit:.0f})."
            )
        elif utilization >= EXPOSURE_CURRENCY_WARN_RATIO:
            warnings.append(
                f"⚠️ {currency} exposure {projected_notional:.0f}/{limit:.0f} ({utilization:.0%})."
            )

    if candidate_volume < MIN_LOT_SIZE - 1e-6:
        block_reason = currency_reasons[-1] if currency_reasons else "currency exposure cap"
        result.update({"allowed": False, "volume": 0.0, "reason": block_reason, "warnings": warnings})
        return result

    if candidate_volume < volume - 1e-6 and currency_reasons:
        reasons.extend(currency_reasons)

    result.update({
        "allowed": True,
        "volume": candidate_volume,
        "reason": "; ".join(reasons),
        "warnings": warnings,
    })
    return result


def resolve_filling_type(symbol_info) -> int:
    """Select a filling mode supported by the symbol, falling back safely."""
    default = getattr(mt5, 'ORDER_FILLING_IOC', 0)
    filling_mask = getattr(symbol_info, 'trade_fillings', 0) or 0
    options = []

    if filling_mask & 0x02 and hasattr(mt5, 'ORDER_FILLING_IOC'):
        options.append(mt5.ORDER_FILLING_IOC)
    if filling_mask & 0x01 and hasattr(mt5, 'ORDER_FILLING_FOK'):
        options.append(mt5.ORDER_FILLING_FOK)
    if filling_mask & 0x04 and hasattr(mt5, 'ORDER_FILLING_RETURN'):
        options.append(mt5.ORDER_FILLING_RETURN)

    if options:
        return options[0]

    mode = getattr(symbol_info, 'trade_filling_mode', None)
    if mode is not None:
        mapping = {
            0: getattr(mt5, 'ORDER_FILLING_FOK', default),
            1: getattr(mt5, 'ORDER_FILLING_IOC', default),
            2: getattr(mt5, 'ORDER_FILLING_RETURN', default),
        }
        return mapping.get(mode, default)

    return default


def describe_filling_type(filling: int) -> str:
    mapping: dict[int, str] = {}
    if hasattr(mt5, 'ORDER_FILLING_IOC'):
        mapping[mt5.ORDER_FILLING_IOC] = 'IOC'
    if hasattr(mt5, 'ORDER_FILLING_FOK'):
        mapping[mt5.ORDER_FILLING_FOK] = 'FOK'
    if hasattr(mt5, 'ORDER_FILLING_RETURN'):
        mapping[mt5.ORDER_FILLING_RETURN] = 'RETURN'
    return mapping.get(filling, f'mode#{filling}')


def sanitize_comment(comment: str | None, fallback: str = "Ultima") -> str:
    """Ensure order comments meet MT5 constraints (ASCII, <=31 chars)."""
    base = comment if comment else fallback
    ascii_comment = base.encode('ascii', errors='ignore').decode('ascii')
    ascii_comment = re.sub(r'[^A-Za-z0-9]+', '-', ascii_comment).strip('-')
    if not ascii_comment:
        ascii_comment = fallback
    max_len = 31
    if len(ascii_comment) > max_len:
        ascii_comment = ascii_comment[:max_len]
    ascii_comment = ascii_comment.rstrip('-')
    if not ascii_comment:
        ascii_comment = fallback[:max_len]
    return ascii_comment


def calculate_atr(symbol: str, timeframe=mt5.TIMEFRAME_H1, period: int = ATR_PERIOD) -> float:
    """Calculate ATR for a symbol by fetching recent price data."""
    try:
        # Get historical data for ATR calculation
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, period + 10)
        if rates is None or len(rates) < period + 1:
            return None
        
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        
        # Use existing compute_atr function
        return compute_atr(df, period)
    except Exception as e:
        print(f"Error calculating ATR for {symbol}: {e}")
        return None


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


def calculate_volatility_pressure(df: pd.DataFrame, short_period: int = 5, long_period: int = 20) -> float:
    """Measure volatility expansion or contraction via ATR ratio."""
    try:
        if df is None or len(df) < long_period + 5:
            return 1.0
        short_atr = compute_atr(df, period=short_period)
        long_atr = compute_atr(df, period=long_period)
        if short_atr is None or long_atr is None or long_atr <= 0:
            return 1.0
        ratio = short_atr / long_atr
        return max(0.5, min(1.8, float(ratio)))
    except Exception:
        return 1.0


def calculate_candle_conviction(df: pd.DataFrame) -> float:
    """Gauge strength of the last completed candle (body size vs wicks)."""
    try:
        if df is None or len(df) < 3:
            return 1.0
        last = df.iloc[-2]
        open_price = float(last['open'])
        close_price = float(last['close'])
        high = float(last['high'])
        low = float(last['low'])
        price_range = max(1e-10, high - low)
        body = abs(close_price - open_price)
        upper_wick = max(0.0, high - max(open_price, close_price))
        lower_wick = max(0.0, min(open_price, close_price) - low)
        wick_ratio = (upper_wick + lower_wick) / price_range
        body_ratio = body / price_range
        directional_bias = 1.08 if close_price > open_price else 0.95
        conviction = body_ratio * (1.0 - 0.5 * wick_ratio) * directional_bias
        return max(0.5, min(1.5, float(conviction)))
    except Exception:
        return 1.0


def calculate_signal_confidence(
    session_priority: int,
    regime_weight: float,
    performance_weight: float,
    guard_factor: float,
    signal_direction: str,
    micro_score: float | None,
    micro_threshold: float | None,
    micro_soft: bool,
    volatility_pressure: float,
    candle_conviction: float,
) -> float:
    """Blend contextual factors into a confidence score around 1.0 baseline."""
    base = 1.0

    # Regime alignment: normalize around 0.3 baseline
    regime_factor = 0.65 + 1.1 * min(1.0, regime_weight / 0.35)
    base *= regime_factor

    # Strategy performance factor (win-rate/profit factor proxy around 1.0-1.5)
    perf_factor = 0.7 + 0.3 * min(1.6, max(0.4, performance_weight))
    base *= perf_factor

    # Session quality scaling (1.0 at top-tier sessions)
    session_factor = 0.7 + 0.3 * min(1.0, max(0.4, session_priority / 5))
    base *= session_factor

    # Guard factor (drawdown recovery throttles)
    guard_factor = max(0.4, min(1.0, guard_factor))
    base *= (0.75 + 0.25 * guard_factor)

    if micro_score is not None and micro_threshold:
        alignment_raw = micro_score if signal_direction == 'buy' else -micro_score
        effective_threshold = max(1e-9, micro_threshold)
        alignment_ratio = alignment_raw / effective_threshold
        if micro_soft and alignment_ratio < 1.0:
            alignment_ratio = max(0.6, alignment_ratio)
        if alignment_ratio >= 1.0:
            micro_factor = min(1.5, 1.0 + 0.25 * (alignment_ratio - 1.0))
        elif alignment_ratio >= 0.0:
            micro_factor = 0.85 + 0.15 * alignment_ratio
        else:
            micro_factor = max(0.55, 1.0 + 0.2 * alignment_ratio)
        base *= micro_factor
    else:
        base *= 0.9  # slight penalty without micro confirmation detail

    base *= max(0.75, min(1.3, volatility_pressure))
    base *= max(0.75, min(1.25, candle_conviction))

    confidence = 0.45 + base * 0.35
    return round(float(min(1.8, confidence)), 3)


def adjust_risk_with_confidence(multiplier: float, confidence: float) -> float:
    """Tilt risk multiplier up/down based on confidence."""
    adjusted = multiplier
    if confidence >= 1.25:
        boost = min(HIGH_CONFIDENCE_BOOST_CAP, confidence + 0.1)
        adjusted *= boost
    elif confidence < 0.9:
        cut = max(0.65, confidence)
        adjusted *= cut
    return adjusted


def send_order(
    symbol,
    signal,
    lot=None,
    deviation=5,
    comment="",
    risk_multiplier: float = 1.0,
    atr: float | None = None,
    sl_mult: float | None = None,
    tp_mult: float | None = None,
    correlation_context: dict[str, float] | None = None,
):
    if not ensure_connection_ready():
        print("Aborting send_order because MT5 connection is not ready.")
        return

    # Fetch symbol details and tick once
    symbol_info = prepare_symbol(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if symbol_info is None or tick is None:
        print(f"Cannot send order for {symbol}: symbol_info or tick is None (symbol_info={symbol_info}, tick={tick})")
        return

    account_info = mt5.account_info()
    account_equity = float(getattr(account_info, "equity", 0.0) or 0.0)
    if account_equity <= 0 and account_info is not None:
        account_equity = float(getattr(account_info, "balance", 0.0) or 0.0)
    positions_snapshot = list(mt5.positions_get() or [])

    price = tick.ask if signal == 'buy' else tick.bid
    spread = (tick.ask - tick.bid) if tick else None
    point = getattr(symbol_info, 'point', None) or 0.0001
    spread_points = (spread / point) if (spread is not None and point) else float('inf')
    atr_reference = atr if atr and atr > 0 else None

    if spread is not None and point:
        if atr_reference:
            spread_ratio_snapshot = spread / atr_reference if atr_reference else None
            print(
                f"📉 {symbol}: Spread snapshot {spread_points:.1f} pts ({spread:.6f}) | "
                f"Spread/ATR {spread_ratio_snapshot:.2f}"
            )
        else:
            print(f"📉 {symbol}: Spread snapshot {spread_points:.1f} pts ({spread:.6f})")
    else:
        print(f"📉 {symbol}: Spread snapshot unavailable (tick data missing)")

    # Set lot size using volatility-based sizing
    if lot is None:
        # Calculate dynamic position size based on volatility
        atr_value = atr if atr and atr > 0 else (calculate_atr(symbol, timeframe=mt5.TIMEFRAME_H1, period=5) or 0.001)
        sl_mult_effective = sl_mult if sl_mult is not None else SL_ATR_MULTIPLIER
        bounded_multiplier = max(RISK_MULTIPLIER_MIN, min(RISK_MULTIPLIER_MAX, risk_multiplier))
        lot = calculate_position_size(
            symbol_info,
            atr_value,
            sl_mult_effective,
            risk_multiplier=bounded_multiplier,
            symbol=symbol,
        )
        print(f"🎯 {symbol}: base lot {lot:.2f} using risk multiplier {bounded_multiplier:.2f}")

        if atr_reference is None and atr_value:
            atr_reference = atr_value
        
        # Apply correlation management
        correlation_context = correlation_context or {}
        should_block, correlation_multiplier = should_limit_correlation_exposure(
            symbol,
            signal,
            confidence=correlation_context.get("confidence"),
            guard_factor=correlation_context.get("guard_factor"),
            drawdown_pressure=correlation_context.get("drawdown"),
        )
        if should_block:
            print(f"🚫 {symbol}: Trade blocked due to correlation limits")
            return
        elif correlation_multiplier < 1.0:
            lot = lot * correlation_multiplier
            print(f"📉 {symbol}: Position size reduced to {lot:.2f} lots due to correlation exposure")
            
    exposure_check = apply_exposure_caps(
        symbol=symbol,
        signal=signal,
        requested_volume=lot,
        price=price,
        symbol_info=symbol_info,
        positions=positions_snapshot,
        account_equity=account_equity,
    )

    if not exposure_check.get("allowed", True):
        reason = exposure_check.get("reason") or "exposure cap"
        print(f"🚫 {symbol}: Exposure cap blocked new order ({reason}).")
        for warning in exposure_check.get("warnings", []):
            print(warning)
        return

    capped_volume = float(exposure_check.get("volume", lot) or 0.0)
    if capped_volume <= 0:
        print(f"🚫 {symbol}: Exposure cap produced zero allowable volume; skipping order.")
        for warning in exposure_check.get("warnings", []):
            print(warning)
        return

    if abs(capped_volume - lot) >= 1e-6:
        lot = capped_volume
        reason = exposure_check.get("reason")
        if reason:
            print(f"🧮 {symbol}: Exposure cap resized order to {lot:.2f} lots ({reason}).")
        else:
            print(f"🧮 {symbol}: Exposure cap resized order to {lot:.2f} lots.")

    for warning in exposure_check.get("warnings", []):
        print(warning)

    if spread_points > SPREAD_POINTS_LIMIT:
        print(f"🚫 {symbol}: Spread {spread_points:.1f} points exceeds limit {SPREAD_POINTS_LIMIT}")
        return

    if atr_reference and atr_reference > 0 and spread is not None:
        spread_ratio = spread / atr_reference
        if spread_ratio > SPREAD_ATR_RATIO_LIMIT:
            print(f"🚫 {symbol}: Spread/ATR ratio {spread_ratio:.2f} exceeds limit {SPREAD_ATR_RATIO_LIMIT}")
            return

    lot = normalize_volume(lot, symbol_info)
    if lot < MIN_LOT_SIZE - 1e-6:
        print(f"🚫 {symbol}: Normalized volume {lot:.2f} lots below broker minimum after exposure trims.")
        return
    print(f"⚖️ {symbol}: Final order volume {lot:.2f} lots")
    order_type = mt5.ORDER_TYPE_BUY if signal == 'buy' else mt5.ORDER_TYPE_SELL

    # Compute SL/TP from ATR if available
    sl = None
    tp = None
    effective_sl_mult = sl_mult if sl_mult is not None else SL_ATR_MULTIPLIER
    effective_tp_mult = tp_mult if tp_mult is not None else TP_ATR_MULTIPLIER
    if atr is not None and atr > 0:
        sl_dist = effective_sl_mult * atr
        tp_dist = effective_tp_mult * atr
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
        print(f"{symbol} ATR={atr:.6f} | SL={sl} | TP={tp} (SLx={effective_sl_mult}, TPx={effective_tp_mult})")

    filling_type = resolve_filling_type(symbol_info)
    fill_label = describe_filling_type(filling_type)
    print(f"⚙️ {symbol}: Fill mode {fill_label}, slippage tolerance {deviation} pts")

    sanitized_comment = sanitize_comment(comment)
    if sanitized_comment != (comment or ""):
        print(f"Adjusted order comment to '{sanitized_comment}'")

    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': symbol,
        'volume': lot,
        'type': order_type,
        'price': price,
        'deviation': deviation,
        'magic': 123456,
        'comment': sanitized_comment,
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': filling_type,
    }
    # Attach SL/TP if computed
    if sl is not None:
        request['sl'] = sl
    if tp is not None:
        request['tp'] = tp
    result = None
    attempt = 0
    max_attempts = 1 + (ORDER_NO_MONEY_MAX_RETRIES if ORDER_NO_MONEY_RETRY_ENABLED else 0)
    no_money_code = getattr(mt5, "TRADE_RETCODE_NO_MONEY", 10019)

    while attempt < max_attempts:
        attempt += 1
        result = mt5.order_send(request)
        if result is None:
            code, message = mt5.last_error()
            print(f"OrderSend returned None for {symbol}. Last error: {code} | {message}")
            print(f"Request: {request}")
            break

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"OrderSend success for {symbol}: {signal} at {price} | SL={request.get('sl')} TP={request.get('tp')}")
            return

        print(f"OrderSend failed for {symbol}: {result.retcode}")
        print(f"Reason: {result.comment}")

        if (
            ORDER_NO_MONEY_RETRY_ENABLED
            and result.retcode == no_money_code
            and request['volume'] > MIN_LOT_SIZE
            and attempt < max_attempts
        ):
            new_volume = max(MIN_LOT_SIZE, request['volume'] * ORDER_NO_MONEY_REDUCTION)
            if abs(new_volume - request['volume']) < 1e-6:
                break
            new_volume = normalize_volume(new_volume, symbol_info)
            if new_volume < MIN_LOT_SIZE or new_volume == request['volume']:
                break
            request['volume'] = new_volume
            lot = new_volume
            print(f"💸 {symbol}: Retcode {result.retcode} (margin shortfall). Retrying with {new_volume:.2f} lots (attempt {attempt + 1}/{max_attempts}).")
            continue

        break

def log_manual_stop_summary(scan_counter: int, trigger: str = "Manual stop requested") -> None:
    """Emit a concise account summary when the live loop halts."""
    print(f"\n🛑 {trigger} — wrapping up after scan #{scan_counter}.")

    account_info = mt5.account_info()
    if account_info:
        margin_level_val = float(getattr(account_info, 'margin_level', 0.0) or 0.0)
        leverage_setting = getattr(account_info, 'leverage', None)
        margin_level_display = f"{margin_level_val:.1f}%" if margin_level_val > 0 else "n/a"
        leverage_display = f"{float(leverage_setting):.0f}x" if leverage_setting else "n/a"
        print(
            "💼 Final account snapshot -> "
            f"Balance: {account_info.balance:.2f} | Equity: {account_info.equity:.2f} | "
            f"Free Margin: {account_info.margin_free:.2f} | Margin Used: {account_info.margin:.2f} | "
            f"Margin Level: {margin_level_display} | Leverage: {leverage_display} | "
            f"Open PnL: {account_info.profit:.2f}"
        )
        baseline_equity = run_baseline_snapshot.get("equity")
        baseline_balance = run_baseline_snapshot.get("balance")
        if baseline_equity is not None and baseline_balance is not None:
            baseline_equity_val = float(baseline_equity)
            baseline_balance_val = float(baseline_balance)
            equity_delta = account_info.equity - baseline_equity_val
            balance_delta = account_info.balance - baseline_balance_val
            equity_pct = (equity_delta / baseline_equity_val) if baseline_equity_val else 0.0
            print(
                "📈 Final run performance -> "
                f"Equity Δ {equity_delta:+.2f} ({equity_pct:+.2%}) | Balance Δ {balance_delta:+.2f}"
            )
        else:
            print("📈 Final run performance -> baseline snapshot not captured yet.")
    else:
        print("💼 Final account snapshot unavailable (could not query MT5).")

    open_positions = mt5.positions_get() or []
    if open_positions:
        print("📂 Final open positions -> " + summarize_open_positions(open_positions))
        log_position_details(open_positions, header="📘 Final position ledger:")
    else:
        print("📂 Final open positions -> none")

    # Persist any guarded state if enabled
    try:
        save_risk_guard_state()
    except Exception as exc:
        print(f"⚠️  Failed to persist risk guard state during shutdown: {exc}")


TELEGRAM_CONTROLLER = initialize_telegram_control()


scan_counter = 0

try:
    while True:
        try:
            scan_counter += 1
            process_telegram_commands(TELEGRAM_CONTROLLER)
            enforce_kill_switch()
            if wait_if_paused(TELEGRAM_CONTROLLER):
                enforce_kill_switch()
                process_telegram_commands(TELEGRAM_CONTROLLER)
                enforce_kill_switch()
                continue
            if not ensure_connection_ready():
                print("Waiting before retrying MT5 connection check...")
                time.sleep(5)
                continue

            cycle_stats: dict[str, int] = defaultdict(int)
            now_local = datetime.now()
            now_utc = datetime.now(timezone.utc)
            print("\n" + "=" * 68)
            print(f"🔁 Scan #{scan_counter} | Local {now_local.strftime('%Y-%m-%d %H:%M:%S')} | UTC {now_utc.strftime('%H:%M:%S')}")

            account_info = mt5.account_info()
            if account_info:
                margin_level_val = float(getattr(account_info, 'margin_level', 0.0) or 0.0)
                leverage_setting = getattr(account_info, 'leverage', None)
                margin_level_display = f"{margin_level_val:.1f}%" if margin_level_val > 0 else "n/a"
                leverage_display = f"{float(leverage_setting):.0f}x" if leverage_setting else "n/a"
                print(
                    "💼 Account snapshot -> "
                    f"Balance: {account_info.balance:.2f} | Equity: {account_info.equity:.2f} | "
                    f"Free Margin: {account_info.margin_free:.2f} | Margin Used: {account_info.margin:.2f} | "
                    f"Open PnL: {account_info.profit:.2f}"
                )
                if run_baseline_snapshot["equity"] is None:
                    run_baseline_snapshot["balance"] = account_info.balance
                    run_baseline_snapshot["equity"] = account_info.equity
                    run_baseline_snapshot["timestamp"] = now_local.isoformat()
                else:
                    baseline_equity = float(run_baseline_snapshot.get("equity") or 0.0)
                    baseline_balance = float(run_baseline_snapshot.get("balance") or 0.0)
                    equity_delta = account_info.equity - baseline_equity
                    balance_delta = account_info.balance - baseline_balance
                    equity_pct = (equity_delta / baseline_equity) if baseline_equity else 0.0
                    margin_util = (account_info.margin / account_info.equity) if account_info.equity else 0.0
                    print(
                        f"📈 Run performance -> Equity Δ {equity_delta:+.2f} ({equity_pct:+.2%}) | Balance Δ {balance_delta:+.2f}"
                    )
                    print(
                        f"⚙️ Leverage usage -> Margin {account_info.margin:.2f} ({margin_util:.1%} of equity) | "
                        f"Margin level {margin_level_display} | Leverage setting {leverage_display}"
                    )
            else:
                print("💼 Account snapshot unavailable (mt5.account_info() returned None).")

            soft_guard_status = evaluate_soft_guard(
                getattr(account_info, 'balance', None) if account_info else None,
                getattr(account_info, 'equity', None) if account_info else None,
            )
        except EmergencyStop:
            log_manual_stop_summary(scan_counter)
            wait_for_kill_clear(TELEGRAM_CONTROLLER)
            continue

        margin_guard_status = evaluate_margin_guard(account_info)

        active_positions = list(mt5.positions_get() or [])

        unwind_result = execute_margin_unwind(margin_guard_status, active_positions)
        if unwind_result.get("closed_volume", 0.0) > 0:
            # Refresh account + guard snapshots after trimming exposure
            time.sleep(0.5)
            account_info = mt5.account_info()
            if account_info:
                soft_guard_status = evaluate_soft_guard(
                    getattr(account_info, 'balance', None),
                    getattr(account_info, 'equity', None),
                )
                margin_guard_status = evaluate_margin_guard(account_info)
            active_positions = list(mt5.positions_get() or [])

        if active_positions:
            print("📂 Open positions -> " + summarize_open_positions(active_positions))
            log_position_details(active_positions, header="📘 Position ledger (all symbols):")
        else:
            print("📂 Open positions -> none")

        active_symbols = get_active_symbols(now_utc)
        if not active_symbols:
            print("⚪ No active symbols configured for the current session. Sleeping for 5 minutes...")
            time.sleep(300)
            continue

        if ENABLE_WEEKEND_CRYPTO and is_weekend_trading_window(now_utc):
            print(f"🪙 Weekend crypto mode active -> {', '.join(active_symbols)}")
        else:
            print(f"🎯 Symbols in scope this cycle -> {', '.join(active_symbols)}")

        # Check if we're in an active trading session
        if not is_market_session_active(active_symbols):
            session_name = get_session_info()
            print(f"⏰ Market session filter: Skipping trading during {session_name} (low liquidity period)")
            time.sleep(60)  # Wait 1 minute before checking again
            continue
            
        session_name = get_session_info()
        print(f"\n🌍 Active trading session: {session_name}")
        print(f"Checking signals at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Monitor closed positions for performance tracking
        monitor_closed_positions()

        equity_snapshot = get_account_equity_quiet()
        if equity_snapshot > 0:
            update_risk_guard(equity_snapshot)
        else:
            print("⚠️  Account equity unavailable; keeping previous risk guard snapshot.")

        risk_guard_allowed = risk_guard_allow_trade()
        guard_factor = risk_guard_drawdown_factor()
        soft_guard_allowed = not bool(soft_guard_status.get("blocked", False))
        soft_guard_factor = float(soft_guard_status.get("throttle", 1.0) or 1.0)
        margin_guard_allowed = bool(margin_guard_status.get("allowed", True))
        margin_guard_factor = float(margin_guard_status.get("throttle", 1.0) or 1.0)

        combined_guard_factor = guard_factor
        combined_guard_factor *= soft_guard_factor if soft_guard_allowed else 0.0
        combined_guard_factor *= margin_guard_factor if margin_guard_allowed else 0.0

        guard_trading_allowed = risk_guard_allowed and soft_guard_allowed and margin_guard_allowed

        if not RISK_GUARD_ENABLED:
            print("🛡️ Risk guard disabled: no daily/weekly drawdown limits enforced this cycle.")
        elif not risk_guard_allowed:
            daily_dd = float(risk_guard_state.get("daily_drawdown", 0.0))
            weekly_dd = float(risk_guard_state.get("weekly_drawdown", 0.0))
            print(f"🛑 Risk guard: blocking new entries (daily DD {daily_dd:.1%}, weekly DD {weekly_dd:.1%}).")
        elif guard_factor < 1.0:
            daily_dd = float(risk_guard_state.get("daily_drawdown", 0.0))
            weekly_dd = float(risk_guard_state.get("weekly_drawdown", 0.0))
            print(f"🛡️ Risk guard moderation: scaling risk to {guard_factor:.0%} (daily DD {daily_dd:.1%}, weekly DD {weekly_dd:.1%}).")

        print(
            f"🧮 Guard factors -> risk {guard_factor:.2f}, soft {soft_guard_factor:.2f}, margin {margin_guard_factor:.2f}, combined {combined_guard_factor:.2f}"
        )

        if SOFT_GUARD_ENABLED:
            transition = soft_guard_status.get("transition")
            soft_dd = float(soft_guard_status.get("drawdown", 0.0) or 0.0)
            status_label = soft_guard_status.get("status", "clear")
            if transition:
                t_type = transition.get("type")
                if t_type == "block":
                    print(f"🩸 Soft equity guard: blocking new entries (DD {soft_dd:.1%} of balance).")
                elif t_type == "resume":
                    print(f"✅ Soft equity guard: trading resumed (DD {soft_dd:.1%} of balance).")
                elif t_type == "throttle":
                    throttle_val = float(transition.get("value", soft_guard_factor))
                    print(f"🩹 Soft equity guard: throttling risk to {throttle_val:.0%} (DD {soft_dd:.1%}).")
                elif t_type == "status":
                    print(f"🟡 Soft equity guard: status {status_label} (DD {soft_dd:.1%}, throttle {soft_guard_factor:.0%}).")
            elif not soft_guard_allowed:
                print(f"🩸 Soft equity guard: blocking new entries (DD {soft_dd:.1%} of balance).")
            elif status_label in ("alert", "caution"):
                indicator = "🟠" if status_label == "alert" else "🟡"
                print(f"{indicator} Soft equity guard: {status_label} mode (DD {soft_dd:.1%}, throttle {soft_guard_factor:.0%}).")

        if MARGIN_GUARD_ENABLED:
            margin_transition = margin_guard_status.get("transition")
            usage = float(margin_guard_status.get("usage", 0.0) or 0.0)
            free_ratio = float(margin_guard_status.get("free_ratio", 0.0) or 0.0)
            status_label = margin_guard_status.get("status", "clear")
            severity = float(margin_guard_status.get("severity", 0.0) or 0.0)
            severity_pct = f", severity {severity:.0%}" if severity > 0 else ""
            if margin_transition:
                t_type = margin_transition.get("type")
                if t_type == "block":
                    print(f"💥 Margin guard: blocking new entries (usage {usage:.0%}, free {free_ratio:.0%}{severity_pct}).")
                elif t_type == "resume":
                    print(f"✅ Margin guard: trading resumed (usage {usage:.0%}, free {free_ratio:.0%}{severity_pct}).")
                elif t_type == "throttle":
                    throttle_val = float(margin_transition.get("value", margin_guard_factor))
                    print(f"🧯 Margin guard: throttling risk to {throttle_val:.0%} (usage {usage:.0%}, free {free_ratio:.0%}{severity_pct}).")
                elif t_type == "status":
                    print(f"🧮 Margin guard: status {status_label} (usage {usage:.0%}, free {free_ratio:.0%}, throttle {margin_guard_factor:.0%}{severity_pct}).")
            elif not margin_guard_allowed:
                print(f"💥 Margin guard: blocking new entries (usage {usage:.0%}, free {free_ratio:.0%}{severity_pct}).")
            elif status_label == "throttle" and margin_guard_factor < 1.0:
                print(f"🧯 Margin guard: throttling risk to {margin_guard_factor:.0%} (usage {usage:.0%}, free {free_ratio:.0%}{severity_pct}).")
        
        for symbol in active_symbols:
            cycle_stats['symbols_total'] += 1
            symbol_info = prepare_symbol(symbol)
            if symbol_info is None:
                cycle_stats['symbols_failed_prepare'] += 1
                print(f"Skipping {symbol} because symbol preparation failed.")
                continue

            tick = mt5.symbol_info_tick(symbol)
            point = getattr(symbol_info, 'point', None) or 0.0
            spread = None
            spread_points = None
            spread_ratio = None
            if tick and point:
                spread = max(0.0, float(tick.ask - tick.bid))
                spread_points = spread / point if point else None

            # Check instrument-specific session priority
            if not should_trade_instrument_in_session(symbol):
                session_priority = get_instrument_session_priority(symbol, session_name)
                print(f"📊 {symbol}: Skipping (session priority {session_priority}/{MIN_SESSION_PRIORITY} for {session_name})")
                cycle_stats['symbols_skipped_session'] += 1
                continue
            else:
                session_priority = get_instrument_session_priority(symbol, session_name)
                print(f"📊 {symbol}: Trading allowed (session priority {session_priority}/{MIN_SESSION_PRIORITY} for {session_name})")

            # Check for news blackout periods
            is_blackout, news_reason = is_news_blackout_period(symbol)
            if is_blackout:
                print(f"📰 {symbol}: Skipping due to news blackout - {news_reason}")
                cycle_stats['symbols_skipped_news'] += 1
                continue

            today = datetime.now()
            from_date = today - timedelta(minutes=bars * 15)
            rates = mt5.copy_rates_range(symbol, timeframe, from_date, today)
            if rates is None:
                print(f"No data for {symbol}")
                cycle_stats['symbols_skipped_data'] += 1
                continue
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            # Compute ATR once per symbol for this cycle
            atr_value = compute_atr(df, ATR_PERIOD)
            if atr_value is not None:
                print(f"{symbol} computed ATR({ATR_PERIOD}) = {atr_value:.6f}")
            else:
                print(f"{symbol} ATR could not be computed (insufficient data).")
                cycle_stats['atr_missing'] += 1

            atr_cache_by_period: dict[int, float | None] = {ATR_PERIOD: atr_value}
            volatility_pressure = calculate_volatility_pressure(df)
            candle_conviction = calculate_candle_conviction(df)
            print(f"{symbol} Volatility pressure: {volatility_pressure:.2f} | Candle conviction: {candle_conviction:.2f}")

            if spread is not None and point:
                if atr_value and atr_value > 0:
                    spread_ratio = spread / atr_value
                    print(
                        f"📉 {symbol} Spread snapshot -> {spread_points:.1f} pts ({spread:.6f}) | "
                        f"Spread/ATR {spread_ratio:.2f}"
                    )
                else:
                    print(f"📉 {symbol} Spread snapshot -> {spread_points:.1f} pts ({spread:.6f})")
            else:
                print(f"📉 {symbol} Spread snapshot unavailable (tick data missing)")

            alpha_surge_active, alpha_risk_scale, alpha_confidence_relief, alpha_note = evaluate_alpha_surge(
                session_priority=session_priority,
                spread_ratio=spread_ratio,
                volatility_pressure=volatility_pressure,
                candle_conviction=candle_conviction,
            )
            if alpha_surge_active:
                cycle_stats['alpha_surge_active'] += 1
                print(
                    f"⚡ {symbol}: Alpha surge detected ({alpha_note}) | risk x{alpha_risk_scale:.2f}, confidence relief {alpha_confidence_relief:.2f}"
                )

            # Detect market regime for strategy optimization
            current_regime = detect_market_regime(df)
            print(f"📊 {symbol} Market Regime: {current_regime}")
            
            # Advanced stop management for existing positions
            if atr_value:
                stop_stats = manage_position_stops(symbol, atr_value)
                if stop_stats:
                    for key, value in stop_stats.items():
                        cycle_stats[key] += value
            
            positions = mt5.positions_get(symbol=symbol) or []
            if positions:
                symbol_snapshots = [build_position_snapshot(p, now_utc) for p in positions]
                detail_lines = [compose_position_line(snap, verbose=False) for snap in symbol_snapshots]
                if len(detail_lines) == 1:
                    print(f"📈 {symbol} Open position -> {detail_lines[0]}")
                else:
                    print(f"📈 {symbol} Open positions ({len(detail_lines)}):")
                    for detail in detail_lines:
                        print(f"    • {detail}")
                cycle_stats['symbols_with_open_positions'] += 1
            else:
                print(f"📈 {symbol}: no open positions")
            have_buy = any(p.type == mt5.POSITION_TYPE_BUY for p in positions)
            have_sell = any(p.type == mt5.POSITION_TYPE_SELL for p in positions)

            all_candidates = []

            for agent_info in agent_dict[symbol]:
                agent = agent_info['agent']
                strategy_name = agent_info.get('label') or type(agent).__name__.replace('Agent', '').replace('_', ' ').strip() or 'Strategy'
                strategy_key = strategy_name.lower().replace(' ', '_')
                
                # Check if strategy should be active in current regime
                if not should_trade_strategy_in_regime(strategy_key, current_regime):
                    regime_weight = get_regime_strategy_weight(strategy_key, current_regime)
                    print(f"🚫 {symbol} {strategy_name}: Disabled in {current_regime} regime (weight: {regime_weight:.1%})")
                    cycle_stats['strategies_disabled_regime'] += 1
                    continue
                
                sl_mult = agent_info.get('sl_mult', SL_ATR_MULTIPLIER)
                tp_mult = agent_info.get('tp_mult', TP_ATR_MULTIPLIER)
                priority = agent_info.get('priority', 999)
                atr_period_override = agent_info.get('atr_period', ATR_PERIOD)
                strategy_atr_value = atr_cache_by_period.get(atr_period_override)
                if strategy_atr_value is None:
                    strategy_atr_value = compute_atr(df, atr_period_override)
                    atr_cache_by_period[atr_period_override] = strategy_atr_value
                if strategy_atr_value is None:
                    print(f"🚫 {symbol} {strategy_name}: Insufficient data for ATR({atr_period_override})")
                    cycle_stats['strategies_skipped_atr'] += 1
                    continue
                
                # Adjust priority based on regime and performance (lower number = higher priority)
                regime_weight = get_regime_strategy_weight(strategy_key, current_regime)
                performance_weight = get_strategy_performance_weight(strategy_key, symbol)
                combined_weight = regime_weight * performance_weight
                regime_adjusted_priority = priority / combined_weight  # Higher weight = better priority
                base_risk_multiplier = calculate_risk_multiplier(session_priority, regime_weight, performance_weight)
                liquidity_adjusted_multiplier = base_risk_multiplier

                liquidity_blocked, liquidity_scale, liquidity_note = evaluate_liquidity_scale(spread_ratio)
                if liquidity_blocked:
                    print(
                        f"💧 {symbol} {strategy_name}: Liquidity block -> {liquidity_note}"
                    )
                    cycle_stats['liquidity_skipped'] += 1
                    continue
                if liquidity_scale != 1.0:
                    liquidity_adjusted_multiplier = max(
                        RISK_MULTIPLIER_MIN,
                        min(
                            RISK_MULTIPLIER_MAX,
                            round(base_risk_multiplier * liquidity_scale, 2),
                        ),
                    )
                    scale_type = 'boosted' if liquidity_scale > 1.0 else 'scaled'
                    counter_key = 'liquidity_boosted' if liquidity_scale > 1.0 else 'liquidity_scaled'
                    cycle_stats[counter_key] += 1
                    print(
                        f"💧 {symbol} {strategy_name}: Liquidity {scale_type} -> {liquidity_note} | risk x{base_risk_multiplier:.2f} → x{liquidity_adjusted_multiplier:.2f}"
                    )

                alpha_adjusted_multiplier = liquidity_adjusted_multiplier
                alpha_conf_threshold = CONFIDENCE_EXECUTION_THRESHOLD
                if alpha_surge_active:
                    alpha_adjusted_multiplier = min(
                        RISK_MULTIPLIER_MAX,
                        max(
                            RISK_MULTIPLIER_MIN,
                            round(liquidity_adjusted_multiplier * alpha_risk_scale, 2),
                        ),
                    )
                    alpha_conf_threshold = max(
                        ALPHA_SURGE_MIN_THRESHOLD,
                        CONFIDENCE_EXECUTION_THRESHOLD - alpha_confidence_relief,
                    )
                    cycle_stats['alpha_surge_scaled'] += 1
                    if alpha_adjusted_multiplier != liquidity_adjusted_multiplier:
                        print(
                            f"⚡ {symbol} {strategy_name}: Alpha surge scaling risk x{liquidity_adjusted_multiplier:.2f} → x{alpha_adjusted_multiplier:.2f} | threshold {CONFIDENCE_EXECUTION_THRESHOLD:.2f} → {alpha_conf_threshold:.2f}"
                        )
                
                signal = agent.get_signal(df)
                if signal not in ('buy', 'sell'):
                    cycle_stats['strategies_no_signal'] += 1
                    print(
                        f"ℹ️ {symbol} {strategy_name}: No actionable signal (raw={signal}) | "
                        f"regime weight {regime_weight:.1%} | performance {performance_weight:.2f}× | adj priority {regime_adjusted_priority:.1f}"
                    )
                    continue
                
                micro_momentum_score = None
                micro_momentum_threshold = None
                micro_soft_pass = False
                micro_override = False
                
                # Multi-timeframe confirmation
                if signal in ('buy', 'sell'):
                    mtf_confirmed = confirm_signal_with_mtf(signal, symbol)
                    mtf_bias = get_mtf_trend_bias(symbol)
                    if not mtf_confirmed:
                        if strategy_key == 'mean_reversion' and current_regime == 'RANGING':
                            print(f"🌀 {symbol} {strategy_name}: Overriding MTF bias ({mtf_bias}) in ranging regime for mean reversion signal {signal}.")
                        else:
                            print(f"🔍 {symbol} {strategy_name}: Signal {signal} rejected by MTF filter (H1 bias: {mtf_bias})")
                            continue  # Skip this signal
                    else:
                        print(f"✅ {symbol} {strategy_name}: Signal {signal} confirmed by MTF (H1 bias: {mtf_bias})")

                    reference_price = float(df['close'].iloc[-1]) if not df['close'].empty else None
                    micro_pass, micro_score, micro_threshold, micro_soft, micro_override = confirm_with_micro_momentum(
                        symbol,
                        signal,
                        current_regime,
                        strategy_atr_value,
                        reference_price,
                        positions,
                        guard_factor=combined_guard_factor,
                    )
                    if not micro_pass:
                        if strategy_key == 'mean_reversion' and current_regime == 'RANGING':
                            required_threshold = micro_threshold if signal == 'buy' else -micro_threshold
                            print(
                                f"🎯 {symbol} {strategy_name}: Micro momentum counter-trend ({micro_score:+.2%} vs required {required_threshold:.2%}); allowing fade entry in range."
                            )
                            micro_momentum_score = None
                            micro_momentum_threshold = None
                            micro_soft_pass = True
                            cycle_stats['micro_overrides'] += 1
                        else:
                            required_threshold = micro_threshold if signal == 'buy' else -micro_threshold
                            comparator = "≥" if signal == 'buy' else "≤"
                            print(
                                f"🎯 {symbol} {strategy_name}: Signal {signal} rejected by micro momentum ({micro_score:+.2%} vs required {comparator} {required_threshold:.2%})"
                            )
                            cycle_stats['signals_filtered_micro'] += 1
                            continue
                    else:
                        micro_momentum_score = micro_score
                        micro_momentum_threshold = micro_threshold
                        micro_soft_pass = micro_soft
                        threshold_display = micro_momentum_threshold
                        comparator = "≥"
                        if signal == 'sell':
                            comparator = "≤"
                            threshold_display = -micro_momentum_threshold
                        qualifiers = []
                        if micro_soft:
                            qualifiers.append("soft confirm")
                        if micro_override:
                            qualifiers.append("override")
                        qualifier_text = f" ({', '.join(qualifiers)})" if qualifiers else ""
                        print(
                            f"🎯 {symbol} {strategy_name}: Micro momentum aligned {micro_score:+.2%} {comparator} {threshold_display:.2%}{qualifier_text}"
                        )
                        if micro_override:
                            cycle_stats['micro_overrides'] += 1
                        cycle_stats['micro_confirms'] += 1
                
                if micro_momentum_score is not None:
                    qualifiers = []
                    if micro_soft_pass:
                        qualifiers.append("soft")
                    if micro_override:
                        qualifiers.append("override")
                    qualifier_suffix = f" ({'/'.join(qualifiers)})" if qualifiers else ""
                    threshold_display = micro_momentum_threshold
                    if signal == 'sell':
                        threshold_display = -micro_momentum_threshold
                    micro_text = (
                        f", micro {micro_momentum_score:+.2%}/{threshold_display:.2%}{qualifier_suffix}"
                    )
                else:
                    micro_text = ""
                confidence = calculate_signal_confidence(
                    session_priority,
                    regime_weight,
                    performance_weight,
                    combined_guard_factor,
                    signal,
                    micro_momentum_score,
                    micro_momentum_threshold,
                    micro_soft_pass,
                    volatility_pressure,
                    candle_conviction,
                )

                adjusted_risk_multiplier = adjust_risk_with_confidence(alpha_adjusted_multiplier, confidence)
                risk_multiplier = apply_risk_guard_to_multiplier(adjusted_risk_multiplier)

                dynamic_conf_threshold = alpha_conf_threshold
                if signal in ('buy', 'sell') and confidence < dynamic_conf_threshold:
                    print(
                        f"⚖️ {symbol} {strategy_name}: Confidence {confidence:.2f} below threshold {dynamic_conf_threshold:.2f}; skipping signal {signal}."
                    )
                    cycle_stats['signals_filtered_confidence'] += 1
                    signal = None

                print(
                    f"{symbol} {strategy_name} Final Signal: {signal} (regime: {regime_weight:.1%}, performance: {performance_weight:.2f}×, adj priority: {regime_adjusted_priority:.1f}, confidence {confidence:.2f}/{dynamic_conf_threshold:.2f}, risk base x{base_risk_multiplier:.2f} → liquidity x{liquidity_adjusted_multiplier:.2f} → alpha x{alpha_adjusted_multiplier:.2f} → guard x{risk_multiplier:.2f}, ATR({atr_period_override})={strategy_atr_value:.6f}{micro_text})"
                )

                if signal in ('buy', 'sell'):
                    cycle_stats['candidates_total'] += 1
                    all_candidates.append({
                        'signal': signal,
                        'label': strategy_name,
                        'priority': regime_adjusted_priority,  # Use regime-adjusted priority
                        'sl_mult': sl_mult,
                        'tp_mult': tp_mult,
                        'regime_weight': regime_weight,
                        'base_risk_multiplier': base_risk_multiplier,
                        'liquidity_risk_multiplier': liquidity_adjusted_multiplier,
                        'alpha_risk_multiplier': alpha_adjusted_multiplier,
                        'risk_multiplier': risk_multiplier,
                        'guard_factor': combined_guard_factor,
                        'micro_momentum': micro_momentum_score,
                        'micro_threshold': micro_momentum_threshold,
                        'micro_soft_pass': micro_soft_pass,
                        'atr_value': strategy_atr_value,
                        'atr_period': atr_period_override,
                        'confidence': confidence,
                        'liquidity_scale': liquidity_scale,
                        'alpha_surge': alpha_surge_active,
                        'alpha_threshold': dynamic_conf_threshold,
                    })

            if all_candidates:
                if not guard_trading_allowed:
                    cycle_stats['symbols_blocked_guard'] += 1
                    cycle_stats['signals_skipped_guard'] += len(all_candidates)
                    if not risk_guard_allowed and not soft_guard_allowed and not margin_guard_allowed:
                        reason = "risk + soft + margin guard"
                    elif not risk_guard_allowed and not soft_guard_allowed:
                        reason = "risk + soft guard"
                    elif not risk_guard_allowed and not margin_guard_allowed:
                        reason = "risk + margin guard"
                    elif not soft_guard_allowed and not margin_guard_allowed:
                        reason = "soft + margin guard"
                    elif not risk_guard_allowed:
                        reason = "risk guard"
                    elif not margin_guard_allowed:
                        reason = "margin guard"
                    else:
                        reason = "soft guard"
                    print(f"{symbol}: Guard active ({reason}), skipping {len(all_candidates)} candidate signals this cycle.")
                    continue
                # Pick the highest confidence signal, tie-breaker by priority
                best_candidate = max(all_candidates, key=lambda c: (c['confidence'], -c['priority']))
                signal_type = best_candidate['signal']
                
                if signal_type == 'buy':
                    context_atr = best_candidate.get('atr_value') or strategy_atr_value
                    if have_sell and not ALLOW_HEDGING:
                        should_exit, exit_reason = should_force_counter_exit(
                            symbol,
                            'buy',
                            best_candidate,
                            positions,
                            context_atr,
                            scan_counter,
                        )
                        if should_exit:
                            closed = close_symbol_positions(
                                symbol,
                                side='sell',
                                reason=f"{best_candidate['label']}-flip",
                            )
                            if closed > 0:
                                cycle_stats['counter_signal_exits'] += closed
                                clear_counter_signal_state(symbol)
                                positions = mt5.positions_get(symbol=symbol) or []
                                have_buy = any(p.type == mt5.POSITION_TYPE_BUY for p in positions)
                                have_sell = any(p.type == mt5.POSITION_TYPE_SELL for p in positions)
                                print(
                                    f"{symbol}: Counter-signal exit cleared {closed} sell positions ({exit_reason}); enabling buy entry."
                                )
                            else:
                                print(
                                    f"{symbol}: Counter-signal exit triggered but failed to flatten sells ({exit_reason}); skipping buy."
                                )
                                cycle_stats['signals_skipped_position_conflict'] += 1
                                continue
                        else:
                            print(
                                f"{symbol}: skipping {best_candidate['label']} buy (priority {best_candidate['priority']}) because opposite position is open"
                            )
                            cycle_stats['signals_skipped_position_conflict'] += 1
                            continue
                    elif have_buy:
                        allow_stack, risk_scale, rationale = evaluate_pyramiding(
                            symbol,
                            'buy',
                            positions,
                            best_candidate['confidence'],
                            strategy_atr_value,
                            guard_factor=combined_guard_factor,
                            strategy_label=best_candidate.get('label'),
                        )
                        if allow_stack:
                            stacked_multiplier = min(
                                RISK_MULTIPLIER_MAX,
                                best_candidate['risk_multiplier'] * risk_scale,
                            )
                            order_comment = build_order_comment(best_candidate['label'], 'buy', best_candidate['priority'])
                            context_drawdown = 0.0
                            context_atr = best_candidate.get('atr_value') or strategy_atr_value
                            if context_atr:
                                same_dd, opposite_dd, worst_dd, total_dd = _calculate_drawdown_pressure_atr(
                                    positions,
                                    context_atr,
                                    'buy',
                                )
                                context_drawdown = same_dd if same_dd > 0 else max(opposite_dd, worst_dd, total_dd)
                            correlation_context = {
                                "confidence": best_candidate['confidence'],
                                "guard_factor": combined_guard_factor,
                                "drawdown": context_drawdown,
                            }
                            print(
                                f"{symbol}: pyramiding BUY via {best_candidate['label']} (priority {best_candidate['priority']}, confidence {best_candidate['confidence']:.2f}) | {rationale} | comment {order_comment}"
                            )
                            cycle_stats['signals_executed'] += 1
                            cycle_stats['signals_executed_pyramid'] += 1
                            send_order(
                                symbol,
                                'buy',
                                comment=order_comment,
                                risk_multiplier=stacked_multiplier,
                                atr=best_candidate['atr_value'],
                                sl_mult=best_candidate['sl_mult'],
                                tp_mult=best_candidate['tp_mult'],
                                correlation_context=correlation_context,
                            )
                        else:
                            print(f"{symbol}: already in buy position, ignoring {best_candidate['label']} (priority {best_candidate['priority']})")
                            cycle_stats['signals_skipped_duplicate_side'] += 1
                    else:
                        guard_floor = best_candidate.get('guard_factor', combined_guard_factor)
                        if guard_floor is not None and guard_floor < MARGIN_ENTRY_GUARD_MIN_FACTOR:
                            print(
                                f"{symbol}: Guard factor {guard_floor:.2f} below entry floor {MARGIN_ENTRY_GUARD_MIN_FACTOR:.2f}; deferring new buy."
                            )
                            cycle_stats['signals_skipped_guard_floor'] += 1
                            continue
                        order_comment = build_order_comment(best_candidate['label'], 'buy', best_candidate['priority'])
                        context_drawdown = 0.0
                        if context_atr:
                            same_dd, opposite_dd, worst_dd, total_dd = _calculate_drawdown_pressure_atr(
                                positions,
                                context_atr,
                                'buy',
                            )
                            context_drawdown = same_dd if same_dd > 0 else max(opposite_dd, worst_dd, total_dd)
                        correlation_context = {
                            "confidence": best_candidate['confidence'],
                            "guard_factor": combined_guard_factor,
                            "drawdown": context_drawdown,
                        }
                        print(f"{symbol}: executing BUY via {best_candidate['label']} (priority {best_candidate['priority']}, confidence {best_candidate['confidence']:.2f}) | comment {order_comment}")
                        cycle_stats['signals_executed'] += 1
                        send_order(
                            symbol,
                            'buy',
                            comment=order_comment,
                            risk_multiplier=best_candidate['risk_multiplier'],
                            atr=best_candidate['atr_value'],
                            sl_mult=best_candidate['sl_mult'],
                            tp_mult=best_candidate['tp_mult'],
                            correlation_context=correlation_context,
                        )
                        have_buy = True
                elif signal_type == 'sell':
                    context_atr = best_candidate.get('atr_value') or strategy_atr_value
                    if have_buy and not ALLOW_HEDGING:
                        should_exit, exit_reason = should_force_counter_exit(
                            symbol,
                            'sell',
                            best_candidate,
                            positions,
                            context_atr,
                            scan_counter,
                        )
                        if should_exit:
                            closed = close_symbol_positions(
                                symbol,
                                side='buy',
                                reason=f"{best_candidate['label']}-flip",
                            )
                            if closed > 0:
                                cycle_stats['counter_signal_exits'] += closed
                                clear_counter_signal_state(symbol)
                                positions = mt5.positions_get(symbol=symbol) or []
                                have_buy = any(p.type == mt5.POSITION_TYPE_BUY for p in positions)
                                have_sell = any(p.type == mt5.POSITION_TYPE_SELL for p in positions)
                                print(
                                    f"{symbol}: Counter-signal exit cleared {closed} buy positions ({exit_reason}); enabling sell entry."
                                )
                            else:
                                print(
                                    f"{symbol}: Counter-signal exit triggered but failed to flatten buys ({exit_reason}); skipping sell."
                                )
                                cycle_stats['signals_skipped_position_conflict'] += 1
                                continue
                        else:
                            print(
                                f"{symbol}: skipping {best_candidate['label']} sell (priority {best_candidate['priority']}) because opposite position is open"
                            )
                            cycle_stats['signals_skipped_position_conflict'] += 1
                            continue
                    elif have_sell:
                        allow_stack, risk_scale, rationale = evaluate_pyramiding(
                            symbol,
                            'sell',
                            positions,
                            best_candidate['confidence'],
                            strategy_atr_value,
                            guard_factor=combined_guard_factor,
                            strategy_label=best_candidate.get('label'),
                        )
                        if allow_stack:
                            stacked_multiplier = min(
                                RISK_MULTIPLIER_MAX,
                                best_candidate['risk_multiplier'] * risk_scale,
                            )
                            order_comment = build_order_comment(best_candidate['label'], 'sell', best_candidate['priority'])
                            context_drawdown = 0.0
                            context_atr = best_candidate.get('atr_value') or strategy_atr_value
                            if context_atr:
                                same_dd, opposite_dd, worst_dd, total_dd = _calculate_drawdown_pressure_atr(
                                    positions,
                                    context_atr,
                                    'sell',
                                )
                                context_drawdown = same_dd if same_dd > 0 else max(opposite_dd, worst_dd, total_dd)
                            correlation_context = {
                                "confidence": best_candidate['confidence'],
                                "guard_factor": combined_guard_factor,
                                "drawdown": context_drawdown,
                            }
                            print(
                                f"{symbol}: pyramiding SELL via {best_candidate['label']} (priority {best_candidate['priority']}, confidence {best_candidate['confidence']:.2f}) | {rationale} | comment {order_comment}"
                            )
                            cycle_stats['signals_executed'] += 1
                            cycle_stats['signals_executed_pyramid'] += 1
                            send_order(
                                symbol,
                                'sell',
                                comment=order_comment,
                                risk_multiplier=stacked_multiplier,
                                atr=best_candidate['atr_value'],
                                sl_mult=best_candidate['sl_mult'],
                                tp_mult=best_candidate['tp_mult'],
                                correlation_context=correlation_context,
                            )
                        else:
                            print(f"{symbol}: already in sell position, ignoring {best_candidate['label']} (priority {best_candidate['priority']})")
                            cycle_stats['signals_skipped_duplicate_side'] += 1
                    else:
                        guard_floor = best_candidate.get('guard_factor', combined_guard_factor)
                        if guard_floor is not None and guard_floor < MARGIN_ENTRY_GUARD_MIN_FACTOR:
                            print(
                                f"{symbol}: Guard factor {guard_floor:.2f} below entry floor {MARGIN_ENTRY_GUARD_MIN_FACTOR:.2f}; deferring new sell."
                            )
                            cycle_stats['signals_skipped_guard_floor'] += 1
                            continue
                        order_comment = build_order_comment(best_candidate['label'], 'sell', best_candidate['priority'])
                        context_drawdown = 0.0
                        if context_atr:
                            same_dd, opposite_dd, worst_dd, total_dd = _calculate_drawdown_pressure_atr(
                                positions,
                                context_atr,
                                'sell',
                            )
                            context_drawdown = same_dd if same_dd > 0 else max(opposite_dd, worst_dd, total_dd)
                        correlation_context = {
                            "confidence": best_candidate['confidence'],
                            "guard_factor": combined_guard_factor,
                            "drawdown": context_drawdown,
                        }
                        print(f"{symbol}: executing SELL via {best_candidate['label']} (priority {best_candidate['priority']}, confidence {best_candidate['confidence']:.2f}) | comment {order_comment}")
                        cycle_stats['signals_executed'] += 1
                        send_order(
                            symbol,
                            'sell',
                            comment=order_comment,
                            risk_multiplier=best_candidate['risk_multiplier'],
                            atr=best_candidate['atr_value'],
                            sl_mult=best_candidate['sl_mult'],
                            tp_mult=best_candidate['tp_mult'],
                            correlation_context=correlation_context,
                        )
                        have_sell = True
                
                # Now skip any remaining lower-priority signals
                remaining_candidates = [c for c in all_candidates if c != best_candidate]
                for candidate in remaining_candidates:
                    print(
                        f"{symbol}: skipping {candidate['label']} {candidate['signal']} (conf {candidate['confidence']:.2f}, priority {candidate['priority']}) "
                        f"- beaten by {best_candidate['label']} (conf {best_candidate['confidence']:.2f}, priority {best_candidate['priority']})"
                    )
        log_cycle_summary(active_symbols, cycle_stats, scan_counter)
        cycle_sleep_deadline = time.time() + SCAN_INTERVAL_SECONDS

        _expire_kill_request_if_needed()
        process_telegram_commands(TELEGRAM_CONTROLLER)
        if not enforce_kill_switch_safe(TELEGRAM_CONTROLLER, scan_counter):
            continue

        sleep_interrupted = False
        while True:
            remaining = cycle_sleep_deadline - time.time()
            if remaining <= 0:
                break
            sleep_chunk = min(remaining, TELEGRAM_POLL_SECONDS)
            if sleep_chunk <= 0:
                sleep_chunk = min(remaining, 0.25)
            time.sleep(sleep_chunk)
            _expire_kill_request_if_needed()
            process_telegram_commands(TELEGRAM_CONTROLLER)
            if not enforce_kill_switch_safe(TELEGRAM_CONTROLLER, scan_counter):
                sleep_interrupted = True
                break

        if sleep_interrupted:
            continue
except KeyboardInterrupt:
    log_manual_stop_summary(scan_counter)
finally:
    mt5.shutdown()
