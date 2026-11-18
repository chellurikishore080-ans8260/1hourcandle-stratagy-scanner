# Created by Kishore
"""
Kishore Signals - Crypto 1-Hour Daily Open Price Action
Scanner + Backtest (Binance Vision Spot USDT).  LONG only, 1H, all times UTC.

Rules
  1. 00:00 UTC 1H candle = strong BEARISH candle
  2. 01:00 UTC 1H candle = strong BULLISH candle
  3. Buy limit at 50% of the 01:00 candle High-Low range
  4. SL slightly below the 01:00 candle low
  5. TP4 = High of the 00:00 candle, TP1-TP3 = 25/50/75% of the distance entry -> TP4
  6. 25% of the position closed at each TP, whatever is left is closed at end of the UTC day
  7. Fixed position size per trade (default Rs 1,00,000)
"""
from __future__ import annotations

import io
import re
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

APP_TITLE = "Kishore Signals"
CACHE_DIR = Path("data/cache")
VISION_URL = "https://data.binance.vision/data/spot/monthly/klines/{s}/1h/{s}-1h-{y}-{m:02d}.zip"
REST_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com"]
ONE_HOUR = pd.Timedelta(hours=1)

STABLES = {
    "USDT", "USDC", "FDUSD", "DAI", "TUSD", "USDE", "USDS", "BUSD", "USDP", "PYUSD", "USD1",
    "USDD", "GUSD", "FRAX", "LUSD", "EURC", "EUR", "RLUSD", "USDT0", "BFUSD", "XUSD", "SUSDE",
    "WBTC", "WETH", "STETH", "WSTETH", "WBETH", "WEETH", "CBBTC", "BSC-USD", "BUIDL", "USDTB",
}

FALLBACK_BASES = [
    "BTC", "ETH", "XRP", "BNB", "SOL", "TRX", "DOGE", "ADA", "LINK", "BCH", "XLM", "AVAX", "SUI",
    "HBAR", "LTC", "SHIB", "TON", "DOT", "UNI", "XMR", "AAVE", "NEAR", "ICP", "ETC", "APT", "ONDO",
    "ATOM", "POL", "ARB", "FIL", "ALGO", "VET", "OP", "INJ", "TIA", "SEI", "RENDER", "FET", "TAO",
    "PEPE", "WLD", "CRV", "ENA", "JUP", "PENDLE", "LDO", "SAND", "MANA", "GRT", "QNT", "IMX",
    "STX", "FLOKI", "BONK", "THETA", "EGLD", "AXS", "RUNE", "KAVA", "CAKE", "GALA", "CHZ", "APE",
    "NEO", "IOTA", "ZEC", "DASH", "EOS", "XTZ", "FLOW", "MKR", "SNX", "COMP", "1INCH", "ZIL",
    "ENJ", "BAT", "KSM", "ROSE", "CELO", "ANKR", "SUSHI", "YFI", "ZRX", "LRC", "DYDX", "GMX",
    "WIF", "JTO", "PYTH", "STRK", "ARKM", "ORDI", "AR", "TWT", "FTT", "NEXO", "XAUT", "PAXG",
]


# ═════════════════════════════════════════════════════════════════════════════
#  Parameters
# ═════════════════════════════════════════════════════════════════════════════
@dataclass
class Params:
    body_ratio: float = 0.60      # min body / range
    min_body_atr: float = 0.50    # min body in ATR multiples
    atr_len: int = 14
    sl_buffer: float = 0.10       # % below the 01:00 low
    tp_fracs: tuple = (0.25, 0.50, 0.75, 1.0)
    sl_first: bool = True         # same candle touches SL and TP -> assume SL first
    be_after_tp1: bool = False    # move SL to entry after TP1
    fee: float = 0.10             # % per side
    slippage: float = 0.05        # % per side
    capital: float = 100000.0     # Rs per trade (fixed)


# ═════════════════════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════════════════════
def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC").tz_localize(None)


def inr(x: float) -> str:
    sign = "-" if x < 0 else ""
    s = str(int(round(abs(x))))
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = head + "," + tail
    return f"{sign}₹{s}"


def fmt(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return ""
    return f"{x:.6g}"


def to_ms(ts: pd.Timestamp) -> int:
    return int(ts.value // 10**6)


# ═════════════════════════════════════════════════════════════════════════════
#  Data (Binance Vision + REST)
# ═════════════════════════════════════════════════════════════════════════════
COLS = ["open_time", "open", "high", "low", "close", "volume"]


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw kline rows -> OHLCV frame indexed by UTC open time.
    Handles millisecond AND microsecond timestamps (Binance Vision spot from 2025-01-01 = microseconds)."""
    df = raw.iloc[:, :6].copy()
    df.columns = COLS
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    t = df["open_time"].astype("float64").to_numpy()
    t = np.where(t > 1e14, t / 1000.0, t)
    df.index = pd.to_datetime(t, unit="ms")
    df.index.name = "open_time"
    return df.drop(columns="open_time")


def rest_klines(symbol: str, **params) -> list:
    q = {"symbol": symbol, "interval": "1h", "limit": 1000, **params}
    for host in REST_HOSTS:
        try:
            r = requests.get(host + "/api/v3/klines", params=q, timeout=20)
            if r.status_code == 200:
                return r.json()
        except requests.RequestException:
            continue
    return []


def fetch_rest_range(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    frames, cur = [], start
    while cur < end:
        raw = rest_klines(symbol, startTime=to_ms(cur), endTime=to_ms(end), limit=1000)
        if not raw:
            break
        df = normalize(pd.DataFrame(raw))
        if df.empty:
            break
        frames.append(df)
        nxt = df.index[-1] + ONE_HOUR
        if nxt <= cur:
            break
        cur = nxt
    return pd.concat(frames) if frames else pd.DataFrame(columns=COLS[1:])


def fetch_month_zip(symbol: str, y: int, m: int):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{symbol}-{y}-{m:02d}.pkl"
    if path.exists():
        try:
            return pd.read_pickle(path)
        except Exception:
            pass
    try:
        r = requests.get(VISION_URL.format(s=symbol, y=y, m=m), timeout=40)
    except requests.RequestException:
        return None
    if r.status_code != 200:
        return None
    try:
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            with z.open(z.namelist()[0]) as f:
                raw = pd.read_csv(f, header=None)
        df = normalize(raw)
    except Exception:
        return None
    try:
        df.to_pickle(path)
    except Exception:
        pass
    return df


def months_between(start: pd.Timestamp, end: pd.Timestamp):
    y, m = start.year, start.month
    while (y, m) <= (end.year, end.month):
        yield y, m
        m += 1
        if m > 12:
            y, m = y + 1, 1


def fetch_symbol(symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """1H candles for [start, end]. Completed months come from Binance Vision zips (cached on disk),
    the current / just-finished month from the REST API."""
    now = utc_now()
    cur = (now.year, now.month)
    prev = (now.year - 1, 12) if now.month == 1 else (now.year, now.month - 1)
    frames = []
    for y, m in months_between(start, end):
        df = None
        if (y, m) != cur:
            df = fetch_month_zip(symbol, y, m)
        if df is None and (y, m) in (cur, prev):
            ms = pd.Timestamp(y, m, 1)
            me = ms + pd.DateOffset(months=1)
            df = fetch_rest_range(symbol, ms, min(me, now))
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=COLS[1:])
    out = pd.concat(frames)
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.loc[(out.index >= start) & (out.index <= end)]


# ═════════════════════════════════════════════════════════════════════════════
#  Universe (CoinGecko market cap  x  Binance Spot USDT)
# ═════════════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=86400, show_spinner=False)
def get_ranking() -> list:
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/coins/markets",
            params={"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": 1,
                    "sparkline": "false"},
            timeout=25,
        )
        if r.status_code == 200:
            bases = [str(c["symbol"]).upper() for c in r.json()]
            if len(bases) > 50:
                return bases
    except Exception:
        pass
    return FALLBACK_BASES


@st.cache_data(ttl=86400, show_spinner=False)
def get_binance_bases() -> set:
    for host in REST_HOSTS:
        try:
            r = requests.get(host + "/api/v3/exchangeInfo", params={"permissions": "SPOT"}, timeout=30)
            if r.status_code == 200:
                return {s["baseAsset"] for s in r.json()["symbols"]
                        if s["quoteAsset"] == "USDT" and s["status"] == "TRADING"}
        except Exception:
            continue
    return set()


def build_universe(n: int, exclude=()) -> list:
    tradable = get_binance_bases()
    out = []
    for base in get_ranking():
        if base in STABLES:
            continue
        if tradable and base not in tradable:
            continue
        sym = base + "USDT"
        if sym in exclude or sym in out:
            continue
        out.append(sym)
        if len(out) == n:
            break
    return out


@st.cache_data(ttl=86400, show_spinner=False)
def get_universe(kind: str) -> list:
    if kind.startswith("Top 20"):
        return build_universe(20, exclude=("XAUTUSDT",)) + ["XAUTUSDT"]
    return build_universe(100)


# ═════════════════════════════════════════════════════════════════════════════
#  Strategy core (shared by scanner + backtest)
# ═════════════════════════════════════════════════════════════════════════════
def calc_atr(df: pd.DataFrame, n: int) -> pd.Series:
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def candle_strength(df: pd.DataFrame, p: Params):
    o, h, l, c = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = calc_atr(df, p.atr_len).to_numpy(float)
    body, rng = np.abs(c - o), h - l
    ratio = np.divide(body, rng, out=np.zeros(len(df)), where=rng > 0)
    strong = (ratio >= p.body_ratio) & (body >= p.min_body_atr * atr)
    return (c < o) & strong, (c > o) & strong


def make_levels(h0: float, h1: float, l1: float, p: Params):
    entry = (h1 + l1) / 2.0
    sl = l1 * (1 - p.sl_buffer / 100.0)
    tp4 = h0
    if not (tp4 > entry > sl):
        return None
    d = tp4 - entry
    tps = [entry + d * p.tp_fracs[0], entry + d * p.tp_fracs[1], entry + d * p.tp_fracs[2], tp4]
    return entry, sl, tps


def find_signals(df: pd.DataFrame, p: Params) -> list:
    n = len(df)
    if n < 30:
        return []
    bear, bull = candle_strength(df, p)
    h, l = df["high"].to_numpy(float), df["low"].to_numpy(float)
    idx, hours = df.index, df.index.hour.to_numpy()
    out = []
    for i in np.flatnonzero((hours == 0) & bear):
        if i + 1 >= n or hours[i + 1] != 1 or idx[i + 1] - idx[i] != ONE_HOUR or not bull[i + 1]:
            continue
        lv = make_levels(h[i], h[i + 1], l[i + 1], p)
        if lv is None:
            continue
        out.append({"i": int(i), "entry": lv[0], "sl": lv[1], "tps": lv[2]})
    return out


def simulate_trade(O, H, L, C, entry, sl, tps, p: Params, final: bool) -> dict:
    """Walk candles AFTER the 01:00 candle (02:00 ... 23:00).
    status: CANCELLED | EXPIRED | PENDING | OPEN | CLOSED"""
    fracs = [0.25, 0.25, 0.25, 0.25]
    filled, fill_px, fill_k = False, None, None
    remaining = [0, 1, 2, 3]
    parts = []                      # (candle_k, fraction, price, label)
    cur_sl = sl
    for k in range(len(O)):
        o, h, l = O[k], H[k], L[k]
        if not filled:
            if o <= entry:
                fill_px = o
            elif l <= entry:
                fill_px = entry
            else:
                if h >= tps[3]:     # TP4 reached before the order filled -> setup cancelled
                    return {"status": "CANCELLED"}
                continue
            filled, fill_k = True, k
        sl_level = cur_sl
        hits = [i for i in remaining if h >= tps[i]]
        sl_hit = l <= sl_level
        if sl_hit and (p.sl_first or not hits):
            parts.append((k, sum(fracs[i] for i in remaining), min(o, sl_level), "SL"))
            remaining = []
            break
        for i in hits:
            parts.append((k, fracs[i], tps[i], f"TP{i + 1}"))
            remaining.remove(i)
        if 0 in hits and p.be_after_tp1:
            cur_sl = max(cur_sl, fill_px)
        if remaining and sl_hit:
            parts.append((k, sum(fracs[i] for i in remaining), min(o, sl_level), "SL"))
            remaining = []
        if not remaining:
            break
    if not filled:
        return {"status": "EXPIRED" if final else "PENDING"}
    if remaining:
        if final:
            parts.append((len(O) - 1, sum(fracs[i] for i in remaining), float(C[-1]), "EOD"))
        else:
            return {"status": "OPEN", "fill_px": fill_px, "fill_k": fill_k, "parts": parts}
    return {"status": "CLOSED", "fill_px": fill_px, "fill_k": fill_k, "parts": parts}


def trade_return(fill_px: float, parts: list, p: Params) -> float:
    gross = sum(f * (px / fill_px - 1) for _, f, px, _ in parts)
    turnover = 1 + sum(f * px / fill_px for _, f, px, _ in parts)
    return gross - (p.fee + p.slippage) / 100.0 * turnover


# ═════════════════════════════════════════════════════════════════════════════
#  Backtest
# ═════════════════════════════════════════════════════════════════════════════
def backtest_symbol(symbol: str, df: pd.DataFrame, p: Params, start: pd.Timestamp):
    counts = {"signals": 0, "cancelled": 0, "expired": 0}
    trades = []
    if df.empty:
        return trades, counts
    n, idx, hours = len(df), df.index, df.index.hour.to_numpy()
    O, H, L, C = (df[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    for s in find_signals(df, p):
        i = s["i"]
        if idx[i] < start:
            continue
        end = i + 23                                   # the 23:00 candle
        if end >= n or hours[end] != 23 or idx[end] - idx[i] != pd.Timedelta(hours=23):
            continue                                   # incomplete day -> skip
        counts["signals"] += 1
        sl_ = slice(i + 2, end + 1)
        res = simulate_trade(O[sl_], H[sl_], L[sl_], C[sl_], s["entry"], s["sl"], s["tps"], p, final=True)
        if res["status"] == "CANCELLED":
            counts["cancelled"] += 1
            continue
        if res["status"] == "EXPIRED":
            counts["expired"] += 1
            continue
        parts, fill_px = res["parts"], res["fill_px"]
        ret = trade_return(fill_px, parts, p)
        times = idx[sl_]
        labels = [x[3] for x in parts]
        tps_hit = sum(1 for lb in labels if lb.startswith("TP"))
        pnl = ret * p.capital
        trades.append({
            "symbol": symbol,
            "date": idx[i].date(),
            "fill_time": times[res["fill_k"]],
            "exit_time": times[parts[-1][0]],
            "entry": s["entry"], "fill": fill_px, "sl": s["sl"],
            "tp1": s["tps"][0], "tp2": s["tps"][1], "tp3": s["tps"][2], "tp4": s["tps"][3],
            "path": " → ".join(labels),
            "tps_hit": tps_hit,
            "ret_pct": ret * 100.0,
            "pnl": pnl,
            "result": "✅ WIN" if pnl > 0 else "❌ LOSS",
        })
    return trades, counts


def run_backtest(symbols: list, start: pd.Timestamp, end: pd.Timestamp, p: Params, progress):
    warm_start = start - pd.Timedelta(days=3)
    all_trades, totals, failed = [], {"signals": 0, "cancelled": 0, "expired": 0}, []
    done = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(fetch_symbol, s, warm_start, end): s for s in symbols}
        for f in as_completed(futs):
            sym = futs[f]
            try:
                df = f.result()
            except Exception:
                df = pd.DataFrame(columns=COLS[1:])
            if df.empty:
                failed.append(sym)
            else:
                tr, cnt = backtest_symbol(sym, df, p, start)
                all_trades += tr
                for k in totals:
                    totals[k] += cnt[k]
            done += 1
            progress.progress(done / len(symbols), text=f"{done}/{len(symbols)}  {sym}")
    cols = ["symbol", "date", "fill_time", "exit_time", "entry", "fill", "sl", "tp1", "tp2", "tp3", "tp4",
            "path", "tps_hit", "ret_pct", "pnl", "result"]
    tr = pd.DataFrame(all_trades, columns=cols)
    if not tr.empty:
        tr = tr.sort_values("exit_time").reset_index(drop=True)
    return tr, totals, failed


def compute_stats(tr: pd.DataFrame) -> dict:
    wins, losses = tr[tr["pnl"] > 0], tr[tr["pnl"] <= 0]
    gp, gl = wins["pnl"].sum(), -losses["pnl"].sum()
    cum = tr["pnl"].cumsum()
    dd = (cum - cum.cummax().clip(lower=0)).min()
    hold = (tr["exit_time"] - tr["fill_time"]).dt.total_seconds().mean() / 3600.0
    n = len(tr)
    return {
        "trades": n, "wins": len(wins), "losses": len(losses),
        "win_rate": len(wins) / n * 100.0,
        "total_pnl": tr["pnl"].sum(),
        "avg_win": wins["pnl"].mean() if len(wins) else 0.0,
        "avg_loss": losses["pnl"].mean() if len(losses) else 0.0,
        "pf": gp / gl if gl > 0 else float("inf"),
        "expectancy": tr["pnl"].mean(),
        "max_dd": dd, "avg_hold_h": hold,
        "best": tr["pnl"].max(), "worst": tr["pnl"].min(),
        "tp_rates": [(tr["tps_hit"] >= k).mean() * 100.0 for k in (1, 2, 3, 4)],
        "sl_rate": tr["path"].str.contains("SL").mean() * 100.0,
        "eod_rate": tr["path"].str.contains("EOD").mean() * 100.0,
    }


def per_symbol_table(tr: pd.DataFrame) -> pd.DataFrame:
    g = tr.groupby("symbol")
    out = pd.DataFrame({
        "Trades": g.size(),
        "Wins": g["pnl"].apply(lambda s: int((s > 0).sum())),
        "Losses": g["pnl"].apply(lambda s: int((s <= 0).sum())),
        "Total PnL": g["pnl"].sum(),
    })
    out["Win rate %"] = (out["Wins"] / out["Trades"] * 100).round(1)
    out["Total PnL"] = out["Total PnL"].round(0)
    return out.sort_values("Total PnL", ascending=False).reset_index().rename(columns={"symbol": "Symbol"})


# ═════════════════════════════════════════════════════════════════════════════
#  Scanner
# ═════════════════════════════════════════════════════════════════════════════
def scan_symbol(symbol: str, p: Params, now: pd.Timestamp) -> dict:
    row = {"Symbol": symbol, "rank": 9}
    raw = rest_klines(symbol, limit=150)
    if not raw:
        row["Status"] = "⚠️ No data"
        return row
    df = normalize(pd.DataFrame(raw))
    if df.empty:
        row["Status"] = "⚠️ No data"
        return row
    row["Last"] = float(df["close"].iloc[-1])
    closed = df[df.index + ONE_HOUR <= now]
    today = now.normalize()
    if today not in closed.index:
        row["Status"] = "⏳ 00:00 candle forming" if now < today + ONE_HOUR else "⚠️ No data"
        row["rank"] = 8
        return row
    bear, bull = candle_strength(closed, p)
    i = closed.index.get_loc(today)
    if not bear[i]:
        row["Status"], row["rank"] = "— No setup (00:00 not strong bearish)", 7
        return row
    t1 = today + ONE_HOUR
    if t1 not in closed.index:
        row["Status"], row["rank"] = "👀 Bearish 00:00 found - waiting for 01:00 close", 2
        return row
    j = closed.index.get_loc(t1)
    if not bull[j]:
        row["Status"], row["rank"] = "— No setup (01:00 not strong bullish)", 7
        return row
    lv = make_levels(float(closed["high"].iloc[i]), float(closed["high"].iloc[j]), float(closed["low"].iloc[j]), p)
    if lv is None:
        row["Status"], row["rank"] = "— Skipped (TP4 <= entry)", 7
        return row
    entry, sl, tps = lv
    rest = closed.iloc[j + 1:]
    res = simulate_trade(rest["open"].to_numpy(float), rest["high"].to_numpy(float),
                         rest["low"].to_numpy(float), rest["close"].to_numpy(float), entry, sl, tps, p, final=False)
    st_ = res["status"]
    if st_ == "PENDING":
        row["Status"], row["rank"] = "🟡 SIGNAL - buy limit active", 0
    elif st_ == "OPEN":
        hit = sum(1 for x in res["parts"] if x[3].startswith("TP"))
        row["Status"], row["rank"] = f"🟢 In trade (TPs hit: {hit})", 1
    elif st_ == "CLOSED":
        row["Status"], row["rank"] = "🏁 Closed today: " + " → ".join(x[3] for x in res["parts"]), 3
    else:
        row["Status"], row["rank"] = "⚪ Cancelled (TP4 reached before fill)", 4
    risk, reward = (entry - sl) / entry * 100.0, (tps[3] - entry) / entry * 100.0
    row.update({
        "Entry": entry, "SL": sl, "TP1": tps[0], "TP2": tps[1], "TP3": tps[2], "TP4": tps[3],
        "Risk %": round(risk, 2), "Reward %": round(reward, 2), "R:R": round(reward / risk, 2),
        "Dist to entry %": round((row["Last"] / entry - 1) * 100.0, 2),
    })
    return row


def run_scan(symbols: list, p: Params, now: pd.Timestamp) -> pd.DataFrame:
    rows = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        for r in ex.map(lambda s: scan_symbol(s, p, now), symbols):
            rows.append(r)
    df = pd.DataFrame(rows).sort_values(["rank", "Symbol"]).drop(columns="rank")
    for c in ("Last", "Entry", "SL", "TP1", "TP2", "TP3", "TP4"):
        if c in df:
            df[c] = df[c].map(fmt)
    cols = ["Symbol", "Status", "Last", "Entry", "SL", "TP1", "TP2", "TP3", "TP4",
            "Risk %", "Reward %", "R:R", "Dist to entry %"]
    return df[[c for c in cols if c in df.columns]].reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════════
#  UI
# ═════════════════════════════════════════════════════════════════════════════
def sidebar_params():
    sb = st.sidebar
    sb.header("Universe")
    kind = sb.radio("Coins", ["Top 20 + XAUTUSDT", "Top 100 spot"])
    if sb.button("🔄 Refresh universe"):
        st.cache_data.clear()
    base = get_universe(kind)
    symbols = sb.multiselect("Symbols", base, default=base)
    extra = sb.text_input("Add symbols (comma separated)", placeholder="e.g. PEPEUSDT, WIFUSDT")
    for s in [x.strip().upper() for x in extra.split(",") if x.strip()]:
        if s not in symbols:
            symbols.append(s)

    sb.header("Strategy")
    body_ratio = sb.slider("Min body / range", 0.30, 0.95, 0.60, 0.05)
    min_body_atr = sb.slider("Min body (x ATR)", 0.0, 2.0, 0.50, 0.1)
    atr_len = sb.number_input("ATR length", 5, 50, 14)
    sl_buffer = sb.number_input("SL buffer below 01:00 low (%)", 0.0, 2.0, 0.10, 0.05)
    sb.caption("TP1 / TP2 / TP3 = % of the distance entry → TP4 (00:00 high)")
    t1 = sb.number_input("TP1 fraction", 0.05, 0.95, 0.25, 0.05)
    t2 = sb.number_input("TP2 fraction", 0.05, 0.95, 0.50, 0.05)
    t3 = sb.number_input("TP3 fraction", 0.05, 0.95, 0.75, 0.05)
    sb.header("Backtest")
    capital = sb.number_input("Rs per trade (fixed)", 1000, 100000000, 100000, 10000)
    fee = sb.number_input("Fee % per side", 0.0, 1.0, 0.10, 0.01)
    slip = sb.number_input("Slippage % per side", 0.0, 1.0, 0.05, 0.01)
    sl_first = sb.checkbox("Same candle hits SL & TP → SL first (conservative)", True)
    be = sb.checkbox("Move SL to entry after TP1", False)
    p = Params(body_ratio=body_ratio, min_body_atr=min_body_atr, atr_len=int(atr_len), sl_buffer=sl_buffer,
               tp_fracs=(t1, t2, t3, 1.0), sl_first=sl_first, be_after_tp1=be, fee=fee, slippage=slip,
               capital=float(capital))
    return symbols, p


def tab_scanner(symbols, p):
    now = utc_now()
    ist = now + pd.Timedelta(hours=5, minutes=30)
    st.caption(f"UTC now: {now:%Y-%m-%d %H:%M}  |  IST: {ist:%H:%M}  |  00:00 UTC candle = 05:30 IST, "
               f"01:00 UTC candle closes at 06:30 IST")
    if st.button("🔍 Scan now", type="primary"):
        with st.spinner(f"Scanning {len(symbols)} symbols..."):
            st.session_state["scan"] = (utc_now(), run_scan(symbols, p, utc_now()))
    if "scan" in st.session_state:
        ts, df = st.session_state["scan"]
        st.write(f"Last scan: {ts:%Y-%m-%d %H:%M} UTC")
        c1, c2, c3 = st.columns(3)
        c1.metric("🟡 Active signals", int(df["Status"].str.startswith("🟡").sum()))
        c2.metric("🟢 In trade", int(df["Status"].str.startswith("🟢").sum()))
        c3.metric("👀 Waiting for 01:00", int(df["Status"].str.startswith("👀").sum()))
        st.dataframe(df, hide_index=True)


def tab_backtest(symbols, p):
    c1, c2 = st.columns(2)
    start = c1.date_input("Start date", value=(utc_now() - pd.Timedelta(days=365)).date())
    end = c2.date_input("End date", value=utc_now().date())
    st.caption(f"{len(symbols)} symbols  |  fixed {inr(p.capital)} per trade  |  first run downloads data "
               f"(cached in data/cache afterwards)")
    if st.button("▶️ Run backtest", type="primary"):
        if not symbols:
            st.warning("Select at least one symbol.")
            return
        prog = st.progress(0.0, text="Downloading data...")
        tr, totals, failed = run_backtest(symbols, pd.Timestamp(start), pd.Timestamp(end) + pd.Timedelta(hours=23),
                                          p, prog)
        prog.empty()
        st.session_state["bt"] = {"tr": tr, "totals": totals, "failed": failed, "p": p}
    bt = st.session_state.get("bt")
    if not bt:
        return
    tr, totals, failed, bp = bt["tr"], bt["totals"], bt["failed"], bt["p"]
    if failed:
        st.warning(f"No data for: {', '.join(sorted(failed))}")
    if tr.empty:
        st.info("No trades in this range / symbols.")
        return
    s = compute_stats(tr)

    st.subheader("Result")
    a = st.columns(5)
    a[0].metric("Total trades", s["trades"])
    a[1].metric("✅ Successful (wins)", s["wins"])
    a[2].metric("❌ Failed (losses)", s["losses"])
    a[3].metric("Win rate", f"{s['win_rate']:.1f}%")
    a[4].metric("Total PnL", inr(s["total_pnl"]))
    b = st.columns(5)
    b[0].metric("Profit factor", "∞" if np.isinf(s["pf"]) else f"{s['pf']:.2f}")
    b[1].metric("Expectancy / trade", inr(s["expectancy"]))
    b[2].metric("Avg win", inr(s["avg_win"]))
    b[3].metric("Avg loss", inr(s["avg_loss"]))
    b[4].metric("Max drawdown", inr(s["max_dd"]))
    d = st.columns(6)
    for k in range(4):
        d[k].metric(f"TP{k + 1} reached", f"{s['tp_rates'][k]:.0f}%")
    d[4].metric("Trades with SL", f"{s['sl_rate']:.0f}%")
    d[5].metric("EOD exits", f"{s['eod_rate']:.0f}%")
    st.caption(f"Signals: {totals['signals']}  |  filled: {s['trades']}  |  cancelled (TP4 before fill): "
               f"{totals['cancelled']}  |  never filled: {totals['expired']}  |  best trade: {inr(s['best'])}  |  "
               f"worst trade: {inr(s['worst'])}  |  avg holding: {s['avg_hold_h']:.1f}h")

    cum = tr["pnl"].cumsum()
    fig = go.Figure(go.Scatter(x=tr["exit_time"], y=cum, mode="lines", name="Cumulative PnL"))
    fig.update_layout(title="Cumulative PnL (Rs)", height=350, margin=dict(t=40, b=10))
    st.plotly_chart(fig)

    m1, m2 = st.columns(2)
    mon = tr.assign(month=tr["exit_time"].dt.to_period("M").astype(str)).groupby("month")["pnl"].sum()
    fm = go.Figure(go.Bar(x=mon.index, y=mon.values, marker_color=np.where(mon.values >= 0, "#16a34a", "#dc2626")))
    fm.update_layout(title="Monthly PnL (Rs)", height=320, margin=dict(t=40, b=10))
    m1.plotly_chart(fm)
    ps = per_symbol_table(tr)
    fs = go.Figure(go.Bar(x=ps["Symbol"], y=ps["Total PnL"],
                          marker_color=np.where(ps["Total PnL"] >= 0, "#16a34a", "#dc2626")))
    fs.update_layout(title="PnL per symbol (Rs)", height=320, margin=dict(t=40, b=10))
    m2.plotly_chart(fs)

    st.subheader("Per symbol")
    st.dataframe(ps, hide_index=True)

    st.subheader("Trade log")
    log = tr.rename(columns={
        "symbol": "Symbol", "date": "Day", "fill_time": "Filled", "exit_time": "Exited", "entry": "Entry",
        "fill": "Fill price", "sl": "SL", "tp1": "TP1", "tp2": "TP2", "tp3": "TP3", "tp4": "TP4",
        "path": "Exit path", "tps_hit": "TPs hit", "ret_pct": "Return %", "pnl": "PnL (Rs)", "result": "Result"})
    log["Return %"] = log["Return %"].round(2)
    log["PnL (Rs)"] = log["PnL (Rs)"].round(0)
    st.dataframe(log.iloc[::-1], hide_index=True)
    st.download_button("⬇️ Download trade log (CSV)", log.to_csv(index=False).encode(),
                       "kishore_signals_trades.csv", "text/csv")


def tab_rules():
    st.markdown("""
**Kishore Signals – Crypto 1-Hour Daily Open Price Action (long only)**

1. 00:00 UTC 1H candle must be a **strong bearish** candle (body ≥ 60% of range and ≥ 0.5×ATR by default).
2. 01:00 UTC 1H candle must be a **strong bullish** candle.
3. **Entry:** buy limit at 50% of the 01:00 candle's High-Low. Valid until the end of that UTC day.
4. **SL:** slightly below the 01:00 candle low.
5. **TP4:** High of the 00:00 candle. TP1-TP3 are 25/50/75% of the distance entry → TP4. 25% of the position is closed at each TP.
6. Unfilled order is cancelled if price reaches TP4 first. Anything still open is closed at the end of the UTC day (23:00 candle close).
7. Every trade uses a **fixed Rs per trade** amount; PnL = amount × net return (fees + slippage included).

**Backtest assumptions:** the order can first fill on the 02:00 candle; if a candle touches both SL and a TP the SL is assumed first (switchable); simultaneous trades in different coins are all taken independently.

**Data:** Binance Vision monthly 1H spot klines (ms and µs timestamps handled), the current month from the Binance REST API.
Spot candles differ slightly from perpetual futures candles, so results will not match TradingView exactly.

*Not financial advice. Backtests do not guarantee future results.*
""")


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📈", layout="wide")
    st.title(f"📈 {APP_TITLE}")
    st.caption("1H Daily Open Price Action - Binance Spot USDT scanner + backtest")
    symbols, p = sidebar_params()
    t1, t2, t3 = st.tabs(["🔍 Scanner", "📊 Backtest", "ℹ️ Rules"])
    with t1:
        tab_scanner(symbols, p)
    with t2:
        tab_backtest(symbols, p)
    with t3:
        tab_rules()


if __name__ == "__main__":
    main()
