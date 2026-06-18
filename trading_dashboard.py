"""

RUN:
    streamlit run trading_dashboard.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import requests
import pandas as pd
import numpy as np
import time
import logging
import random
import warnings
from collections import Counter
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()

# ── Suppress third-party deprecation noise ─────────────────────────────────────
# yfinance uses pd.Timestamp.utcnow() which is deprecated in Pandas 2.x.
# This is an upstream yfinance bug — suppress until they patch it.
warnings.filterwarnings("ignore", message=".*utcnow.*",        category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Pandas4Warning.*",category=Warning)
warnings.filterwarnings("ignore", message=".*utcnow.*",        category=DeprecationWarning)
# Suppress pandas internal performance and style warnings
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
# Catch-all for any remaining yfinance/pandas timestamp noise
import logging as _log
_log.getLogger("yfinance").setLevel(_log.ERROR)

# yfinance — required
import yfinance as yf

# tradingview-ta — optional (MA fallback if missing)
try:
    from tradingview_ta import TA_Handler, Interval
    _TV_TA = True
except ImportError:
    _TV_TA = False
    logging.warning("tradingview-ta not installed — MA(20) fallback active for all assets.")

# fredapi — optional (manual CB table fallback if missing)
try:
    from fredapi import Fred
    _FREDAPI = True
except ImportError:
    _FREDAPI = False
    logging.warning("fredapi not installed — manual CB stance table will be used.")

# streamlit-autorefresh — optional (dashboard timed auto-refresh)
try:
    from streamlit_autorefresh import st_autorefresh
    _AUTOREFRESH = True
except ImportError:
    _AUTOREFRESH = False
    logging.warning("streamlit-autorefresh not installed — auto-refresh disabled.")

# cot_reports — optional (CFTC Commitment of Traders data)
try:
    from cot_reports import cot_year
    _COT_LIB = True
except ImportError:
    _COT_LIB = False
    logging.warning("cot_reports not installed — COT section will be hidden. pip install cot_reports")


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────

FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")

# ── Scoring weights (must sum to 1.0) ────────────────────────────────────────
WEIGHTS = {"fundamental": 0.40, "technical": 0.35, "sentiment": 0.25}

# ── Cache TTLs (seconds) ─────────────────────────────────────────────────────
TTL_TECH = 1800    # 30 min
TTL_SENT = 3600    # 1 hour
TTL_FUND = 86400   # 24 hours

# ── Retry config ──────────────────────────────────────────────────────────────
RETRY_DELAYS = [1, 3, 10]

# ── Auto-refresh interval ──────────────────────────────────────────────────────
AUTO_REFRESH_INTERVAL_SEC: int = 1800   # 30 minutes

# ── Bias → percentage mapping (for weighted scoring) ─────────────────────────
BIAS_PCT = {"Bullish": 100.0, "Neutral": 50.0, "Bearish": 0.0}

# ── Status thresholds ────────────────────────────────────────────────────────
STATUS_THRESHOLDS = [
    (75.0, "READY"),
    (60.0, "WATCH"),
    (55.0, "WAIT"),
    (0.0,  "AVOID"),
]

# ── Risk-regime currency groupings ───────────────────────────────────────────
RISK_ON_CCY  = {"AUD", "NZD", "CAD"}    # Benefit when mood = RISK_ON
RISK_OFF_CCY = {"USD", "JPY", "CHF"}    # Benefit when mood = RISK_OFF
CURRENCIES   = ["USD", "EUR", "GBP", "AUD", "JPY", "CAD", "NZD", "CHF"]


# ─────────────────────────────────────────────────────────────────────────────
# ASSET REGISTRY
# ─────────────────────────────────────────────────────────────────────────────
# tv_scr: TradingView screener id
# tv_exch: TradingView exchange id
# tv_sym: TradingView symbol (may differ from Yahoo ticker)
# base/quote: currency pair components (None for non-forex)
# ig_id: IG Labs market ID for client sentiment

ASSETS = [
    # ── Forex Majors ─────────────────────────────────────────────────────────
    {
        "symbol": "EURUSD=X", "name": "EUR/USD", "cls": "Forex",
        "tv_sym": "EURUSD",   "tv_exch": "FX",      "tv_scr": "forex",
        "base": "EUR", "quote": "USD",
        "ig_id": "CS.D.EURUSD.CFD.IP",
    },
    {
        "symbol": "GBPUSD=X", "name": "GBP/USD", "cls": "Forex",
        "tv_sym": "GBPUSD",   "tv_exch": "FX",      "tv_scr": "forex",
        "base": "GBP", "quote": "USD",
        "ig_id": "CS.D.GBPUSD.CFD.IP",
    },
    {
        "symbol": "AUDUSD=X", "name": "AUD/USD", "cls": "Forex",
        "tv_sym": "AUDUSD",   "tv_exch": "FX",      "tv_scr": "forex",
        "base": "AUD", "quote": "USD",
        "ig_id": "CS.D.AUDUSD.CFD.IP",
    },
    {
        "symbol": "USDJPY=X", "name": "USD/JPY", "cls": "Forex",
        "tv_sym": "USDJPY",   "tv_exch": "FX",      "tv_scr": "forex",
        "base": "USD", "quote": "JPY",
        "ig_id": "CS.D.USDJPY.CFD.IP",
    },
    {
        "symbol": "USDCAD=X", "name": "USD/CAD", "cls": "Forex",
        "tv_sym": "USDCAD",   "tv_exch": "FX",      "tv_scr": "forex",
        "base": "USD", "quote": "CAD",
        "ig_id": "CS.D.USDCAD.CFD.IP",
    },
    {
        "symbol": "NZDUSD=X", "name": "NZD/USD", "cls": "Forex",
        "tv_sym": "NZDUSD",   "tv_exch": "FX",      "tv_scr": "forex",
        "base": "NZD", "quote": "USD",
        "ig_id": "CS.D.NZDUSD.CFD.IP",
    },
    {
        "symbol": "USDCHF=X", "name": "USD/CHF", "cls": "Forex",
        "tv_sym": "USDCHF",   "tv_exch": "FX",      "tv_scr": "forex",
        "base": "USD", "quote": "CHF",
        "ig_id": "CS.D.USDCHF.CFD.IP",
    },
    # ── Indices ───────────────────────────────────────────────────────────────
    {
        "symbol": "SPY",  "name": "SPX 500", "cls": "Index",
        "tv_sym": "SPY",  "tv_exch": "AMEX",   "tv_scr": "america",
        "base": None, "quote": None,
        "ig_id": "CS.D.SPX500.CFD.IP",
    },
    {
        "symbol": "QQQ",  "name": "NAS 100", "cls": "Index",
        "tv_sym": "QQQ",  "tv_exch": "NASDAQ", "tv_scr": "america",
        "base": None, "quote": None,
        "ig_id": "CS.D.NASDAQ.CFD.IP",
    },
    {
        "symbol": "DIA",  "name": "US 30",   "cls": "Index",
        "tv_sym": "DIA",  "tv_exch": "AMEX",   "tv_scr": "america",
        "base": None, "quote": None,
        "ig_id": "CS.D.DOW.CFD.IP",
    },
    {
        "symbol": "^GDAXI", "name": "DAX 40", "cls": "Index",
        "tv_sym": "GER40",  "tv_exch": "XETR",   "tv_scr": "germany",
        "base": None, "quote": None,
        "ig_id": None,
    },
    # ── Commodities ───────────────────────────────────────────────────────────
    {
        "symbol": "GC=F", "name": "Gold",    "cls": "Commodity",
        "tv_sym": "GOLD", "tv_exch": "TVC",    "tv_scr": "cfd",
        "base": None, "quote": None,
        "ig_id": "CS.D.GOLD.CFD.IP",
    },
    {
        "symbol": "SI=F",   "name": "Silver",  "cls": "Commodity",
        "tv_sym": "SILVER", "tv_exch": "TVC",    "tv_scr": "cfd",
        "base": None, "quote": None,
        "ig_id": "CS.D.SILVER.CFD.IP",
    },
    {
        "symbol": "CL=F",  "name": "Oil WTI", "cls": "Commodity",
        "tv_sym": "USOIL", "tv_exch": "TVC",    "tv_scr": "cfd",
        "base": None, "quote": None,
        "ig_id": "CS.D.OIL.CFD.IP",
    },
    # ── Crypto ────────────────────────────────────────────────────────────────
    {
        "symbol": "BTC-USD",  "name": "Bitcoin", "cls": "Crypto",
        "tv_sym": "BTCUSDT",  "tv_exch": "BINANCE", "tv_scr": "crypto",
        "base": None, "quote": None,
        "ig_id": "CS.D.BITCOIN.CFD.IP",
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# CENTRAL BANK MANUAL STANCE TABLE
# Update after each CB meeting. Used when FRED data is unavailable.
# ─────────────────────────────────────────────────────────────────────────────
CB_MANUAL_STANCES = {
    "USD": ("Hawkish",  "Fed on hold — rates elevated, cuts deferred"),
    "EUR": ("Neutral",  "ECB cutting cycle pausing, data-dependent"),
    "GBP": ("Neutral",  "BoE cautious, gradual cuts ahead"),
    "JPY": ("Hawkish",  "BoJ normalising, slow pace of hikes"),
    "AUD": ("Dovish",   "RBA easing, soft growth outlook"),
    "CAD": ("Dovish",   "BoC cutting aggressively on weak GDP"),
    "NZD": ("Dovish",   "RBNZ aggressive rate cuts"),
    "CHF": ("Dovish",   "SNB cutting to manage CHF strength"),
}

# FRED series IDs for live CB rate data (add more as needed)
FRED_CB_SERIES = {
    "USD": "DFF",     # Effective Federal Funds Rate
    "EUR": "ECBDFR",  # ECB Deposit Facility Rate
}

CB_STANCE_TO_BIAS = {"Hawkish": "Bullish", "Dovish": "Bearish", "Neutral": "Neutral"}


# ─────────────────────────────────────────────────────────────────────────────
# INSTITUTIONAL COLOUR PALETTE
# ─────────────────────────────────────────────────────────────────────────────
# All UI colour decisions flow from this single source of truth.

_C = {
    "bg":          "#0d1117",
    "surface":     "#161b22",
    "surface2":    "#1c2128",
    "border":      "#30363d",
    "border_sub":  "#21262d",
    "text_pri":    "#e6edf3",
    "text_sec":    "#8b949e",
    "text_ter":    "#484f58",
    "bullish":     "#3fb950",
    "bullish_bg":  "#0d1f0f",
    "bearish":     "#f85149",
    "bearish_bg":  "#1f0d0d",
    "neutral":     "#8b949e",
    "accent":      "#1f6feb",
    "ready_text":  "#3fb950",
    "ready_bg":    "#0d2314",
    "watch_text":  "#d29922",
    "watch_bg":    "#271d04",
    "wait_text":   "#8b949e",
    "wait_bg":     "#1c2128",
    "avoid_text":  "#f85149",
    "avoid_bg":    "#2d1015",
}

# Snapshot of the dark palette — used to restore when switching light → dark
_C_DARK: dict = dict(_C)

# Light / professional-white palette
_CL: dict = {
    "bg":          "#f6f8fa",
    "surface":     "#ffffff",
    "surface2":    "#f0f2f5",
    "border":      "#d0d7de",
    "border_sub":  "#e8ecf0",
    "text_pri":    "#24292f",
    "text_sec":    "#57606a",
    "text_ter":    "#8c959f",
    "bullish":     "#1a7f37",
    "bullish_bg":  "#dafbe1",
    "bearish":     "#cf222e",
    "bearish_bg":  "#ffebe9",
    "neutral":     "#57606a",
    "accent":      "#0969da",
    "ready_text":  "#1a7f37",
    "ready_bg":    "#dafbe1",
    "watch_text":  "#7d4e00",
    "watch_bg":    "#fff8c5",
    "wait_text":   "#57606a",
    "wait_bg":     "#f0f2f5",
    "avoid_text":  "#cf222e",
    "avoid_bg":    "#ffebe9",
}

# Kept for compatibility with any internal references
STATUS_STYLE = {
    "READY": (f"color:{_C['ready_text']};background:{_C['ready_bg']};"),
    "WATCH": (f"color:{_C['watch_text']};background:{_C['watch_bg']};"),
    "WAIT":  (f"color:{_C['wait_text']} ;background:{_C['wait_bg']} ;"),
    "AVOID": (f"color:{_C['avoid_text']};background:{_C['avoid_bg']};"),
}
BIAS_COLOR = {
    "Bullish": _C["bullish"],
    "Bearish": _C["bearish"],
    "Neutral": _C["neutral"],
}


# ═════════════════════════════════════════════════════════════════════════════
# UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

class RateLimitError(Exception):
    """Raised when a remote API returns HTTP 429 — do not retry."""


def retry(func, *args, delays=RETRY_DELAYS, **kwargs):
    """
    Exponential-backoff retry wrapper.
    RateLimitError (429) is re-raised immediately — no retries, fall through to
    the next fallback rather than hammering a rate-limited endpoint.
    """
    last_exc = None
    for attempt, delay in enumerate(delays):
        try:
            return func(*args, **kwargs)
        except RateLimitError:
            raise                        # propagate immediately, never retry 429
        except Exception as exc:
            last_exc = exc
            if attempt < len(delays) - 1:
                time.sleep(delay)
    raise last_exc


# ═════════════════════════════════════════════════════════════════════════════
# PILLAR 1 — TECHNICAL BIAS (TradingView TA + MA fallback)
# ═════════════════════════════════════════════════════════════════════════════

def _tv_recommendation_to_bias(rec: str):
    """Map TradingView RECOMMENDATION string → (bias, confidence)."""
    mapping = {
        "STRONG_BUY":  ("Bullish", "Strong"),
        "BUY":         ("Bullish", "Moderate"),
        "NEUTRAL":     ("Neutral", "Weak"),
        "SELL":        ("Bearish", "Moderate"),
        "STRONG_SELL": ("Bearish", "Strong"),
    }
    return mapping.get(rec, ("Neutral", "Weak"))


# ── Multi-Timeframe intervals ─────────────────────────────────────────────────
# Defined here so they're available to all functions.
# Populated only when tradingview-ta is installed.
if _TV_TA:
    _TV_INTERVALS: dict = {
        "1H":    Interval.INTERVAL_1_HOUR,
        "4H":    Interval.INTERVAL_4_HOURS,
        "Daily": Interval.INTERVAL_1_DAY,
    }


def _fetch_tv_mtf(tv_sym: str, tv_exch: str, tv_scr: str) -> dict:
    """
    Fetch TradingView recommendations across three timeframes: 1H, 4H, Daily.
    A 0.8–1.5 s random jitter is applied between each individual TF call to
    stay well under TradingView's per-session rate limit.

    Raises RateLimitError immediately on any HTTP 429 — caller trips the
    circuit breaker and skips TV entirely for all remaining assets this session.

    Returns: {"1H": "BUY", "4H": "NEUTRAL", "Daily": "STRONG_BUY", ...}
    """
    results: dict = {}
    for tf_label, interval in _TV_INTERVALS.items():
        time.sleep(random.uniform(0.8, 1.5))   # polite inter-TF jitter
        try:
            handler = TA_Handler(
                symbol=tv_sym,
                exchange=tv_exch,
                screener=tv_scr,
                interval=interval,
            )
            results[tf_label] = handler.get_analysis().summary["RECOMMENDATION"]
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "rate limit" in msg.lower():
                raise RateLimitError(f"TradingView 429 for {tv_sym} @ {tf_label}") from exc
            # Non-429 failure on one TF → record Neutral, keep going
            logging.warning("TV TA failed for %s @ %s: %s", tv_sym, tf_label, exc)
            results[tf_label] = "NEUTRAL"
    return results


def _mtf_consensus(recs: dict) -> tuple[str, str, str]:
    """
    Vote across all fetched timeframes to derive a consensus bias.

    Rules:
      • 3/3 agree Bullish  → Bullish, Strong
      • 2/3 agree Bullish  → Bullish, Moderate
      • 3/3 agree Bearish  → Bearish, Strong
      • 2/3 agree Bearish  → Bearish, Moderate
      • Any other split    → Neutral, Weak  (no consensus)

    Returns: (bias, confidence, detail_string)
    """
    bias_per_tf = {tf: _tv_recommendation_to_bias(rec)[0] for tf, rec in recs.items()}
    detail      = " | ".join(f"{tf}: {b}" for tf, b in bias_per_tf.items())
    biases      = list(bias_per_tf.values())

    bull_count = biases.count("Bullish")
    bear_count = biases.count("Bearish")
    total      = len(biases)

    if bull_count >= 2:
        conf = "Strong" if bull_count == total else "Moderate"
        return "Bullish", conf, f"{detail} → {bull_count}/{total} Bullish"
    if bear_count >= 2:
        conf = "Strong" if bear_count == total else "Moderate"
        return "Bearish", conf, f"{detail} → {bear_count}/{total} Bearish"
    return "Neutral", "Weak", f"{detail} → No consensus"


# ── ATR thresholds by asset class ─────────────────────────────────────────────
# ATR(14) as % of price must exceed this minimum for the MA direction to be
# considered valid. Below it the market is ranging — bias is suppressed.
_ATR_MIN_PCT: dict = {
    "Forex":     0.05,   # 0.05% of price (forex moves in tight bands)
    "Index":     0.30,   # 0.30%
    "Commodity": 0.30,
    "Crypto":    0.50,   # crypto needs more ATR to confirm a genuine trend
}


def _fetch_ma_atr_vol_bias(symbol: str, cls: str) -> tuple[str, str, str]:
    """
    Enhanced MA fallback with three independent filters:

    1. MA(20) — directional bias candidate (Bullish / Bearish / Neutral).
    2. ATR(14) volatility gate — if ATR(14) as % of price is below the
       asset-class minimum, the market is ranging and the MA signal is
       unreliable → override bias to Neutral.
    3. Volume(20) confirmation — if the current bar's volume exceeds
       1.1× the 20-bar volume MA *in the direction of the MA bias*,
       upgrade confidence from Weak → Moderate.

    Returns: (bias, confidence, detail_note)
    """
    min_atr_pct = _ATR_MIN_PCT.get(cls, 0.30)

    ticker = yf.Ticker(symbol)
    hist   = ticker.history(period="10d", interval="1h")
    hist   = hist.dropna(subset=["Close", "High", "Low", "Volume"])

    if len(hist) < 20:
        return "Neutral", "Weak", "Insufficient history for MA/ATR/Vol fallback"

    closes  = hist["Close"]
    highs   = hist["High"]
    lows    = hist["Low"]
    volumes = hist["Volume"]
    price   = closes.iloc[-1]

    # ── 1. MA(20) bias ───────────────────────────────────────────────────────
    ma20 = closes.rolling(20).mean().iloc[-1]
    if price > ma20 * 1.001:
        ma_bias = "Bullish"
    elif price < ma20 * 0.999:
        ma_bias = "Bearish"
    else:
        ma_bias = "Neutral"

    # ── 2. ATR(14) volatility gate ───────────────────────────────────────────
    prev_close = closes.shift(1)
    true_range = pd.concat(
        [highs - lows,
         (highs - prev_close).abs(),
         (lows  - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr14   = true_range.rolling(14).mean().iloc[-1]
    atr_pct = (atr14 / price) * 100.0

    if atr_pct < min_atr_pct:
        return (
            "Neutral",
            "Weak",
            (
                f"ATR {atr_pct:.3f}% < min {min_atr_pct}% ({cls}) "
                f"→ market ranging — MA bias suppressed | MA20={ma20:.5g}"
            ),
        )

    # ── 3. Volume(20) confirmation ────────────────────────────────────────────
    vol_ma20 = volumes.rolling(20).mean().iloc[-1]
    curr_vol = volumes.iloc[-1]
    vol_ratio = curr_vol / vol_ma20 if vol_ma20 > 0 else 0.0
    vol_confirms = vol_ratio > 1.10 and ma_bias != "Neutral"

    conf = "Moderate" if vol_confirms else "Weak"
    vol_note = (
        f"Vol {vol_ratio:.1f}× avg ({'✓ confirms' if vol_confirms else '✗ weak'})"
    )
    detail = (
        f"MA20={ma20:.5g} | Price={price:.5g} | "
        f"ATR%={atr_pct:.3f}% (min {min_atr_pct}%) | {vol_note}"
    )
    return ma_bias, conf, detail


@st.cache_data(ttl=TTL_TECH, show_spinner=False)
def _cached_technical(symbol: str, tv_sym: str, tv_exch: str, tv_scr: str, cls: str):
    """
    Cached technical data fetch. Returns:
        (bias, confidence, source_label, detail_note)

    PRIMARY: TradingView Multi-Timeframe (1H + 4H + Daily).
      • Consensus: ≥2/3 timeframes must agree on direction.
      • 3/3 → Strong; 2/3 → Moderate; <2/3 → Neutral/Weak.
      • 0.8–1.5 s random jitter between each individual TF call.

    FALLBACK (TV unavailable or 429): MA(20) + ATR(14) + Volume(20).
      • ATR gate: if ATR% < class minimum → bias suppressed (ranging market).
      • Volume confirm: current vol >1.1× avg → Weak upgrades to Moderate.

    CIRCUIT BREAKER: first 429 sets _tv_429_tripped in session_state;
    all remaining assets skip TV entirely for that session.
    """
    tv_tripped = st.session_state.get("_tv_429_tripped", False)

    # ── Attempt TradingView MTF ──────────────────────────────────────────────
    if _TV_TA and not tv_tripped:
        try:
            recs               = retry(_fetch_tv_mtf, tv_sym, tv_exch, tv_scr)
            bias, conf, detail = _mtf_consensus(recs)
            return bias, conf, "TradingView MTF (1H/4H/D)", detail
        except RateLimitError:
            logging.warning(
                "TradingView 429 for %s — circuit breaker tripped; "
                "all remaining assets will use MA+ATR+Vol fallback this session.",
                symbol,
            )
            st.session_state["_tv_429_tripped"] = True
        except Exception as exc:
            logging.warning("TradingView MTF failed for %s: %s", symbol, exc)

    # ── MA + ATR + Volume fallback ────────────────────────────────────────────
    try:
        bias, conf, detail = retry(_fetch_ma_atr_vol_bias, symbol, cls)
        rl_tag = " [TV rate-limited]" if tv_tripped else ""
        return bias, conf, f"MA+ATR+Vol Fallback{rl_tag}", detail
    except Exception as exc:
        logging.error("MA+ATR+Vol fallback failed for %s: %s", symbol, exc)

    return "Neutral", "Weak", "Data stale (>2h)", "All sources failed"


def get_technical_bias_with_consensus(
    symbol: str, tv_sym: str, tv_exch: str, tv_scr: str, cls: str
):
    """
    3-step consensus guard: flip bias only if ≥2 of 3 most-recent readings agree.
    History persisted in st.session_state across Streamlit reruns.
    Returns: (bias, confidence, source_label, detail_note)
    """
    bias, conf, source, detail = _cached_technical(symbol, tv_sym, tv_exch, tv_scr, cls)

    hist_key = f"_tech_hist_{symbol}"
    history  = st.session_state.get(hist_key, [])
    history.append(bias)
    history  = history[-3:]   # keep last 3 readings
    st.session_state[hist_key] = history

    # Consensus: dominant bias in last 3 readings must appear ≥2 times
    if len(history) >= 2:
        dominant, dom_count = Counter(history).most_common(1)[0]
        if dom_count >= 2:
            bias = dominant

    return bias, conf, source, detail


# ═════════════════════════════════════════════════════════════════════════════
# PILLAR 2A — GLOBAL MARKET MOOD (Alternative.me Fear & Greed)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=TTL_SENT, show_spinner=False)
def fetch_fear_greed() -> dict:
    """
    Fetch Fear & Greed Index from Alternative.me free API.
    Returns dict with value, label, regime, mood (RISK_ON/RISK_OFF/NEUTRAL).
    Raises on failure — caller handles fallback.
    """
    url  = "https://api.alternative.me/fng/?limit=1"
    resp = retry(requests.get, url, timeout=10)
    resp.raise_for_status()
    entry = resp.json()["data"][0]
    val   = int(entry["value"])
    label = entry["value_classification"]

    if val <= 24:
        regime, mood = "Extreme Fear", "RISK_OFF"
    elif val <= 44:
        regime, mood = "Fear",         "RISK_OFF"
    elif val <= 55:
        regime, mood = "Neutral",      "NEUTRAL"
    elif val <= 75:
        regime, mood = "Greed",        "RISK_ON"
    else:
        regime, mood = "Extreme Greed","RISK_ON"

    return {"value": val, "label": label, "regime": regime, "mood": mood}


def _fng_currency_bias(currency: str, mood: str) -> str:
    """Map a single currency to Bullish/Bearish/Neutral based on risk regime."""
    if mood == "RISK_ON":
        if currency in RISK_ON_CCY:  return "Bullish"
        if currency in RISK_OFF_CCY: return "Bearish"
    elif mood == "RISK_OFF":
        if currency in RISK_OFF_CCY: return "Bullish"
        if currency in RISK_ON_CCY:  return "Bearish"
    return "Neutral"


def _fng_nonforex_bias(asset_cls: str, mood: str) -> str:
    """Risk regime → directional bias for non-forex assets."""
    if mood == "NEUTRAL":
        return "Neutral"
    if asset_cls == "Index":
        return "Bullish" if mood == "RISK_ON" else "Bearish"
    if asset_cls == "Crypto":
        return "Bullish" if mood == "RISK_ON" else "Bearish"
    if asset_cls == "Commodity":
        # Precious metals / Oil are defensive Risk-OFF beneficiaries
        return "Bullish" if mood == "RISK_OFF" else "Bearish"
    return "Neutral"


def fng_bias_for_asset(asset: dict, mood: str) -> str:
    """Derive asset-level directional bias from the F&G risk mood."""
    base  = asset.get("base")
    quote = asset.get("quote")
    if base and quote:
        b_bias = _fng_currency_bias(base,  mood)
        q_bias = _fng_currency_bias(quote, mood)
        if b_bias == "Bullish" and q_bias == "Bearish": return "Bullish"
        if b_bias == "Bearish" and q_bias == "Bullish": return "Bearish"
        return _fng_nonforex_bias(asset["cls"], mood)   # ambiguous pair → use regime
    return _fng_nonforex_bias(asset["cls"], mood)


# ═════════════════════════════════════════════════════════════════════════════
# PILLAR 2B — RETAIL POSITIONING (IG Client Sentiment — Contrarian)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=TTL_SENT, show_spinner=False)
def fetch_ig_sentiment(market_id: str | None) -> dict | None:
    """
    Pull IG Labs public client sentiment.
    Endpoint: GET https://labs.ig.com/rest/client-sentiment/{marketId}
    Returns dict with long_pct, short_pct, contrarian_bias — or None on failure.
    """
    if not market_id:
        return None
    try:
        url  = f"https://labs.ig.com/rest/client-sentiment/{market_id}"
        resp = retry(
            requests.get, url,
            timeout=10,
            headers={"Accept": "application/json"},
        )
        if resp.status_code != 200:
            return None
        data      = resp.json()
        long_pct  = float(data.get("longPositionPercentage",  50.0))
        short_pct = float(data.get("shortPositionPercentage", 50.0))

        # Contrarian signal: majority longs → expect price to fall
        if long_pct > 60:
            contrarian_bias = "Bearish"
        elif short_pct > 60:
            contrarian_bias = "Bullish"
        else:
            contrarian_bias = "Neutral"

        return {
            "long_pct":        long_pct,
            "short_pct":       short_pct,
            "contrarian_bias": contrarian_bias,
        }
    except Exception as exc:
        logging.warning("IG Sentiment failed for %s: %s", market_id, exc)
        return None


def get_sentiment_bias(asset: dict, fng_data: dict):
    """
    Combine global mood (F&G) and IG contrarian signal.
    Returns: (bias, note, weights_adjusted)
        weights_adjusted=True means IG was unavailable → redistribute 25% sent.
    """
    mood         = fng_data["mood"]
    fng_val      = fng_data["value"]
    global_bias  = fng_bias_for_asset(asset, mood)
    ig_result    = fetch_ig_sentiment(asset.get("ig_id"))

    if ig_result is None:
        note = (
            f"F&G {fng_val} ({fng_data['label']}) — "
            "Retail Data Unavailable, Global Mood Only"
        )
        return global_bias, note, True   # weights_adjusted

    # Average global mood + contrarian IG (equal weight within this pillar)
    score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
    avg_score = (
        score_map[global_bias] + score_map[ig_result["contrarian_bias"]]
    ) / 2.0

    if avg_score >  0.25: combined = "Bullish"
    elif avg_score < -0.25: combined = "Bearish"
    else:                 combined = "Neutral"

    note = (
        f"F&G {fng_val} ({fng_data['label']}) | "
        f"IG Retail: {ig_result['long_pct']:.0f}% long / "
        f"{ig_result['short_pct']:.0f}% short (Contrarian Signal)"
    )
    return combined, note, False


# ═════════════════════════════════════════════════════════════════════════════
# PILLAR 3A — ANALYST CONSENSUS (yfinance)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=TTL_FUND, show_spinner=False)
def fetch_analyst_consensus(symbol: str):
    """
    Pull analyst recommendation from yfinance info dict.
    Returns: (bias, n_analysts)  bias ∈ {Bullish, Bearish, Neutral, None}
    """
    try:
        info = yf.Ticker(symbol).info
        rec  = (info.get("recommendationKey") or "").lower()
        n    = info.get("numberOfAnalystOpinions") or 0
        if not rec or n == 0:
            return None, 0
        if rec in ("strong_buy", "buy", "outperform"):
            return "Bullish", n
        if rec in ("sell", "strong_sell", "underperform"):
            return "Bearish", n
        return "Neutral", n
    except Exception as exc:
        logging.warning("yfinance consensus failed for %s: %s", symbol, exc)
        return None, 0


# ═════════════════════════════════════════════════════════════════════════════
# PILLAR 3B — CENTRAL BANK STANCE (FRED + manual fallback)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(ttl=TTL_FUND, show_spinner=False)
def _fred_rate_direction(series_id: str, fred_key: str) -> str | None:
    """
    Fetch 120-day window of a FRED rate series.
    Rising (>+0.10 pp) → Hawkish; Falling (<-0.10 pp) → Dovish; else Neutral.
    Returns None on any failure.
    """
    if not _FREDAPI or not fred_key or fred_key == "YOUR_FREE_FRED_KEY":
        return None
    try:
        from fredapi import Fred as _Fred
        fred  = _Fred(api_key=fred_key)
        end   = datetime.today()
        start = end - timedelta(days=120)
        data  = fred.get_series(series_id, observation_start=start, observation_end=end).dropna()
        if len(data) < 4:
            return None
        diff = data.iloc[-1] - data.iloc[-4]
        if diff >  0.10: return "Hawkish"
        if diff < -0.10: return "Dovish"
        return "Neutral"
    except Exception as exc:
        logging.warning("FRED fetch failed for %s: %s", series_id, exc)
        return None


def get_cb_stance_for_currency(currency: str, fred_key: str):
    """
    Returns: (stance, bias, note)
    Tries FRED live data first, falls back to manual table.
    """
    series = FRED_CB_SERIES.get(currency)
    if series:
        live_stance = _fred_rate_direction(series, fred_key)
        if live_stance:
            return live_stance, CB_STANCE_TO_BIAS[live_stance], f"FRED live ({series})"

    # Manual fallback
    manual_stance, manual_desc = CB_MANUAL_STANCES.get(currency, ("Neutral", "No data"))
    return manual_stance, CB_STANCE_TO_BIAS[manual_stance], f"Manual: {manual_desc}"


def get_fundamental_bias(asset: dict, fng_data: dict, fred_key: str):
    """
    Combine analyst consensus + CB stance differential (for forex)
    or analyst consensus alone (for non-forex).
    Returns: (bias, note)
    """
    symbol = asset["symbol"]
    base   = asset.get("base")
    quote  = asset.get("quote")
    cls    = asset["cls"]

    analyst_bias, n_analysts = fetch_analyst_consensus(symbol)

    # ── Forex: CB differential model ────────────────────────────────────────
    if cls == "Forex" and base and quote:
        _, base_bias,  base_note  = get_cb_stance_for_currency(base,  fred_key)
        _, quote_bias, quote_note = get_cb_stance_for_currency(quote, fred_key)

        score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}
        diff      = score_map[base_bias] - score_map[quote_bias]
        if diff >  0: cb_pair_bias = "Bullish"
        elif diff < 0: cb_pair_bias = "Bearish"
        else:         cb_pair_bias = "Neutral"

        if analyst_bias is None:
            # Fallback: use CB stance only
            return (
                cb_pair_bias,
                f"Proxy: CB Stance — {base} {base_bias} / {quote} {quote_bias}",
            )

        # Average analyst + CB differential
        avg = (score_map[analyst_bias] + score_map[cb_pair_bias]) / 2.0
        if avg >  0.25: final = "Bullish"
        elif avg < -0.25: final = "Bearish"
        else:           final = "Neutral"
        return (
            final,
            f"Analyst({n_analysts}) + CB diff ({base} {base_bias}/{quote} {quote_bias})",
        )

    # ── Non-forex: analyst consensus or risk-regime proxy ───────────────────
    if analyst_bias is not None:
        return analyst_bias, f"Analyst consensus ({n_analysts} opinions)"

    # Proxy: risk regime as fundamental proxy
    proxy = _fng_nonforex_bias(cls, fng_data["mood"])
    return proxy, "Proxy: CB Stance / Risk Regime (no analyst data)"


# ═════════════════════════════════════════════════════════════════════════════
# SCORING ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def calculate_weighted_score(
    tech_bias: str,
    sent_bias: str,
    fund_bias: str,
    sent_weights_adjusted: bool = False,
) -> float:
    """
    Weighted score: Fundamentals 40% / Technicals 35% / Sentiment 25%.
    If sentiment unavailable, redistribute 25% equally to Tech (+12.5%) and Fund (+12.5%).
    Bullish=100, Neutral=50, Bearish=0.
    """
    if sent_weights_adjusted:
        w = {"fundamental": 0.525, "technical": 0.475, "sentiment": 0.0}
    else:
        w = WEIGHTS

    score = (
        BIAS_PCT.get(fund_bias, 50.0) * w["fundamental"]
        + BIAS_PCT.get(tech_bias, 50.0) * w["technical"]
        + BIAS_PCT.get(sent_bias, 50.0) * w["sentiment"]
    )
    return round(score, 1)


def apply_veto_rule(score: float, tech_bias: str, fund_bias: str):
    """
    Veto rule: if Technicals and Fundamentals directly oppose each other,
    cap maximum status at WATCH (74.9). Returns (adjusted_score, veto_applied).
    """
    directly_opposed = (
        {tech_bias, fund_bias} == {"Bullish", "Bearish"}
    )
    if directly_opposed and score >= 75.0:
        return 74.9, True
    return score, False


def get_trade_status(score: float) -> str:
    for threshold, label in STATUS_THRESHOLDS:
        if score >= threshold:
            return label
    return "AVOID"


def detect_conflict(tech_bias: str, sent_bias: str, fund_bias: str):
    """
    Conflict = any two pillars hold directly opposing views.
    Returns (conflict_flag, conflict_note).
    """
    biases   = [tech_bias, sent_bias, fund_bias]
    has_bull = "Bullish" in biases
    has_bear = "Bearish" in biases
    if has_bull and has_bear:
        bull_pillars = [
            n for n, b in zip(["Tech","Sent","Fund"], biases) if b == "Bullish"
        ]
        bear_pillars = [
            n for n, b in zip(["Tech","Sent","Fund"], biases) if b == "Bearish"
        ]
        note = (
            f"Mixed Signals – Low Reliability "
            f"({'/'.join(bull_pillars)} Bullish vs {'/'.join(bear_pillars)} Bearish)"
        )
        return True, note
    return False, ""


# ═════════════════════════════════════════════════════════════════════════════
# CURRENCY STRENGTH MAP
# ═════════════════════════════════════════════════════════════════════════════

def build_currency_strength_map(results: list[dict]) -> dict:
    """
    For each forex pair result, credit base and debit quote for each pillar.
    Strength = aggregate score across all pillars and all pairs involving that currency.
    Normalise to 0–100%.
    """
    raw_scores = {c: 0   for c in CURRENCIES}
    pair_counts = {c: 0  for c in CURRENCIES}

    score_map = {"Bullish": 1, "Neutral": 0, "Bearish": -1}

    for row in results:
        if row["cls"] != "Forex":
            continue
        base  = row.get("base")
        quote = row.get("quote")
        if not base or not quote:
            continue

        for pillar_key in ("tech_bias", "sent_bias", "fund_bias"):
            pillar_score = score_map.get(row[pillar_key], 0)
            raw_scores[base]  += pillar_score
            raw_scores[quote] -= pillar_score   # inverse for quote currency

        pair_counts[base]  += 1
        pair_counts[quote] += 1

    # Normalise: each pair contributes 3 pillars × ±1 per currency
    strength = {}
    for ccy in CURRENCIES:
        n = pair_counts.get(ccy, 0)
        if n == 0:
            strength[ccy] = 50.0
        else:
            max_possible = n * 3          # 3 pillars × number of pairs
            raw          = raw_scores[ccy]
            # Scale from [-max_possible, +max_possible] → [0, 100]
            normalised   = ((raw / max_possible) + 1.0) / 2.0 * 100.0
            strength[ccy] = round(max(0.0, min(100.0, normalised)), 1)

    return dict(sorted(strength.items(), key=lambda x: -x[1]))


# ═════════════════════════════════════════════════════════════════════════════
# PAIRING ENGINE — DIVERGENCE MODEL
# ═════════════════════════════════════════════════════════════════════════════

def build_high_conviction_setups(currency_strength: dict, top_n: int = 5) -> list[dict]:
    """
    Compare strongest vs weakest currencies.
    Divergence Score = (|strength_diff| / 100) × 100
    (Max possible diff = 100 → Score = 100%).
    """
    sorted_ccy = sorted(currency_strength.items(), key=lambda x: -x[1])
    setups     = []

    for i, (strong_ccy, strong_pct) in enumerate(sorted_ccy):
        for j, (weak_ccy, weak_pct) in enumerate(sorted_ccy):
            if i >= j:
                continue
            diff  = strong_pct - weak_pct
            if diff < 10.0:
                continue
            score  = round(diff, 1)
            status = get_trade_status(score)
            setups.append({
                "Pair":                  f"{strong_ccy}/{weak_ccy}",
                "Direction":             f"Long {strong_ccy} / Short {weak_ccy}",
                "Strong Currency (%)":   strong_pct,
                "Weak Currency (%)":     weak_pct,
                "Divergence Score (%)":  score,
                "Status":                status,
            })

    setups.sort(key=lambda x: -x["Divergence Score (%)"])
    return setups[:top_n]


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_full_analysis(fred_key: str) -> tuple[list[dict], dict]:
    """
    Execute all three pillars for every asset.
    Returns (results_list, fng_data).
    """
    # ── Fetch global sentiment ───────────────────────────────────────────────
    try:
        fng_data = fetch_fear_greed()
    except Exception as exc:
        st.warning(f"⚠ Fear & Greed API unavailable — Neutral fallback applied. ({exc})")
        fng_data = {
            "value": 50, "label": "Neutral",
            "regime": "Neutral", "mood": "NEUTRAL",
        }

    results     = []
    progress    = st.progress(0, text="Initialising analysis…")
    total       = len(ASSETS)

    for idx, asset in enumerate(ASSETS):
        sym  = asset["symbol"]
        name = asset["name"]
        progress.progress(idx / total, text=f"Analysing {name} ({idx + 1}/{total})…")

        # ── Pillar 1: Technical ──────────────────────────────────────────────
        tech_bias, tech_conf, tech_src, tech_note = get_technical_bias_with_consensus(
            sym, asset["tv_sym"], asset["tv_exch"], asset["tv_scr"], asset["cls"]
        )

        # ── Pillar 2: Sentiment ──────────────────────────────────────────────
        sent_bias, sent_note, sent_adj = get_sentiment_bias(asset, fng_data)

        # ── Pillar 3: Fundamental ────────────────────────────────────────────
        fund_bias, fund_note = get_fundamental_bias(asset, fng_data, fred_key)

        # ── Scoring ──────────────────────────────────────────────────────────
        score         = calculate_weighted_score(tech_bias, sent_bias, fund_bias, sent_adj)
        score, vetoed = apply_veto_rule(score, tech_bias, fund_bias)
        conflict, conflict_note = detect_conflict(tech_bias, sent_bias, fund_bias)

        # A veto that wasn't otherwise flagged as conflict → add veto note
        if vetoed and not conflict:
            conflict      = True
            conflict_note = "Veto applied: Tech ↔ Fundamental directly oppose (capped at WATCH)"

        status = get_trade_status(score)

        results.append({
            "symbol":        sym,
            "name":          name,
            "cls":           asset["cls"],
            "base":          asset.get("base"),
            "quote":         asset.get("quote"),
            # Pillar outputs
            "tech_bias":     tech_bias,
            "tech_conf":     tech_conf,
            "tech_src":      tech_src,
            "tech_note":     tech_note or "",
            "sent_bias":     sent_bias,
            "sent_note":     sent_note,
            "sent_adj":      sent_adj,
            "fund_bias":     fund_bias,
            "fund_note":     fund_note,
            # Score & status
            "score":         score,
            "status":        status,
            "vetoed":        vetoed,
            "conflict":      conflict,
            "conflict_note": conflict_note,
        })

        time.sleep(0.08)   # polite rate limiting between assets

    progress.progress(1.0, text="Analysis complete ✓")
    time.sleep(0.5)
    progress.empty()
    return results, fng_data


# ═════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL UI — HELPERS & RENDER FUNCTIONS
# ═════════════════════════════════════════════════════════════════════════════

def inject_global_css() -> None:
    """
    Inject the full institutional CSS block.
    Every colour reference uses _C[...] so dark/light toggle affects
    the sidebar, main content, and all custom HTML components uniformly.
    """
    # Sidebar bg is slightly deeper than the main bg in dark mode.
    # We derive it by using surface2 for dark and surface for light.
    sidebar_bg = _C["surface2"] if _C["bg"] != "#f6f8fa" else _C["surface"]

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

    /* ══════════════════════════════════════════════════════
       BASE
    ══════════════════════════════════════════════════════ */
    html, body, .stApp, .stAppViewContainer, [data-testid="stAppViewContainer"] {{
        background-color: {_C['bg']} !important;
        color: {_C['text_pri']} !important;
    }}
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }}

    /* ══════════════════════════════════════════════════════
       SIDEBAR — fully theme-aware (no hardcoded colours)
    ══════════════════════════════════════════════════════ */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    section[data-testid="stSidebar"] > div:first-child {{
        background-color: {sidebar_bg} !important;
        border-right: 1px solid {_C['border']} !important;
    }}
    /* All text inside sidebar */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] small {{
        color: {_C['text_sec']} !important;
    }}
    section[data-testid="stSidebar"] strong {{
        color: {_C['text_pri']} !important;
    }}
    /* Text input */
    section[data-testid="stSidebar"] .stTextInput input,
    section[data-testid="stSidebar"] input {{
        background-color: {_C['surface']} !important;
        border-color: {_C['border']} !important;
        color: {_C['text_pri']} !important;
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px !important;
    }}
    /* Checkbox */
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] [data-testid="stCheckbox"] label {{
        font-size: 12px !important;
        color: {_C['text_sec']} !important;
    }}
    /* Regular sidebar buttons */
    section[data-testid="stSidebar"] .stButton button {{
        background-color: {_C['surface']} !important;
        border: 1px solid {_C['border']} !important;
        color: {_C['text_sec']} !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.04em !important;
        border-radius: 4px !important;
        transition: border-color 0.15s ease, color 0.15s ease !important;
        text-transform: uppercase !important;
        width: 100% !important;
    }}
    section[data-testid="stSidebar"] .stButton button:hover {{
        border-color: {_C['accent']} !important;
        color: {_C['text_pri']} !important;
    }}
    /* Caption text inside sidebar */
    section[data-testid="stSidebar"] .stCaption p,
    section[data-testid="stSidebar"] .stCaption {{
        color: {_C['text_ter']} !important;
        font-size: 10px !important;
    }}

    /* ── Hide "keyboard" InputInstructions rendered by Streamlit ── */
    [data-testid="InputInstructions"],
    [data-testid="stTextInput"] [data-testid="InputInstructions"],
    .stTextInput [data-testid="InputInstructions"] {{
        display: none !important;
    }}

    /* ══════════════════════════════════════════════════════
       TYPOGRAPHY
    ══════════════════════════════════════════════════════ */
    p, span, div, label, td, th {{
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }}
    h1, h2, h3, h4 {{
        font-family: 'Inter', -apple-system, sans-serif !important;
        font-weight: 600 !important;
        color: {_C['text_pri']} !important;
    }}

    /* ── Progress bar ── */
    .stProgress > div > div > div > div {{
        background-color: {_C['accent']} !important;
    }}

    /* ── Streamlit chrome ── */
    #MainMenu, footer {{ visibility: hidden; }}
    header[data-testid="stHeader"] {{
        background-color: {_C['bg']} !important;
        border-bottom: 1px solid {_C['border']} !important;
    }}

    /* ── Divider ── */
    hr {{ border-color: {_C['border']} !important; margin: 0.6rem 0 !important; }}

    /* ── Caption ── */
    .stCaption p {{
        color: {_C['text_ter']} !important;
        font-size: 10px !important;
        letter-spacing: 0.02em !important;
    }}

    /* ── Toast / Alerts ── */
    [data-testid="stToast"] {{
        background-color: {_C['surface2']} !important;
        border: 1px solid {_C['border']} !important;
        color: {_C['text_pri']} !important;
    }}
    [data-testid="stInfo"] {{
        background-color: {_C['surface']} !important;
        border-left-color: {_C['accent']} !important;
        color: {_C['text_sec']} !important;
    }}
    [data-testid="stWarning"] {{
        background-color: {_C['watch_bg']} !important;
        border-left-color: {_C['watch_text']} !important;
    }}
    [data-testid="stSuccess"] {{
        background-color: {_C['ready_bg']} !important;
        border-left-color: {_C['ready_text']} !important;
        color: {_C['ready_text']} !important;
    }}

    /* ══════════════════════════════════════════════════════
       EXPANDER — fix arrow / title overlap for all
       Streamlit versions (details-based AND div-based)
    ══════════════════════════════════════════════════════ */
    [data-testid="stExpander"] {{
        background-color: {_C['surface']} !important;
        border: 1px solid {_C['border']} !important;
        border-radius: 6px !important;
    }}
    /* Streamlit ≥ 1.30 renders expander header as a button, not <summary> */
    [data-testid="stExpander"] > div:first-child,
    [data-testid="stExpander"] [data-testid="stExpanderHeader"],
    [data-testid="stExpander"] [role="button"] {{
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        cursor: pointer !important;
        color: {_C['text_ter']} !important;
        font-size: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.10em !important;
        text-transform: uppercase !important;
        padding: 10px 14px !important;
    }}
    /* Legacy <details><summary> structure */
    [data-testid="stExpander"] details > summary {{
        display: flex !important;
        align-items: center !important;
        gap: 8px !important;
        cursor: pointer !important;
        color: {_C['text_ter']} !important;
        font-size: 10px !important;
        font-weight: 600 !important;
        letter-spacing: 0.10em !important;
        text-transform: uppercase !important;
        list-style: none !important;  /* removes default triangle on Chrome */
    }}
    /* Force the icon to fixed width so text never overlaps it */
    [data-testid="stExpanderToggleIcon"],
    [data-testid="stExpander"] svg:first-child {{
        min-width: 18px !important;
        width: 18px !important;
        height: 18px !important;
        flex-shrink: 0 !important;
    }}
    /* The text label beside the icon */
    [data-testid="stExpander"] [data-testid="stExpanderHeader"] p,
    [data-testid="stExpander"] summary > span,
    [data-testid="stExpander"] summary > p {{
        flex: 1 1 auto !important;
        margin: 0 !important;
        padding-left: 0 !important;
        overflow: hidden !important;
        text-overflow: ellipsis !important;
        white-space: nowrap !important;
    }}

    /* ── Scrollbar ── */
    ::-webkit-scrollbar {{ width: 5px; height: 5px; }}
    ::-webkit-scrollbar-track {{ background: {_C['bg']}; }}
    ::-webkit-scrollbar-thumb {{ background: {_C['border']}; border-radius: 3px; }}

    /* ── Native st.dataframe container (used by COT section) — same card
       frame as the rest of the dashboard for visual consistency. Internal
       header/row colours are rendered by Streamlit's own canvas grid and
       are not stylable via CSS, but the outer border/radius matches. ── */
    [data-testid="stDataFrame"] {{
        border: 1px solid {_C['border']} !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }}

    /* ── Primary button (Run Analysis) ── */
    .stButton button[kind="primary"],
    button[data-testid="baseButton-primary"] {{
        background-color: {_C['accent']} !important;
        border-color: {_C['accent']} !important;
        color: #ffffff !important;
        font-family: 'Inter', sans-serif !important;
        font-size: 11px !important;
        font-weight: 600 !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
        border-radius: 4px !important;
    }}

    /* ══════════════════════════════════════════════════════
       TERMINAL TABLE
    ══════════════════════════════════════════════════════ */
    .t-wrap {{
        overflow-x: auto;
        -webkit-overflow-scrolling: touch;
        border: 1px solid {_C['border']};
        border-radius: 6px;
    }}
    .t-tbl {{
        width: 100%;
        border-collapse: collapse;
        font-size: 11px;
        background: {_C['surface']};
    }}
    .t-tbl thead tr {{
        background: {_C['surface2']};
        position: sticky;
        top: 0;
        z-index: 2;
    }}
    .t-tbl th {{
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {_C['text_ter']};
        padding: 9px 13px;
        text-align: left;
        border-bottom: 1px solid {_C['border']};
        white-space: nowrap;
        user-select: none;
    }}
    .t-tbl th.num {{ text-align: right; }}
    .t-tbl td {{
        padding: 7px 12px;
        border-bottom: 1px solid {_C['border_sub']};
        color: {_C['text_pri']};
        vertical-align: middle;
        line-height: 1.4;
        text-align: left;
    }}
    .t-tbl td.num {{
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        letter-spacing: 0.03em;
    }}
    .t-tbl tbody tr:last-child td {{ border-bottom: none; }}
    .t-tbl tbody tr:nth-child(even) td {{
        background: {_C['surface2']}55;
    }}
    .t-tbl tbody tr:hover td {{
        background: {_C['surface2']};
        transition: background 0.1s ease;
    }}

    /* ── Asset name / class chip ── */
    .asset-name {{
        font-weight: 600;
        font-size: 12px;
        color: {_C['text_pri']};
        letter-spacing: 0.02em;
    }}
    .asset-cls {{
        display: inline-block;
        font-size: 9px;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: {_C['text_ter']};
        background: {_C['surface2']};
        border: 1px solid {_C['border']};
        padding: 1px 5px;
        border-radius: 3px;
        margin-top: 2px;
    }}

    /* ── Status pill ── */
    .pill {{
        display: inline-block;
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        padding: 2px 8px;
        border-radius: 999px;
        white-space: nowrap;
        font-family: 'JetBrains Mono', monospace;
    }}
    .pill-READY {{ color:{_C['ready_text']}; background:{_C['ready_bg']}; border:1px solid {_C['ready_text']}33; }}
    .pill-WATCH {{ color:{_C['watch_text']}; background:{_C['watch_bg']}; border:1px solid {_C['watch_text']}33; }}
    .pill-WAIT  {{ color:{_C['wait_text']};  background:{_C['wait_bg']};  border:1px solid {_C['border']}; }}
    .pill-AVOID {{ color:{_C['avoid_text']}; background:{_C['avoid_bg']}; border:1px solid {_C['avoid_text']}33; }}

    /* ── Bias spans ── */
    .b-bull {{ color:{_C['bullish']}; font-weight:500; }}
    .b-bear {{ color:{_C['bearish']}; font-weight:500; }}
    .b-neut {{ color:{_C['neutral']}; font-weight:400; }}
    .b-conf {{ font-size:9px; color:{_C['text_ter']}; margin-left:3px; font-family:'JetBrains Mono',monospace; }}
    .b-src  {{ font-size:9px; color:{_C['accent']};   margin-left:3px; font-family:'JetBrains Mono',monospace; }}

    /* ── Score colouring ── */
    .sc-rdy {{ color:{_C['ready_text']}; font-weight:700; font-family:'JetBrains Mono',monospace; }}
    .sc-wch {{ color:{_C['watch_text']}; font-weight:600; font-family:'JetBrains Mono',monospace; }}
    .sc-wt  {{ color:{_C['neutral']};    font-weight:500; font-family:'JetBrains Mono',monospace; }}
    .sc-av  {{ color:{_C['bearish']};    font-weight:500; font-family:'JetBrains Mono',monospace; }}

    /* ── Signal / conflict ── */
    .sig-conflict {{ color:{_C['watch_text']}; font-size:10px; line-height:1.35; }}
    .sig-none     {{ color:{_C['text_ter']};   font-size:10px; }}

    /* ══════════════════════════════════════════════════════
       KPI CARDS
    ══════════════════════════════════════════════════════ */
    .kpi-row {{
        display: flex;
        gap: 10px;
        margin-bottom: 14px;
        flex-wrap: wrap;
    }}
    .kpi-card {{
        flex: 1;
        min-width: 100px;
        min-height: 92px;
        background: {_C['surface']};
        border: 1px solid {_C['border']};
        border-radius: 8px;
        padding: 16px;
        transition: border-color 0.15s ease;
    }}
    .kpi-card:hover {{
        border-color: {_C['accent']}66;
    }}
    .kpi-lbl {{
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {_C['text_ter']};
        margin-bottom: 6px;
    }}
    .kpi-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 24px;
        font-weight: 700;
        color: {_C['text_pri']};
        line-height: 1;
        margin-bottom: 4px;
    }}
    .kpi-delta {{
        font-size: 10px;
        color: {_C['text_sec']};
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }}

    /* ══════════════════════════════════════════════════════
       SECTION HEADER
    ══════════════════════════════════════════════════════ */
    .sec-hdr {{
        font-size: 9px;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: {_C['text_ter']};
        padding-bottom: 8px;
        border-bottom: 1px solid {_C['border']};
        margin-bottom: 10px;
    }}

    /* ══════════════════════════════════════════════════════
       CURRENCY STRENGTH BARS
    ══════════════════════════════════════════════════════ */
    .sbar-outer {{
        background: {_C['surface']};
        border: 1px solid {_C['border']};
        border-radius: 8px;
        padding: 16px;
        transition: border-color 0.15s ease;
    }}
    .sbar-outer:hover {{
        border-color: {_C['accent']}66;
    }}
    .sbar-row {{
        display: flex;
        align-items: center;
        gap: 8px;
        padding: 5px 0;
    }}
    .sbar-rank {{
        font-size: 9px;
        color: {_C['text_ter']};
        width: 14px;
        text-align: center;
        flex-shrink: 0;
        font-family: 'JetBrains Mono', monospace;
    }}
    .sbar-ccy {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 11px;
        font-weight: 600;
        color: {_C['text_pri']};
        width: 30px;
        flex-shrink: 0;
    }}
    .sbar-bg {{
        flex: 1;
        height: 4px;
        background: {_C['border_sub']};
        border-radius: 2px;
        overflow: hidden;
    }}
    .sbar-fill {{ height: 100%; border-radius: 2px; }}
    .sbar-pct {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 10px;
        color: {_C['text_sec']};
        width: 38px;
        text-align: right;
        flex-shrink: 0;
    }}
    .sbar-bias {{
        font-size: 9px;
        width: 50px;
        text-align: right;
        flex-shrink: 0;
    }}

    /* ══════════════════════════════════════════════════════
       HIGH-CONVICTION SETUPS
    ══════════════════════════════════════════════════════ */
    .setups-outer {{
        background: {_C['surface']};
        border: 1px solid {_C['border']};
        border-radius: 8px;
        padding: 16px;
        transition: border-color 0.15s ease;
    }}
    .setups-outer:hover {{
        border-color: {_C['accent']}66;
    }}
    .setup-row {{
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 9px 0;
        border-bottom: 1px solid {_C['border_sub']};
    }}
    .setup-row:last-child {{ border-bottom: none; }}
    .setup-pair {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        font-weight: 600;
        color: {_C['text_pri']};
        letter-spacing: 0.04em;
    }}
    .setup-dir {{
        font-size: 10px;
        color: {_C['text_sec']};
        margin-top: 2px;
    }}
    .setup-right {{
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .setup-score {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 13px;
        font-weight: 600;
        text-align: right;
        min-width: 42px;
    }}

    /* ── Freshness dot ── */
    .dot-live  {{ display:inline-block;width:6px;height:6px;border-radius:50%;background:{_C['bullish']};
                  margin-right:5px;vertical-align:middle;animation:pulse 2s infinite; }}
    .dot-stale {{ display:inline-block;width:6px;height:6px;border-radius:50%;background:{_C['neutral']};
                  margin-right:5px;vertical-align:middle; }}
    @keyframes pulse {{ 0%,100%{{opacity:1}} 50%{{opacity:0.3}} }}

    /* ── Notes table in expander ── */
    .notes-tbl {{ width:100%; border-collapse:collapse; font-size:10px; }}
    .notes-tbl th {{
        font-size:9px; letter-spacing:0.09em; text-transform:uppercase;
        color:{_C['text_ter']}; padding:5px 8px;
        border-bottom:1px solid {_C['border']}; text-align:left;
    }}
    .notes-tbl td {{
        padding:5px 8px; color:{_C['text_sec']};
        border-bottom:1px solid {_C['border_sub']}; vertical-align:top;
        line-height:1.4;
    }}
    .notes-tbl tr:last-child td {{ border-bottom:none; }}

    /* ── Landing panel ── */
    .landing-panel {{
        background: {_C['surface']};
        border: 1px solid {_C['border']};
        border-radius: 8px;
        padding: 36px 40px;
        text-align: center;
        margin: 2rem auto;
        max-width: 600px;
    }}
    .landing-title {{
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: {_C['text_sec']};
        margin-bottom: 8px;
    }}
    .landing-body {{
        font-size: 12px;
        color: {_C['text_ter']};
        line-height: 1.6;
        margin: 0 0 20px;
    }}

    /* ══════════════════════════════════════════════════════
       MOBILE RESPONSIVENESS
    ══════════════════════════════════════════════════════ */
    @media (max-width: 768px) {{
        /* Stack KPI cards 2-per-row on mobile */
        .kpi-row {{
            gap: 8px;
        }}
        .kpi-card {{
            min-width: calc(50% - 4px);
            flex: 1 1 calc(50% - 4px);
            padding: 10px 12px;
        }}
        .kpi-val {{
            font-size: 20px;
        }}
        /* Make table horizontally scrollable */
        .t-wrap {{
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
            border-radius: 4px;
        }}
        .t-tbl {{
            font-size: 11px;
            min-width: 620px;
        }}
        .t-tbl th {{ padding: 7px 8px; font-size: 8px; }}
        .t-tbl td {{ padding: 7px 8px; font-size: 11px; }}
        /* Currency strength — shrink gap */
        .sbar-row {{ gap: 5px; }}
        .sbar-pct {{ width: 32px; font-size: 9px; }}
        .sbar-bias {{ width: 40px; font-size: 9px; }}
        /* High-conviction setups */
        .setup-pair {{ font-size: 11px; }}
        .setup-dir  {{ font-size: 9px; }}
        /* Section headers */
        .sec-hdr {{ font-size: 8px; letter-spacing: 0.12em; }}
        /* Ensure block-container doesn't clip */
        .block-container {{
            padding-left: 0.75rem !important;
            padding-right: 0.75rem !important;
        }}
        /* Pill badges */
        .pill {{ font-size: 8px; padding: 2px 6px; }}
        /* Body text minimum 14px on mobile */
        p, span, td, th, label, div.setup-dir {{ font-size: 14px !important; }}
        .t-tbl td, .t-tbl th {{ font-size: 11px !important; }}
        .kpi-lbl {{ font-size: 9px !important; }}
        .sec-hdr, .notes-tbl td, .notes-tbl th {{ font-size: 10px !important; }}
    }}

    /* ══════════════════════════════════════════════════════
       PRINT / SAVE AS PDF
    ══════════════════════════════════════════════════════ */
    @media print {{
        section[data-testid="stSidebar"],
        [data-testid="stToolbar"],
        [data-testid="stHeader"],
        [data-testid="stDecoration"],
        .stButton,
        #MainMenu,
        footer {{ display: none !important; }}
        .stApp, .stAppViewContainer {{
            background-color: #ffffff !important;
        }}
        .block-container {{ padding: 0 !important; max-width: 100% !important; }}
        body {{
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
        }}
        .t-tbl, .sbar-outer, .setups-outer, .kpi-card, [data-testid="stDataFrame"] {{
            border: 1px solid #d0d7de !important;
        }}
    }}
    </style>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT PRIMITIVES
# ─────────────────────────────────────────────────────────────────────────────

def make_badge(status: str) -> str:
    """Pill-shaped status badge — no emoji, monospace, tight."""
    return f'<span class="pill pill-{status}">{status}</span>'


def make_bias_cell(bias: str, conf: str | None = None, src_tag: str | None = None) -> str:
    """Colored bias text with optional confidence and source tag."""
    css = {"Bullish": "b-bull", "Bearish": "b-bear"}.get(bias, "b-neut")
    html = f'<span class="{css}">{bias}</span>'
    if conf:
        html += f'<span class="b-conf">({conf})</span>'
    if src_tag:
        html += f'<span class="b-src">[{src_tag}]</span>'
    return html


def make_score_cell(score: float) -> str:
    """Score colored by threshold only — direction shown in Status column."""
    if score >= 75:   css = "sc-rdy"
    elif score >= 60: css = "sc-wch"
    elif score >= 55: css = "sc-wt"
    else:             css = "sc-av"
    return f'<span class="{css}">{score:.1f}</span>'


def make_card(label: str, value: str, delta: str = "", color: str = "") -> str:
    """KPI card HTML block."""
    val_style = f"color:{color};" if color else ""
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-lbl">{label}</div>'
        f'<div class="kpi-val" style="{val_style}">{value}</div>'
        f'<div class="kpi-delta">{delta}</div>'
        f'</div>'
    )


def _bar_color(pct: float) -> str:
    if pct >= 65: return _C["bullish"]
    if pct <= 35: return _C["bearish"]
    return _C["neutral"]


# ═════════════════════════════════════════════════════════════════════════════
# MACRO CONTEXT — UST 10Y / 2Y, Yield Curve Spread, VIX
# ═════════════════════════════════════════════════════════════════════════════
# Pure addition — no existing logic touched. Uses yfinance only, no API keys.

TTL_MACRO = 3600  # 1 hour


@st.cache_data(ttl=TTL_MACRO, show_spinner=False)
def fetch_macro_context() -> dict:
    """
    Fetch US 10Y yield (^TNX), US 2Y yield (FRED DGS2), and VIX (^VIX).

    ^TNX is a CBOE yield index quoted by Yahoo in *index points* equal to
    yield × 10 (e.g. a 10Y yield of 4.25% shows as 42.5) — divide by 10.

    Yahoo Finance has no reliable 2-Year yield ticker (^UST2Y does not exist).
    We fetch DGS2 directly from FRED's free public CSV endpoint — no API key
    required: https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2

    Returns a dict:
        {
            "tnx": float | None,        # 10Y yield, %
            "tnx_5d_chg": float | None,  # change vs 5 trading days ago, pp
            "us2y": float | None,        # 2Y yield, %
            "spread": float | None,      # 10Y - 2Y, pp
            "vix": float | None,         # VIX level
            "history": pd.DataFrame,     # date, spread — last ~6 months
            "error": str | None,
        }
    """
    out = {
        "tnx": None, "tnx_5d_chg": None, "us2y": None,
        "spread": None, "vix": None,
        "history": pd.DataFrame(), "error": None,
    }

    try:
        # ── 10Y yield (Yahoo quotes this ×10) ─────────────────────────────────
        tnx_hist = yf.Ticker("^TNX").history(period="7mo", interval="1d")

        if not tnx_hist.empty:
            tnx_close = tnx_hist["Close"].dropna() / 10.0
            out["tnx"] = round(float(tnx_close.iloc[-1]), 2)
            if len(tnx_close) > 5:
                out["tnx_5d_chg"] = round(float(tnx_close.iloc[-1] - tnx_close.iloc[-6]), 2)

        # ── 2Y yield — FRED public CSV (no API key required) ──────────────────
        # Yahoo Finance has no reliable 2Y yield ticker. FRED's fredgraph.csv
        # endpoint is publicly accessible, free, and always current.
        # URL: https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2
        two_y_close = pd.Series(dtype=float)
        try:
            from io import StringIO as _StringIO
            _fred_r = requests.get(
                "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DGS2",
                timeout=10,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            if _fred_r.status_code == 200:
                _fred_df = pd.read_csv(_StringIO(_fred_r.text), parse_dates=["DATE"])
                _fred_df = _fred_df[_fred_df["DGS2"] != "."].copy()
                _fred_df["DGS2"] = pd.to_numeric(_fred_df["DGS2"], errors="coerce")
                _fred_df = _fred_df.dropna(subset=["DGS2"])
                # Keep last 7 months to align with TNX history window
                _cutoff = pd.Timestamp.now() - pd.DateOffset(months=7)
                _fred_df = _fred_df[_fred_df["DATE"] >= _cutoff]
                if not _fred_df.empty:
                    two_y_close = _fred_df.set_index("DATE")["DGS2"]
        except Exception as _exc:
            logging.warning("FRED DGS2 fetch failed: %s", _exc)

        # ── Fallback: US Treasury XML API (no key, completely free) ───────────
        # Fetches the last 7 months of daily yield curve data from Treasury.gov
        # and extracts BC_2YEAR. Runs only if FRED returned empty.
        if two_y_close.empty:
            try:
                import xml.etree.ElementTree as _ET
                _ns = {
                    "a": "http://www.w3.org/2005/Atom",
                    "m": "http://schemas.microsoft.com/ado/2007/08/dataservices/metadata",
                    "d": "http://schemas.microsoft.com/ado/2007/08/dataservices",
                }
                _tsy_rows = []
                for _mo in range(7):
                    _dt = pd.Timestamp.now() - pd.DateOffset(months=_mo)
                    _url = (
                        "https://home.treasury.gov/resource-center/data-chart-center/"
                        f"interest-rates/pages/xml?data=daily_treasury_yield_curve"
                        f"&field_tdr_date_value={_dt.strftime('%Y%m')}"
                    )
                    try:
                        _tr = requests.get(_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
                        if _tr.status_code != 200:
                            continue
                        _root = _ET.fromstring(_tr.text)
                        for _entry in _root.findall(".//a:entry/a:content/m:properties", _ns):
                            _d_el  = _entry.find("d:NEW_DATE",  _ns)
                            _y_el  = _entry.find("d:BC_2YEAR",  _ns)
                            if _d_el is not None and _y_el is not None and _y_el.text:
                                try:
                                    _tsy_rows.append({
                                        "date": pd.to_datetime(_d_el.text),
                                        "val":  float(_y_el.text),
                                    })
                                except Exception:
                                    pass
                    except Exception as _exc2:
                        logging.warning("Treasury XML month %s failed: %s", _dt.strftime("%Y-%m"), _exc2)
                if _tsy_rows:
                    _tsy_df = (
                        pd.DataFrame(_tsy_rows)
                        .sort_values("date")
                        .drop_duplicates("date")
                    )
                    two_y_close = _tsy_df.set_index("date")["val"]
                    logging.info("2Y yield loaded from US Treasury XML (%d rows)", len(two_y_close))
            except Exception as _exc3:
                logging.warning("US Treasury XML fallback failed: %s", _exc3)

        if not two_y_close.empty:
            out["us2y"] = round(float(two_y_close.iloc[-1]), 2)

        # ── 10Y-2Y spread + 6-month history ───────────────────────────────────
        if not tnx_hist.empty and not two_y_close.empty:
            # Align daily indexes: both to tz-naive date-only
            tnx_s = (tnx_hist["Close"].dropna() / 10.0)
            tnx_s.index = pd.to_datetime(tnx_s.index).tz_localize(None).normalize()
            _2y_s = two_y_close.copy()
            _2y_s.index = pd.to_datetime(_2y_s.index).normalize()
            df = pd.DataFrame({"tnx": tnx_s, "us2y": _2y_s}).dropna()
            df["spread"] = df["tnx"] - df["us2y"]
            if not df.empty:
                out["spread"] = round(float(df["spread"].iloc[-1]), 2)
                hist = df["spread"].reset_index()
                hist.columns = ["date", "spread"]
                out["history"] = hist.tail(126)

        # ── VIX ────────────────────────────────────────────────────────────────
        vix_hist = yf.Ticker("^VIX").history(period="5d", interval="1d")
        if not vix_hist.empty:
            out["vix"] = round(float(vix_hist["Close"].dropna().iloc[-1]), 2)

    except Exception as exc:
        logging.warning("Macro context fetch failed: %s", exc)
        out["error"] = str(exc)


    return out


def _vix_regime(vix: float | None) -> tuple[str, str]:
    """Return (regime_label, colour) for a VIX level."""
    if vix is None:
        return "—", _C["text_ter"]
    if vix > 30:
        return "Panic", _C["bearish"]
    if vix > 20:
        return "Fear", _C["watch_text"]
    if vix >= 12:
        return "Normal", _C["bullish"]
    return "Complacent", _C["neutral"]


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: MACRO CONTEXT
# ─────────────────────────────────────────────────────────────────────────────

def render_macro_context() -> None:
    """
    4-card macro strip: US 10Y yield, US 2Y yield, 10Y-2Y spread (with
    recession-risk interpretation), and VIX (with regime label).
    Followed by a compact 6-month line chart of the 10Y-2Y spread.
    Purely additive — placed between KPI bar and Verdict Matrix.
    """
    st.markdown('<p class="sec-hdr">Macro Context</p>', unsafe_allow_html=True)

    macro = fetch_macro_context()

    if macro["error"] and macro["tnx"] is None and macro["vix"] is None:
        st.markdown(
            f'<p style="font-size:11px;color:{_C["watch_text"]};">'
            f'Macro data temporarily unavailable — yfinance could not be reached. '
            f'Dashboard continues normally.</p>',
            unsafe_allow_html=True,
        )
        return

    # ── Card 1: US 10Y Yield ───────────────────────────────────────────────────
    tnx = macro["tnx"]
    tnx_chg = macro["tnx_5d_chg"]
    if tnx is not None:
        tnx_val = f"{tnx:.2f}%"
        if tnx_chg is not None:
            chg_col = _C["bullish"] if tnx_chg >= 0 else _C["bearish"]
            chg_sym = "+" if tnx_chg >= 0 else ""
            tnx_delta = f'<span style="color:{chg_col};">{chg_sym}{tnx_chg:.2f}pp (5d)</span>'
        else:
            tnx_delta = "5d change unavailable"
    else:
        tnx_val, tnx_delta = "—", "Data unavailable"

    # ── Card 2: US 2Y Yield ────────────────────────────────────────────────────
    us2y = macro["us2y"]
    us2y_val   = f"{us2y:.2f}%" if us2y is not None else "—"
    us2y_delta = "Short-end policy rate proxy" if us2y is not None else "Data unavailable"

    # ── Card 3: 10Y-2Y Spread ──────────────────────────────────────────────────
    spread = macro["spread"]
    if spread is not None:
        spread_val = f"{spread:+.2f}pp"
        if spread < 0:
            spread_col   = _C["bearish"]
            spread_delta = "Inverted curve → recession risk within 12-18 months"
        else:
            spread_col   = _C["bullish"]
            spread_delta = "Normal curve — no inversion signal"
    else:
        spread_val, spread_col, spread_delta = "—", _C["text_ter"], "Data unavailable"

    # ── Card 4: VIX ────────────────────────────────────────────────────────────
    vix = macro["vix"]
    regime_lbl, regime_col = _vix_regime(vix)
    vix_val   = f"{vix:.2f}" if vix is not None else "—"
    vix_delta = f"Regime: {regime_lbl}" if vix is not None else "Data unavailable"

    st.markdown(
        '<div class="kpi-row">'
        + make_card("US 10Y Yield",    tnx_val,    tnx_delta,    _C["text_pri"])
        + make_card("US 2Y Yield",     us2y_val,   us2y_delta,   _C["text_pri"])
        + make_card("10Y-2Y Spread",   spread_val, spread_delta, spread_col)
        + make_card("VIX",             vix_val,    vix_delta,    regime_col)
        + '</div>',
        unsafe_allow_html=True,
    )

    # ── 6-month yield curve spread history ────────────────────────────────────
    hist = macro["history"]
    if not hist.empty:
        chart_df = hist.set_index("date")[["spread"]].rename(columns={"spread": "10Y-2Y Spread (pp)"})
        st.markdown(
            f'<p style="font-size:9px;font-weight:700;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:{_C["text_ter"]};margin:10px 0 4px;">'
            f'10Y-2Y Spread &middot; 6-Month History</p>',
            unsafe_allow_html=True,
        )
        st.line_chart(
            chart_df,
            height=140,
            color=_C["accent"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: TERMINAL HEADER
# ─────────────────────────────────────────────────────────────────────────────

def render_terminal_header(fng_data: dict) -> None:
    """Title + subtitle left — live UTC clock + data freshness right."""
    col_l, col_r = st.columns([3, 1])

    with col_l:
        st.markdown(
            f'<p style="font-size:13px;font-weight:700;letter-spacing:0.07em;'
            f'text-transform:uppercase;color:{_C["text_pri"]};margin:0;">'
            f'Institutional Trading Terminal</p>'
            f'<p style="font-size:11px;color:{_C["text_ter"]};margin:3px 0 0;'
            f'letter-spacing:0.02em;">'
            f'3-Pillar Framework &nbsp;&middot;&nbsp; 1&ndash;3 Day Outlook '
            f'&nbsp;&middot;&nbsp; Fund 40% / Tech 35% / Sent 25%</p>',
            unsafe_allow_html=True,
        )

    with col_r:
        last_run = st.session_state.get("_last_analysis_time")
        if last_run:
            age_s = int((datetime.now() - last_run).total_seconds())
            if age_s < 3600:
                dot = '<span class="dot-live"></span>'
                freshness = f"Live &middot; {age_s // 60}m ago"
            else:
                dot = '<span class="dot-stale"></span>'
                freshness = f"Cached &middot; {age_s // 3600}h ago"
        else:
            dot, freshness = "", "No data"

        st.markdown(
            f'<div style="text-align:right;padding-top:3px;">'
            f'{dot}<span style="font-size:10px;color:{_C["text_ter"]};">{freshness}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
        # Live UTC clock — st.iframe replaces deprecated st.components.v1.html
        st.iframe(
            f"""
            <div style="text-align:right;font-family:'JetBrains Mono',Menlo,monospace;
                        font-size:13px;font-weight:500;color:{_C['text_sec']};
                        padding-top:1px;letter-spacing:0.05em;">
              <span id="clk">--:--:--</span>
              <span style="font-size:10px;color:{_C['text_ter']};margin-left:3px;">UTC</span>
            </div>
            <script>
            (function(){{
              function tick(){{
                var d=new Date();
                var p=function(n){{return String(n).padStart(2,'0');}};
                document.getElementById('clk').textContent=
                  p(d.getUTCHours())+':'+p(d.getUTCMinutes())+':'+p(d.getUTCSeconds());
              }}
              tick(); setInterval(tick,1000);
            }})();
            </script>
            """,
            height=28,
        )

    st.markdown(
        f'<hr style="border:none;border-top:1px solid {_C["border"]};margin:10px 0 14px;">',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: KPI BAR
# ─────────────────────────────────────────────────────────────────────────────

def render_kpi_bar(results: list[dict], fng_data: dict) -> None:
    """Six custom HTML cards replacing all st.metric() calls."""
    ready   = sum(1 for r in results if r["status"] == "READY")
    watch   = sum(1 for r in results if r["status"] == "WATCH")
    wait    = sum(1 for r in results if r["status"] == "WAIT")
    avoid   = sum(1 for r in results if r["status"] == "AVOID")
    conf_ct = sum(1 for r in results if r["conflict"])
    total   = len(results)

    fng_val  = fng_data.get("value", "—")
    fng_lbl  = fng_data.get("label", "—")
    mood     = fng_data.get("mood", "NEUTRAL")
    mood_sym = {"RISK_ON": "▲", "RISK_OFF": "▼", "NEUTRAL": "—"}.get(mood, "—")
    mood_col = {"RISK_ON": _C["bullish"], "RISK_OFF": _C["bearish"], "NEUTRAL": _C["neutral"]}.get(mood, _C["neutral"])

    st.markdown(
        f'<div class="kpi-row">'
        + make_card("Ready",     str(ready),   f"{ready}/{total} assets",   _C["ready_text"])
        + make_card("Watch",     str(watch),   f"{watch}/{total} assets",   _C["watch_text"])
        + make_card("Wait",      str(wait),    f"{wait}/{total} assets",    _C["neutral"])
        + make_card("Avoid",     str(avoid),   f"{avoid}/{total} assets",   _C["avoid_text"])
        + make_card("Conflicts", str(conf_ct), f"{conf_ct} mixed signals",  _C["watch_text"] if conf_ct else _C["text_ter"])
        + make_card("Fear &amp; Greed", str(fng_val), f"{mood_sym} {fng_lbl}", mood_col)
        + '</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: VERDICT MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def render_verdict_matrix(results: list[dict]) -> None:
    st.markdown('<p class="sec-hdr">Verdict Matrix</p>', unsafe_allow_html=True)
    st.caption(
        "15 assets  ·  Fund 40%  ·  Tech 35%  ·  Sent 25%  ·  "
        "[MTF] TradingView 1H/4H/D  ·  [MA] MA+ATR+Vol fallback  ·  "
        "Hover rows for detail"
    )

    rows_html = ""
    for r in results:
        # Technical source tag
        if "MTF" in r["tech_src"]:
            src_tag = "MTF"
        elif "Fallback" in r["tech_src"] or "stale" in r["tech_src"]:
            src_tag = "MA"
        else:
            src_tag = None

        tech_html   = make_bias_cell(r["tech_bias"], r["tech_conf"], src_tag)
        sent_html   = make_bias_cell(r["sent_bias"])
        fund_html   = make_bias_cell(r["fund_bias"])
        score_html  = make_score_cell(r["score"])
        # Display SHORT for bearish setups (score ≤25), otherwise normal status
        if r["score"] <= 25:
            status_html = (
                f'<span class="pill pill-AVOID" style="background:{_C["bearish_bg"]}; '
                f'color:{_C["bearish"]}; border:1px solid {_C["bearish"]}33;">SHORT</span>'
            )
        else:
            status_html = make_badge(r["status"])

        if r["conflict"]:
            raw_note   = r["conflict_note"]
            short_note = (raw_note[:54] + "…") if len(raw_note) > 57 else raw_note
            signal_html = (
                f'<span class="sig-conflict" title="{raw_note}">'
                f'{short_note}</span>'
            )
        else:
            signal_html = '<span class="sig-none">—</span>'

        rows_html += (
            f"<tr>"
            f"<td><div class='asset-name'>{r['name']}</div>"
            f"<div><span class='asset-cls'>{r['cls']}</span></div></td>"
            f"<td>{tech_html}</td>"
            f"<td>{sent_html}</td>"
            f"<td>{fund_html}</td>"
            f"<td class='num'>{score_html}</td>"
            f"<td>{status_html}</td>"
            f"<td>{signal_html}</td>"
            f"</tr>"
        )

    table_html = (
        '<div class="t-wrap">'
        '<table class="t-tbl"><thead><tr>'
        '<th style="min-width:100px;">Asset</th>'
        '<th style="min-width:140px;">Technical</th>'
        '<th style="min-width:90px;">Sentiment</th>'
        '<th style="min-width:100px;">Fundamental</th>'
        '<th class="num" style="min-width:70px;">Score %</th>'
        '<th style="min-width:78px;">Status</th>'
        '<th style="min-width:190px;">Signal</th>'
        f'</tr></thead><tbody>{rows_html}</tbody></table></div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

    # Source notes — clean expander, no emoji header
    with st.expander("Pillar Source Notes"):
        notes_rows = ""
        for r in results:
            notes_rows += (
                f"<tr>"
                f"<td><strong style='color:{_C['text_pri']}'>{r['name']}</strong></td>"
                f"<td>{r['tech_src']}</td>"
                f"<td>{r['tech_note'] or '—'}</td>"
                f"<td>{r['sent_note']}</td>"
                f"<td>{r['fund_note']}</td>"
                f"</tr>"
            )
        notes_html = (
            '<div style="overflow-x:auto;">'
            '<table class="notes-tbl"><thead><tr>'
            '<th>Asset</th><th>Tech Source</th><th>Tech Detail</th>'
            '<th>Sentiment Note</th><th>Fundamental Note</th>'
            f'</tr></thead><tbody>{notes_rows}</tbody></table></div>'
        )
        st.markdown(notes_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: CURRENCY STRENGTH
# ─────────────────────────────────────────────────────────────────────────────

def render_currency_strength(currency_strength: dict) -> None:
    st.markdown('<p class="sec-hdr">Currency Strength</p>', unsafe_allow_html=True)
    st.caption("Strongest → Weakest  ·  Normalised 0–100% from 3-pillar aggregation")

    bars = ""
    for rank, (ccy, pct) in enumerate(currency_strength.items(), 1):
        bar_col  = _bar_color(pct)
        bias     = "Bullish" if pct >= 60 else ("Bearish" if pct <= 40 else "Neutral")
        bias_cls = {"Bullish": "b-bull", "Bearish": "b-bear"}.get(bias, "b-neut")
        bars += (
            f'<div class="sbar-row">'
            f'<span class="sbar-rank">{rank}</span>'
            f'<span class="sbar-ccy">{ccy}</span>'
            f'<div class="sbar-bg">'
            f'<div class="sbar-fill" style="width:{pct}%;background:{bar_col};"></div>'
            f'</div>'
            f'<span class="sbar-pct">{pct:.1f}%</span>'
            f'<span class="sbar-bias {bias_cls}">{bias}</span>'
            f'</div>'
        )

    st.markdown(f'<div class="sbar-outer">{bars}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: HIGH-CONVICTION SETUPS
# ─────────────────────────────────────────────────────────────────────────────

def render_high_conviction_panel(setups: list[dict]) -> None:
    st.markdown('<p class="sec-hdr">High-Conviction Setups</p>', unsafe_allow_html=True)
    st.caption("Top 5 currency divergence trades  ·  Score = |Strength Diff| %")

    if not setups:
        st.markdown(
            f'<p style="color:{_C["text_ter"]};font-size:12px;padding:10px 0;">'
            f'No high-conviction setups — strength readings are converging.</p>',
            unsafe_allow_html=True,
        )
        return

    rows = ""
    for s in setups:
        score  = s["Divergence Score (%)"]
        status = s["Status"]
        dir_str = s["Direction"]

        # Color Long/Short in direction string
        if "/" in dir_str and "Long" in dir_str:
            parts = dir_str.split(" / ")
            long_part  = parts[0].replace("Long ", "")
            short_part = parts[1].replace("Short ", "") if len(parts) > 1 else ""
            dir_html = (
                f'<span class="b-bull">Long</span> {long_part}'
                + (f' <span style="color:{_C["text_ter"]};">/</span>'
                   f' <span class="b-bear">Short</span> {short_part}' if short_part else "")
            )
        else:
            dir_html = dir_str

        sc_cls = "sc-rdy" if score >= 75 else ("sc-wch" if score >= 60 else "sc-wt")
        rows += (
            f'<div class="setup-row">'
            f'<div>'
            f'<div class="setup-pair">{s["Pair"]}</div>'
            f'<div class="setup-dir">{dir_html}</div>'
            f'</div>'
            f'<div class="setup-right">'
            f'<div class="setup-score"><span class="{sc_cls}">{score:.1f}%</span></div>'
            f'{make_badge(status)}'
            f'</div>'
            f'</div>'
        )

    st.markdown(f'<div class="setups-outer">{rows}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER: INITIAL STATE LANDING PANEL
# ─────────────────────────────────────────────────────────────────────────────

def render_initial_state() -> bool:
    """Landing panel shown before first analysis. Returns True if user clicks Run."""
    st.markdown(
        f'<div class="landing-panel">'
        f'<p class="landing-title">Terminal Ready</p>'
        f'<p class="landing-body">'
        f'Initialise the 3-pillar engine to score 15 assets across Fundamentals, '
        f'Technicals, and Sentiment. First run approximately 60 seconds. '
        f'Subsequent runs served from cache.'
        f'</p>'
        f'</div>',
        unsafe_allow_html=True,
    )
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        return st.button("Run Analysis", type="primary", width="stretch")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

def main() -> None:
    st.set_page_config(
        page_title            = "Jesse Trading Terminal",
        page_icon             = "▣",
        layout                = "wide",
        initial_sidebar_state = "expanded",
    )

    # ── Apply active colour palette BEFORE any rendering ─────────────────────
    theme = st.session_state.get("_theme", "dark")
    _C.update(_CL if theme == "light" else _C_DARK)
    STATUS_STYLE.update({
        "READY": (f"color:{_C['ready_text']};background:{_C['ready_bg']};"),
        "WATCH": (f"color:{_C['watch_text']};background:{_C['watch_bg']};"),
        "WAIT":  (f"color:{_C['wait_text']} ;background:{_C['wait_bg']} ;"),
        "AVOID": (f"color:{_C['avoid_text']};background:{_C['avoid_bg']};"),
    })
    BIAS_COLOR.update({
        "Bullish": _C["bullish"],
        "Bearish": _C["bearish"],
        "Neutral": _C["neutral"],
    })

    inject_global_css()

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        _sep  = (
            f'<hr style="border:none;border-top:1px solid {_C["border"]};'
            f'margin:12px 0;">'
        )
        _shdr = lambda t: (
            f'<p style="font-size:9px;font-weight:700;letter-spacing:0.12em;'
            f'text-transform:uppercase;color:{_C["text_ter"]};margin:10px 0 6px;">'
            f'{t}</p>'
        )

        # ── Auto-Refresh ──────────────────────────────────────────────────────
        st.markdown(_sep, unsafe_allow_html=True)
        st.markdown(_shdr("Auto-Refresh"), unsafe_allow_html=True)
        auto_refresh_enabled = st.checkbox(
            "Refresh every 30 minutes",
            value=False,
            key="auto_refresh_enabled",
            help="Silently re-runs the full 3-pillar analysis every 30 minutes.",
        )
        if auto_refresh_enabled:
            if not _AUTOREFRESH:
                st.caption("Install: pip install streamlit-autorefresh")
            elif "_last_analysis_time" in st.session_state:
                elapsed_s   = int(
                    (datetime.now() - st.session_state["_last_analysis_time"]).total_seconds()
                )
                remaining_s = max(0, AUTO_REFRESH_INTERVAL_SEC - elapsed_s)
                mins, secs  = divmod(remaining_s, 60)
                st.caption(f"Next refresh in {mins:02d}:{secs:02d}")
            else:
                st.caption("Run analysis once to start the timer.")

        # ── Display — Dark / Light Mode ───────────────────────────────────────
        st.markdown(_sep, unsafe_allow_html=True)
        st.markdown(_shdr("Display"), unsafe_allow_html=True)
        theme_label = "Light Mode" if theme == "dark" else "Dark Mode"
        if st.button(theme_label, key="theme_toggle_btn", width="stretch"):
            st.session_state["_theme"] = "light" if theme == "dark" else "dark"
            st.rerun()

        # ── Export ────────────────────────────────────────────────────────────
        st.markdown(_sep, unsafe_allow_html=True)
        st.markdown(_shdr("Export"), unsafe_allow_html=True)
        if st.button("Print / Save as PDF", key="print_btn", width="stretch"):
            st.session_state["_trigger_print"] = True
            st.rerun()
        if st.button("Save Screenshot", key="screenshot_btn", width="stretch"):
            st.session_state["_trigger_screenshot"] = True
            st.rerun()

        # ── Session ───────────────────────────────────────────────────────────
        st.markdown(_sep, unsafe_allow_html=True)
        st.markdown(_shdr("Session"), unsafe_allow_html=True)
        if st.button("Clear Caches", key="clear_caches_btn", width="stretch"):
            st.cache_data.clear()
            for k in list(st.session_state.keys()):
                if k.startswith("_tech_hist_") or k in (
                    "_results", "_fng", "_tv_429_tripped"
                ):
                    del st.session_state[k]
            st.success("Caches cleared.")
        if st.button("Force Refresh", type="primary",
                     key="force_refresh_btn", width="stretch"):
            for k in ("_results", "_fng", "_tv_429_tripped"):
                st.session_state.pop(k, None)
            st.rerun()

    # ── Auto-Refresh Engine ───────────────────────────────────────────────────
    if auto_refresh_enabled and _AUTOREFRESH:
        ar_count   = st_autorefresh(
            interval=AUTO_REFRESH_INTERVAL_SEC * 1000,
            key="autorefresh_engine",
        )
        prev_count = st.session_state.get("_ar_count", 0)
        if ar_count > prev_count and "_results" in st.session_state:
            st.session_state["_ar_count"] = ar_count
            st.toast("Auto-refresh triggered — re-running 3-pillar analysis.")
            for k in ("_results", "_fng", "_tv_429_tripped"):
                st.session_state.pop(k, None)
            with st.spinner("Auto-refreshing…"):
                results_new, fng_new = run_full_analysis(FRED_API_KEY)
            st.session_state["_results"]            = results_new
            st.session_state["_fng"]                = fng_new
            st.session_state["_last_analysis_time"] = datetime.now()
            st.rerun()
        elif ar_count > prev_count:
            st.session_state["_ar_count"] = ar_count

    # ── Initial state ─────────────────────────────────────────────────────────
    if "_results" not in st.session_state:
        st.markdown(
            f'<p style="font-size:18px;font-weight:700;letter-spacing:0.05em;'
            f'text-transform:uppercase;color:{_C["text_pri"]};margin:0 0 18px;">'
            f'Jesse Trading Terminal</p>'
            f'<hr style="border:none;border-top:1px solid {_C["border"]};'
            f'margin:0 0 20px;">',
            unsafe_allow_html=True,
        )
        run_clicked = render_initial_state()
        if run_clicked:
            with st.spinner("Running 3-pillar analysis across 15 assets…"):
                results, fng = run_full_analysis(FRED_API_KEY)
            st.session_state["_results"]            = results
            st.session_state["_fng"]                = fng
            st.session_state["_last_analysis_time"] = datetime.now()
            st.rerun()
        return

    # ── Dashboard — data available ────────────────────────────────────────────
    results  = st.session_state["_results"]
    fng_data = st.session_state["_fng"]

    render_terminal_header(fng_data)
    render_kpi_bar(results, fng_data)
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {_C["border"]};margin:4px 0 14px;">',
        unsafe_allow_html=True,
    )

    # ── Macro Context — additive section between KPI bar and Verdict Matrix ──
    render_macro_context()
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {_C["border"]};margin:14px 0;">',
        unsafe_allow_html=True,
    )

    render_verdict_matrix(results)
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {_C["border"]};margin:14px 0;">',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([1, 1.55])
    with col_l:
        ccy_strength = build_currency_strength_map(results)
        render_currency_strength(ccy_strength)
    with col_r:
        setups = build_high_conviction_setups(ccy_strength, top_n=5)
        render_high_conviction_panel(setups)

    # ── COT — Institutional Context Layer (absolute bottom) ───────────────────
    render_cot_section(results)

    # ── Footer — single clean line ────────────────────────────────────────────
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {_C["border"]};margin:16px 0 8px;">',
        unsafe_allow_html=True,
    )
    last_t = st.session_state.get("_last_analysis_time")
    ts_str = last_t.strftime("%H:%M:%S  %d %b %Y") if last_t else "—"
    st.markdown(
        f'<p style="font-size:10px;color:{_C["text_ter"]}; '
        f'font-family:\'JetBrains Mono\',monospace;letter-spacing:0.03em;">'
        f'Updated {ts_str} &nbsp;&nbsp;|&nbsp;&nbsp; '
        f'Built by Tusiime Jesse'
        f'</p>',
        unsafe_allow_html=True,
    )

    # ── Print trigger ──────────────────────────────────────────────────────────
    if st.session_state.pop("_trigger_print", False):
        st.markdown("<script>window.print();</script>", unsafe_allow_html=True)

    # ── Screenshot trigger ────────────────────────────────────────────────────
    if st.session_state.pop("_trigger_screenshot", False):
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.markdown(
            f"""
            <script
              src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"
              onload="(function(){{
                var el = document.querySelector('.block-container') || document.body;
                html2canvas(el, {{
                  backgroundColor: '{_C["bg"]}',
                  scale: 2,
                  useCORS: true,
                  logging: false
                }}).then(function(canvas) {{
                  var a  = document.createElement('a');
                  a.download = 'jesse_terminal_{ts_file}.png';
                  a.href = canvas.toDataURL('image/png');
                  document.body.appendChild(a);
                  a.click();
                  document.body.removeChild(a);
                }});
              }})()">
            </script>
            """,
            unsafe_allow_html=True,
        )


# ═════════════════════════════════════════════════════════════════════════════
# INSTITUTIONAL CONTEXT LAYER — COMMITMENT OF TRADERS (CFTC)
# Uses the `cot_reports` library (pip install cot_reports).
# Falls back gracefully if the library or network is unavailable.
# ═════════════════════════════════════════════════════════════════════════════

# ── Asset → CFTC contract name mapping ───────────────────────────────────────
# Financial futures: leveraged money columns (lev_money_positions_long_all etc.)
# Commodity futures: managed money columns (m_money_positions_long_all etc.)

# ── Individual currency → CFTC contract mapping (leveraged money) ───────────
# FIX: previously COT data was fetched per CURRENCY PAIR (e.g. "EURO FX" used
# for EUR/USD), which only reflects ONE side of the trade. A pair's true COT
# bias requires comparing BOTH legs' speculator positioning. We now fetch
# each of the 8 majors individually — including USD via the ICE USD Index
# futures contract, the CFTC-reported proxy for dollar positioning — then
# combine base vs. quote per pair below.
_COT_CURRENCY_MAP: dict[str, str] = {
    "EUR": "EURO FX - CHICAGO MERCANTILE EXCHANGE",
    "GBP": "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE",
    "JPY": "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE",
    "CHF": "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE",
    "CAD": "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "AUD": "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "NZD": "NEW ZEALAND DOLLAR - CHICAGO MERCANTILE EXCHANGE",
    "USD": "USD INDEX - ICE FUTURES U.S.",
}

# Dashboard forex pairs → (base_currency, quote_currency). The combined COT
# signal for a pair is derived from base_net − quote_net (see
# calculate_cot_pair_signal below).
_COT_FOREX_PAIRS: dict[str, tuple[str, str]] = {
    "EUR/USD": ("EUR", "USD"),
    "GBP/USD": ("GBP", "USD"),
    "AUD/USD": ("AUD", "USD"),
    "NZD/USD": ("NZD", "USD"),
    "USD/CAD": ("USD", "CAD"),
    "USD/JPY": ("USD", "JPY"),
    "USD/CHF": ("USD", "CHF"),
}

# Non-forex financial futures — single-contract handling, unchanged.
_COT_NONFOREX_FINANCIAL_MAP: dict[str, str] = {
    "S&P 500": "E-MINI S&P 500 STOCK INDEX - CHICAGO MERCANTILE EXCHANGE",
    "NAS 100": "NASDAQ-100 STOCK INDEX MINI - CHICAGO MERCANTILE EXCHANGE",
    "US 30":   "DJIA x $5 MINI - CHICAGO BOARD OF TRADE",
    "Bitcoin": "BITCOIN - CHICAGO MERCANTILE EXCHANGE",
}

_COT_COMMODITY_MAP: dict[str, str] = {
    "Gold":    "GOLD - COMMODITY EXCHANGE INC.",
    "Oil WTI": "CRUDE OIL, LIGHT SWEET - NEW YORK MERCANTILE EXCHANGE",
}

# Velocity thresholds — financial contracts trade larger notional than commodities
_VEL_THRESH_FIN  = 2000
_VEL_THRESH_COMM = 1000


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_financial_cot(year: int) -> pd.DataFrame | None:
    """
    Fetch Traders-in-Financial-Futures report for `year`.
    Returns the full DataFrame or None on any failure.
    Falls back to year-1 if the current year is unavailable.
    """
    if not _COT_LIB:
        return None
    for yr in (year, year - 1):
        try:
            df = cot_year(year=yr, cot_report_type="traders_in_financial_futures_fut")
            if df is not None and not df.empty:
                # Normalise column names to lowercase with underscores
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                return df
        except Exception as exc:
            logging.warning("COT financial fetch failed for %d: %s", yr, exc)
    return None


@st.cache_data(ttl=86400, show_spinner=False)
def fetch_commodity_cot(year: int) -> pd.DataFrame | None:
    """
    Fetch Disaggregated Futures report for `year`.
    Returns the full DataFrame or None on any failure.
    Falls back to year-1 if the current year is unavailable.
    """
    if not _COT_LIB:
        return None
    for yr in (year, year - 1):
        try:
            df = cot_year(year=yr, cot_report_type="disaggregated_fut")
            if df is not None and not df.empty:
                df.columns = [c.lower().replace(" ", "_") for c in df.columns]
                return df
        except Exception as exc:
            logging.warning("COT commodity fetch failed for %d: %s", yr, exc)
    return None


def _find_name_col(df: pd.DataFrame) -> str | None:
    """Return the column that holds the contract/market name."""
    candidates = [
        "market_and_exchange_names",
        "contract_market_name",
        "market_name",
        "cftc_contract_market_code",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Fuzzy fallback
    for c in df.columns:
        if "market" in c and "name" in c:
            return c
    return None


def _extract_asset_rows(df: pd.DataFrame, contract_name: str) -> pd.DataFrame:
    """Filter `df` to rows matching `contract_name` (case-insensitive)."""
    name_col = _find_name_col(df)
    if name_col is None:
        return pd.DataFrame()
    mask = df[name_col].str.upper().str.strip() == contract_name.upper().strip()
    return df[mask].copy()


def _get_net_cols(df: pd.DataFrame, report_type: str) -> tuple[str, str, str]:
    """
    Return (long_col, short_col, oi_col) for the appropriate speculator category.
    Financial: leveraged money.  Commodity: managed money.
    """
    if report_type == "financial":
        long_col  = next((c for c in df.columns if "lev_money" in c and "long"  in c and "all" in c), None)
        short_col = next((c for c in df.columns if "lev_money" in c and "short" in c and "all" in c), None)
    else:
        long_col  = next((c for c in df.columns if "m_money" in c and "long"  in c and "all" in c), None)
        short_col = next((c for c in df.columns if "m_money" in c and "short" in c and "all" in c), None)

    oi_col = next((c for c in df.columns if "open_interest" in c and "all" in c), None)
    if oi_col is None:
        oi_col = next((c for c in df.columns if "open_interest" in c), None)

    return long_col or "", short_col or "", oi_col or ""


def _get_cot_currency_series(df: pd.DataFrame, contract_name: str, report_type: str) -> pd.DataFrame:
    """
    Build a clean weekly time series for a single CFTC contract.
    Returns a DataFrame with columns [date, net, oi], sorted chronologically.
    Used both for non-forex single-contract assets and as the building block
    for combining two currencies into a forex-pair signal.
    """
    asset_rows = _extract_asset_rows(df, contract_name)
    if asset_rows.empty:
        return pd.DataFrame()

    long_col, short_col, oi_col = _get_net_cols(asset_rows, report_type)
    if not long_col or not short_col:
        return pd.DataFrame()

    rows = asset_rows.copy()
    rows[long_col]  = pd.to_numeric(rows[long_col],  errors="coerce")
    rows[short_col] = pd.to_numeric(rows[short_col], errors="coerce")
    rows            = rows.dropna(subset=[long_col, short_col])
    if rows.empty:
        return pd.DataFrame()

    rows["net"] = rows[long_col] - rows[short_col]
    rows["oi"]  = pd.to_numeric(rows[oi_col], errors="coerce") if oi_col and oi_col in rows.columns else np.nan

    # Date — avoid numeric "yymmdd" code columns (see calculate_cot_signal note)
    date_col = next((c for c in rows.columns if "date" in c and "yymmdd" not in c), None)
    if date_col is None:
        date_col = next((c for c in rows.columns if "date" in c), None)
    if date_col is None:
        return pd.DataFrame()

    rows["_date"] = pd.to_datetime(rows[date_col], errors="coerce")
    rows = rows.dropna(subset=["_date"]).sort_values("_date").reset_index(drop=True)

    return rows[["_date", "net", "oi"]].rename(columns={"_date": "date"})


def _signal_from_net_series(
    net_series: pd.Series,
    oi_latest: float | None,
    vel_thresh: float,
    lookback_weeks: int = 52,
) -> dict:
    """
    Shared classification engine: given a chronologically-ordered net-position
    series (and optionally the latest open interest), compute velocity,
    52-week crowding percentile, exposure, and the ordered-priority signal.

    This is the single source of truth for the signal rules so that both
    single-contract assets (indices, commodities, BTC) and combined forex
    pairs (base_net − quote_net) are classified identically:
        percentile > 90            → CROWDED LONGS
        percentile < 10            → CROWDED SHORTS
        velocity > +thresh (p<90)  → BUY SETUPS
        velocity < -thresh (p>10)  → SHORT SETUPS
        otherwise                  → NEUTRAL
    """
    if len(net_series) < 2:
        return {
            "velocity": None, "percentile": None, "exposure_pct": None,
            "signal": "NO DATA", "interpretation": "Insufficient history",
        }

    current_net  = net_series.iloc[-1]
    previous_net = net_series.iloc[-2]
    velocity     = int(current_net - previous_net)

    window   = min(lookback_weeks, len(net_series))
    net_win  = net_series.iloc[-window:]
    min_val  = net_win.min()
    max_val  = net_win.max()
    if max_val == min_val:
        percentile = 50.0
    else:
        percentile = float((current_net - min_val) / (max_val - min_val) * 100)
    percentile = round(max(0.0, min(100.0, percentile)), 1)

    exposure_pct: float | None = None
    if oi_latest and oi_latest > 0:
        exposure_pct = round(abs(current_net) / oi_latest * 100, 1)

    if percentile > 90:
        signal, interpretation = "CROWDED LONGS", "Overbought — risk of reversal"
    elif percentile < 10:
        signal, interpretation = "CROWDED SHORTS", "Oversold — risk of short squeeze"
    elif velocity > vel_thresh and percentile < 90:
        signal, interpretation = "BUY SETUPS", "Institutions accumulating, room to run"
    elif velocity < -vel_thresh and percentile > 10:
        signal, interpretation = "SHORT SETUPS", "Institutions distributing, downside ahead"
    else:
        signal, interpretation = "NEUTRAL", "Positioning flat, no edge"

    return {
        "velocity":       velocity,
        "percentile":     percentile,
        "exposure_pct":   exposure_pct,
        "signal":         signal,
        "interpretation": interpretation,
    }


def calculate_cot_signal(
    asset_rows: pd.DataFrame,
    report_type: str,
    lookback_weeks: int = 52,
) -> dict:
    """
    Single-contract COT signal — used for non-forex assets (indices,
    commodities, BTC), each backed by exactly one CFTC contract.
    Forex pairs use calculate_cot_pair_signal() instead (see below),
    since a pair requires combining two currencies' positioning.

    Returns a dict with keys:
        velocity, percentile, exposure_pct, signal, interpretation, data_date

    Behavior/output for non-forex assets is unchanged — this function now
    delegates the velocity/percentile/threshold rules to the shared
    _signal_from_net_series() classifier so forex pairs and single-contract
    assets are scored with identical logic.
    """
    _FAIL = {
        "velocity": None, "percentile": None, "exposure_pct": None,
        "signal": "NO DATA", "interpretation": "Insufficient history",
        "data_date": None,
    }

    if asset_rows.empty:
        return _FAIL

    long_col, short_col, oi_col = _get_net_cols(asset_rows, report_type)
    if not long_col or not short_col:
        return _FAIL

    # Ensure numeric
    try:
        rows = asset_rows.copy()
        rows[long_col]  = pd.to_numeric(rows[long_col],  errors="coerce")
        rows[short_col] = pd.to_numeric(rows[short_col], errors="coerce")
        rows            = rows.dropna(subset=[long_col, short_col])
        if len(rows) < 2:
            return _FAIL

        rows["net"] = rows[long_col] - rows[short_col]
        rows        = rows.reset_index(drop=True)
    except Exception as exc:
        logging.warning("COT signal calc error: %s", exc)
        return _FAIL

    # ── Data date ────────────────────────────────────────────────────────────
    # Prefer the real date-string columns; avoid numeric "yymmdd" code columns
    # (e.g. as_of_date_in_form_yymmdd), which pd.to_datetime mis-parses as
    # nanosecond timestamps near the 1970 epoch.
    date_col = next(
        (c for c in rows.columns
         if "date" in c and "yymmdd" not in c),
        None,
    )
    if date_col is None:
        date_col = next((c for c in rows.columns if "date" in c), None)
    data_date = None
    if date_col:
        try:
            data_date = pd.to_datetime(rows[date_col].iloc[-1]).strftime("%d %b %Y")
        except Exception:
            pass

    # ── Latest open interest (for exposure %) ──────────────────────────────────
    oi_latest = None
    if oi_col and oi_col in rows.columns:
        try:
            oi_latest = pd.to_numeric(rows[oi_col].iloc[-1], errors="coerce")
        except Exception:
            oi_latest = None

    vel_thresh = _VEL_THRESH_COMM if report_type == "commodity" else _VEL_THRESH_FIN
    sig = _signal_from_net_series(rows["net"], oi_latest, vel_thresh, lookback_weeks)
    sig["data_date"] = data_date
    return sig


def calculate_cot_pair_signal(
    fin_df: pd.DataFrame | None,
    base_ccy: str,
    quote_ccy: str,
    lookback_weeks: int = 52,
) -> dict:
    """
    FIX: combine two individual currencies' COT positioning into a single
    forex-pair signal, instead of relying on one currency's contract alone.

        net_diff(week) = base_currency_net(week) − quote_currency_net(week)

    The combined net_diff series is then run through the exact same ordered
    classification rules as every other asset (via _signal_from_net_series):
        net_diff velocity > +threshold → BUY SETUPS   (base strength vs quote)
        net_diff velocity < -threshold → SHORT SETUPS  (quote strength vs base)
        52w percentile of net_diff > 90 / < 10 → CROWDED LONGS / CROWDED SHORTS

    Exposure % is the average of each leg's own |net| / open-interest ratio,
    since "% of OI" has no single well-defined meaning across two contracts.

    Returns the same dict shape as calculate_cot_signal:
        velocity, percentile, exposure_pct, signal, interpretation, data_date
    """
    _FAIL = {
        "velocity": None, "percentile": None, "exposure_pct": None,
        "signal": "NO DATA", "interpretation": "Insufficient history",
        "data_date": None,
    }

    base_contract  = _COT_CURRENCY_MAP.get(base_ccy)
    quote_contract = _COT_CURRENCY_MAP.get(quote_ccy)
    if fin_df is None or not base_contract or not quote_contract:
        return _FAIL

    base_series  = _get_cot_currency_series(fin_df, base_contract,  "financial")
    quote_series = _get_cot_currency_series(fin_df, quote_contract, "financial")
    if base_series.empty or quote_series.empty:
        return _FAIL

    merged = pd.merge(
        base_series, quote_series, on="date", suffixes=("_base", "_quote"), how="inner"
    ).sort_values("date").reset_index(drop=True)
    if len(merged) < 2:
        return _FAIL

    merged["net_diff"] = merged["net_base"] - merged["net_quote"]

    sig = _signal_from_net_series(merged["net_diff"], None, _VEL_THRESH_FIN, lookback_weeks)

    # Exposure: average of each leg's own % of its own open interest
    exp_parts: list[float] = []
    current_base_net  = merged["net_base"].iloc[-1]
    current_quote_net = merged["net_quote"].iloc[-1]
    oi_base_latest  = merged["oi_base"].iloc[-1]  if "oi_base"  in merged.columns else None
    oi_quote_latest = merged["oi_quote"].iloc[-1] if "oi_quote" in merged.columns else None
    if oi_base_latest and oi_base_latest > 0:
        exp_parts.append(abs(current_base_net) / oi_base_latest * 100)
    if oi_quote_latest and oi_quote_latest > 0:
        exp_parts.append(abs(current_quote_net) / oi_quote_latest * 100)
    sig["exposure_pct"] = round(sum(exp_parts) / len(exp_parts), 1) if exp_parts else None

    try:
        sig["data_date"] = pd.to_datetime(merged["date"].iloc[-1]).strftime("%d %b %Y")
    except Exception:
        sig["data_date"] = None

    return sig


def _cot_vs_verdict(cot_signal: str, verdict_status: str) -> str:
    """
    Compare COT signal to the 3-pillar verdict and return alignment label.
    """
    bullish_verdicts = {"READY", "WATCH"}
    if cot_signal == "BUY SETUPS"    and verdict_status in bullish_verdicts: return "ALIGNED"
    if cot_signal == "SHORT SETUPS"  and verdict_status in bullish_verdicts: return "ALIGNED"
    if cot_signal == "BUY SETUPS"    and verdict_status in {"AVOID", "WAIT"}: return "CONFLICT"
    if cot_signal == "SHORT SETUPS"  and verdict_status in {"AVOID", "WAIT"}: return "CONFLICT"
    if cot_signal in ("CROWDED LONGS", "CROWDED SHORTS") and verdict_status == "READY":
        return "WARNING"
    return "—"


def _signal_color(signal: str) -> str:
    """Map signal string to hex colour from the active _C palette."""
    return {
        "BUY SETUPS":    _C["bullish"],
        "SHORT SETUPS":  _C["bearish"],
        "CROWDED LONGS": _C["watch_text"],
        "CROWDED SHORTS":_C["watch_text"],
        "NEUTRAL":       _C["neutral"],
        "NO DATA":       _C["text_ter"],
    }.get(signal, _C["text_ter"])


def _verdict_color(label: str) -> str:
    return {
        "ALIGNED":  _C["bullish"],
        "CONFLICT": _C["watch_text"],
        "WARNING":  _C["bearish"],
        "—":        _C["text_ter"],
    }.get(label, _C["text_ter"])


def render_cot_section(results: list[dict]) -> None:
    """
    Standalone COT section — placed at the absolute bottom of the dashboard.
    Fetches CFTC data, scores each asset, compares to 3-pillar verdict,
    and renders a clean institutional dataframe.

    If `cot_reports` is not installed or data cannot be fetched the section
    shows a single-line warning — the rest of the dashboard is unaffected.
    """
    # ── Section header — same .sec-hdr + caption pattern as every other
    #    section (Verdict Matrix, Currency Strength, etc.) for consistency ──
    st.markdown(
        f'<hr style="border:none;border-top:1px solid {_C["border"]};margin:20px 0 14px;">',
        unsafe_allow_html=True,
    )
    st.markdown('<p class="sec-hdr">Institutional Context (COT Analysis)</p>', unsafe_allow_html=True)
    st.caption(
        "Weekly Commitment of Traders  ·  Large Speculator Positioning  ·  "
        "Contrarian signal: extreme longs = bearish risk, extreme shorts = bullish risk"
    )

    if not _COT_LIB:
        st.markdown(
            f'<p style="font-size:11px;color:{_C["watch_text"]};">'
            f'COT section disabled — install the library: '
            f'<code>pip install cot_reports</code></p>',
            unsafe_allow_html=True,
        )
        return

    # ── Fetch data (cached 24 h) ──────────────────────────────────────────────
    cur_year   = datetime.now().year
    fin_df     = fetch_financial_cot(cur_year)
    comm_df    = fetch_commodity_cot(cur_year)

    if fin_df is None and comm_df is None:
        st.markdown(
            f'<p style="font-size:11px;color:{_C["watch_text"]};">'
            f'COT data temporarily unavailable — CFTC server could not be reached. '
            f'Dashboard will retry in 24 hours.</p>',
            unsafe_allow_html=True,
        )
        return

    # ── Build a verdict lookup: asset name → status ───────────────────────────
    verdict_map: dict[str, str] = {r["name"]: r["status"] for r in results}

    # ── Score every asset ─────────────────────────────────────────────────────
    rows_out  = []
    dates_seen: list[str] = []

    def _process(asset_name: str, contract_name: str,
                 df: pd.DataFrame | None, report_type: str) -> None:
        if df is None:
            rows_out.append({
                "Asset":          asset_name,
                "COT Signal":     "NO DATA",
                "Velocity":       None,
                "Percentile":     None,
                "Exposure % OI":  None,
                "vs Verdict":     "—",
                "Interpretation": "Data unavailable",
                "_sig_col":       _C["text_ter"],
                "_ver_col":       _C["text_ter"],
            })
            return

        asset_rows = _extract_asset_rows(df, contract_name)
        sig        = calculate_cot_signal(asset_rows, report_type)

        if sig["data_date"]:
            dates_seen.append(sig["data_date"])

        verdict  = verdict_map.get(asset_name, "—")
        vs_label = _cot_vs_verdict(sig["signal"], verdict)

        rows_out.append({
            "Asset":          asset_name,
            "COT Signal":     sig["signal"],
            "Velocity":       sig["velocity"],
            "Percentile":     sig["percentile"],
            "Exposure % OI":  sig["exposure_pct"],
            "vs Verdict":     vs_label,
            "Interpretation": sig["interpretation"],
            "_sig_col":       _signal_color(sig["signal"]),
            "_ver_col":       _verdict_color(vs_label),
        })

    def _process_pair(pair_name: str, base_ccy: str, quote_ccy: str,
                       df: pd.DataFrame | None) -> None:
        """
        FIX: forex pairs now combine both currencies' individual COT
        positioning (base_net − quote_net) instead of relying on a single
        currency's contract. Same row shape/columns as _process above.
        """
        if df is None:
            rows_out.append({
                "Asset":          pair_name,
                "COT Signal":     "NO DATA",
                "Velocity":       None,
                "Percentile":     None,
                "Exposure % OI":  None,
                "vs Verdict":     "—",
                "Interpretation": "Data unavailable",
                "_sig_col":       _C["text_ter"],
                "_ver_col":       _C["text_ter"],
            })
            return

        sig = calculate_cot_pair_signal(df, base_ccy, quote_ccy)

        if sig["data_date"]:
            dates_seen.append(sig["data_date"])

        verdict  = verdict_map.get(pair_name, "—")
        vs_label = _cot_vs_verdict(sig["signal"], verdict)

        rows_out.append({
            "Asset":          pair_name,
            "COT Signal":     sig["signal"],
            "Velocity":       sig["velocity"],
            "Percentile":     sig["percentile"],
            "Exposure % OI":  sig["exposure_pct"],
            "vs Verdict":     vs_label,
            "Interpretation": sig["interpretation"],
            "_sig_col":       _signal_color(sig["signal"]),
            "_ver_col":       _verdict_color(vs_label),
        })

    # Forex pairs — combined base/quote currency signal (the fix)
    for pair_name, (base_ccy, quote_ccy) in _COT_FOREX_PAIRS.items():
        _process_pair(pair_name, base_ccy, quote_ccy, fin_df)

    # Non-forex financial assets — unchanged single-contract handling
    for name, contract in _COT_NONFOREX_FINANCIAL_MAP.items():
        _process(name, contract, fin_df, "financial")

    # DAX40 — no COT, fixed row
    rows_out.append({
        "Asset":          "DAX 40",
        "COT Signal":     "NO COT DATA",
        "Velocity":       None,
        "Percentile":     None,
        "Exposure % OI":  None,
        "vs Verdict":     "—",
        "Interpretation": "No direct COT — use S&P 500 as proxy",
        "_sig_col":       _C["text_ter"],
        "_ver_col":       _C["text_ter"],
    })

    # Commodity assets — unchanged
    for name, contract in _COT_COMMODITY_MAP.items():
        _process(name, contract, comm_df, "commodity")

    # ── Build display DataFrame ───────────────────────────────────────────────
    df_display = pd.DataFrame(rows_out)

    # Styled signal column — colour via markdown HTML in a single column_config
    # We render the signal as coloured text by injecting minimal HTML
    def _fmt_signal(row: pd.Series) -> str:
        return (
            f'<span style="color:{row["_sig_col"]};font-weight:600;'
            f'font-family:JetBrains Mono,monospace;font-size:11px;">'
            f'{row["COT Signal"]}</span>'
        )

    def _fmt_verdict(row: pd.Series) -> str:
        return (
            f'<span style="color:{row["_ver_col"]};font-weight:600;'
            f'font-size:11px;">'
            f'{row["vs Verdict"]}</span>'
        )

    df_display["COT Signal HTML"] = df_display.apply(_fmt_signal, axis=1)
    df_display["vs Verdict HTML"] = df_display.apply(_fmt_verdict, axis=1)

    # ── Render via st.dataframe with column_config ────────────────────────────
    render_cols = ["Asset", "COT Signal", "Velocity", "Percentile",
                   "Exposure % OI", "vs Verdict", "Interpretation"]

    styled = df_display[render_cols].copy()

    st.dataframe(
        styled,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Asset": st.column_config.TextColumn(
                "Asset",
                width="small",
            ),
            "COT Signal": st.column_config.TextColumn(
                "COT Signal",
                width="medium",
                help="BUY SETUPS / SHORT SETUPS / CROWDED LONGS / CROWDED SHORTS / NEUTRAL",
            ),
            "Velocity": st.column_config.NumberColumn(
                "Velocity",
                format="%+d",
                width="small",
                help="Week-over-week change in net speculator position (contracts)",
            ),
            "Percentile": st.column_config.ProgressColumn(
                "Percentile (52w)",
                format="%.0f%%",
                min_value=0,
                max_value=100,
                width="medium",
                help=">90% = crowded long, <10% = crowded short",
            ),
            "Exposure % OI": st.column_config.NumberColumn(
                "Exposure % OI",
                format="%.1f%%",
                width="small",
                help="|Net positions| as % of open interest",
            ),
            "vs Verdict": st.column_config.TextColumn(
                "vs Verdict",
                width="small",
                help="ALIGNED = COT confirms 3-pillar verdict · CONFLICT = opposing · WARNING = crowded at READY",
            ),
            "Interpretation": st.column_config.TextColumn(
                "Interpretation",
                width="large",
            ),
        },
    )

    # ── Signal colour legend (no emojis) ──────────────────────────────────────
    legend_items = [
        ("BUY SETUPS",     _C["bullish"]),
        ("SHORT SETUPS",   _C["bearish"]),
        ("CROWDED LONGS",  _C["watch_text"]),
        ("CROWDED SHORTS", _C["watch_text"]),
        ("NEUTRAL",        _C["neutral"]),
        ("ALIGNED",        _C["bullish"]),
        ("CONFLICT",       _C["watch_text"]),
        ("WARNING",        _C["bearish"]),
    ]
    legend_html = "".join(
        f'<span style="display:inline-block;margin-right:14px;font-size:9px;'
        f'font-family:JetBrains Mono,monospace;color:{col};">{lbl}</span>'
        for lbl, col in legend_items
    )
    st.markdown(
        f'<div style="margin-top:8px;">{legend_html}</div>',
        unsafe_allow_html=True,
    )

    # ── Footer — data date ────────────────────────────────────────────────────
    latest_date = dates_seen[-1] if dates_seen else "unknown"
    st.markdown(
        f'<p style="font-size:9px;color:{_C["text_ter"]};margin-top:8px;'
        f'font-family:JetBrains Mono,monospace;letter-spacing:0.03em;">'
        f'COT data updates weekly (Fridays) &nbsp;&middot;&nbsp; '
        f'Data as of: {latest_date} &nbsp;&middot;&nbsp; '
        f'Source: CFTC via cot_reports library'
        f'</p>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()

