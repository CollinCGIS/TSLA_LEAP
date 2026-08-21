"""
TSLA LEAPS Investment System
-----------------------------
Rules-based decision support for buying/managing TSLA LEAPS calls.
Data: Yahoo Finance via yfinance (no API key required).

Run locally:   streamlit run app.py
Deploy free:   push this folder to a GitHub repo, then deploy on
               https://share.streamlit.io (Streamlit Community Cloud) —
               no secrets needed since only yfinance (no key) is used.

This tool is for decision support and education only — it is not
financial advice, and it does not place trades.
"""
import io
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

import leaps_core as lc

TICKER = "TSLA"
BENCHMARK = "SPY"
TICKER_SECTOR = "Consumer Discretionary"  # TSLA's GICS sector, for Alpha Vantage's SECTOR endpoint

st.set_page_config(page_title="TSLA LEAPS Investment System", layout="wide")
st.title("🎯 TSLA LEAPS Investment System")
st.caption(
    "Rules-based decision support — not financial advice. Verify all data before trading."
)

# ---------------------------------------------------------------------------
# Sidebar: portfolio + risk settings + existing position
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Portfolio & Risk Settings")
    portfolio_value = st.number_input("Portfolio value ($)", min_value=1000.0,
                                       value=1_000_000.0, step=1000.0, format="%.2f")
    risk_per_trade = st.slider("Risk per new trade (% of portfolio)", 0.5, 10.0, 2.0, 0.5) / 100
    stop_loss_pct = st.slider("Mental stop-loss (% of option premium)", 10, 90, 35, 5) / 100
    max_alloc_pct = st.slider("Max allocation to this position (% of portfolio)", 1, 50, 10, 1) / 100

    st.divider()
    st.header("Existing Position (optional)")
    have_position = st.checkbox("I already hold a TSLA LEAPS position")
    held_contracts = held_cost_basis = held_strike = held_expiry_days = 0
    if have_position:
        held_contracts = st.number_input("Contracts held", min_value=0, value=5, step=1)
        held_cost_basis = st.number_input("Avg. cost basis per contract ($, i.e. premium paid x100)",
                                           min_value=0.0, value=4000.0, step=50.0)
        held_strike = st.number_input("Strike price of held LEAP ($)", min_value=0.0, value=250.0, step=5.0)
        held_expiry_days = st.number_input("Days remaining to expiry", min_value=0, value=300, step=1)

    st.divider()
    st.caption("Data refreshes automatically every 15 minutes.")


# ---------------------------------------------------------------------------
# Data fetching (cached)
# ---------------------------------------------------------------------------

def _yf_retry(func, retries: int = 2, base_delay: float = 1.5):
    """Small retry-with-backoff wrapper for yfinance calls. Yahoo Finance
    (an unofficial, keyless data source) intermittently rate-limits requests
    — this is *more* likely on shared cloud hosting like Streamlit Community
    Cloud, where many apps share the same outbound IP, than on a home
    connection. Retries a couple of times with a short pause before giving
    up, rather than failing on the first hiccup."""
    last_exc = None
    for attempt in range(retries + 1):
        try:
            return func()
        except Exception as e:
            last_exc = e
            if attempt < retries:
                time.sleep(base_delay * (attempt + 1))
    raise last_exc


@st.cache_data(ttl=1800, show_spinner="Fetching TSLA price history...")
def fetch_price_history(ticker: str, period: str = "2y") -> pd.DataFrame:
    try:
        df = _yf_retry(lambda: yf.Ticker(ticker).history(period=period, auto_adjust=True))
    except Exception:
        return pd.DataFrame()  # caller checks .empty and shows a clear message — never crashes the page
    if df.empty:
        return df
    df["MA_20"] = df["Close"].rolling(20).mean()
    df["MA_50"] = df["Close"].rolling(50).mean()
    df["MA_200"] = df["Close"].rolling(200).mean()

    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI_14"] = 100 - (100 / (1 + rs))
    df["RSI_14"] = df["RSI_14"].fillna(50)

    df["VOL_AVG_20"] = df["Volume"].rolling(20).mean()

    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df["REALIZED_VOL_63D"] = log_ret.rolling(63).std() * np.sqrt(252)
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_option_expirations(ticker: str) -> list[str]:
    try:
        return list(_yf_retry(lambda: yf.Ticker(ticker).options))
    except Exception:
        return []  # caller treats this the same as "no LEAPS found" and explains why


@st.cache_data(ttl=1800, show_spinner="Fetching LEAPS option chain...")
def fetch_leaps_chain(ticker: str, expirations: tuple[str, ...],
                       min_days: int, max_days: int) -> pd.DataFrame:
    today = datetime.now(timezone.utc).date()
    frames = []
    tk = yf.Ticker(ticker)
    for exp in expirations:
        exp_date = datetime.strptime(exp, "%Y-%m-%d").date()
        days = (exp_date - today).days
        if not (min_days <= days <= max_days):
            continue
        try:
            chain = _yf_retry(lambda: tk.option_chain(exp), retries=1)
        except Exception:
            continue
        calls = chain.calls.copy()
        if calls.empty:
            continue
        calls["expiration"] = exp
        calls["days_to_expiry"] = days
        frames.append(calls)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    keep = ["contractSymbol", "strike", "lastPrice", "bid", "ask", "volume",
            "openInterest", "impliedVolatility", "expiration", "days_to_expiry"]
    for c in keep:
        if c not in out.columns:
            out[c] = np.nan
    return out[keep]


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_days_to_next_earnings(ticker: str) -> int | None:
    """Days until the next earnings report, or None if unavailable."""
    try:
        edf = _yf_retry(lambda: yf.Ticker(ticker).get_earnings_dates(limit=8))
        if edf is None or edf.empty:
            return None
        today = pd.Timestamp.now(tz=edf.index.tz) if edf.index.tz else pd.Timestamp.now()
        future = edf.index[edf.index >= today]
        if len(future) == 0:
            return None
        next_date = future.min()
        return int((next_date - today).days)
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_near_term_put_call_ratio(ticker: str) -> float | None:
    """Real put/call *volume* ratio from the nearest available expiration,
    as an actual sentiment read rather than a placeholder."""
    try:
        tk = yf.Ticker(ticker)
        exps = _yf_retry(lambda: tk.options)
        if not exps:
            return None
        chain = _yf_retry(lambda: tk.option_chain(exps[0]), retries=1)
        call_vol = pd.to_numeric(chain.calls["volume"], errors="coerce").fillna(0).sum()
        put_vol = pd.to_numeric(chain.puts["volume"], errors="coerce").fillna(0).sum()
        if call_vol <= 0:
            return None
        return float(put_vol / call_vol)
    except Exception:
        return None



FRED_SERIES_ID = "DGS1"  # 1-year constant-maturity Treasury: closest tenor match to a 12-14mo LEAP
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


@st.cache_data(ttl=43200, show_spinner=False)  # 12h — a daily-frequency series doesn't need refreshing often
def fetch_risk_free_rate(series_id: str = FRED_SERIES_ID) -> tuple[float, str]:
    """Risk-free rate for Black-Scholes, pulled from FRED (free, keyless-to-sign-up).
    Returns (rate_as_decimal, source_label). Falls back to the static default in
    leaps_core if no FRED_API_KEY is configured or the request fails, so the app
    still works with zero setup — this is a quality upgrade, not a hard dependency.
    """
    api_key = os.environ.get("FRED_API_KEY")
    if not api_key:
        return lc.RISK_FREE_RATE, "default (set FRED_API_KEY for a live Treasury yield)"
    try:
        resp = requests.get(FRED_URL, params={
            "series_id": series_id, "api_key": api_key, "file_type": "json",
            "sort_order": "desc", "limit": 5,
        }, timeout=10)
        resp.raise_for_status()
        for obs in resp.json().get("observations", []):
            val = obs.get("value")
            if val not in (None, ".", ""):
                return float(val) / 100.0, f"FRED {series_id} ({obs.get('date')})"
        return lc.RISK_FREE_RATE, "default (FRED returned no recent observations)"
    except Exception:
        return lc.RISK_FREE_RATE, "default (FRED request failed)"


def _av_api_key() -> str | None:
    return os.environ.get("ALPHAVANTAGE_API_KEY") or os.environ.get("AV_API_KEY")


def _av_soft_failed(data: dict) -> bool:
    """Alpha Vantage returns HTTP 200 with a JSON body even when rate-limited
    or erroring — it signals this via 'Note'/'Information'/'Error Message'
    keys instead of a proper HTTP error code."""
    return any(k in data for k in ("Note", "Information", "Error Message"))


AV_URL = "https://www.alphavantage.co/query"


@st.cache_data(ttl=21600, show_spinner=False)  # 6h — sector performance doesn't need to be more frequent
def fetch_av_sector_performance(sector_name: str) -> tuple[float | None, str]:
    api_key = _av_api_key()
    if not api_key:
        return None, "no ALPHAVANTAGE_API_KEY set"
    try:
        resp = requests.get(AV_URL, params={"function": "SECTOR", "apikey": api_key}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if _av_soft_failed(data):
            return None, "Alpha Vantage rate-limited or errored"
        for rank_key in ("Rank A: Real-Time Performance", "Rank B: 1 Day Performance"):
            block = data.get(rank_key)
            if block and sector_name in block:
                pct = float(str(block[sector_name]).strip().rstrip("%"))
                return pct, rank_key
        return None, f"'{sector_name}' not found in response"
    except Exception as e:
        return None, f"request failed: {e}"


@st.cache_data(ttl=7200, show_spinner=False)  # 2h — news flow changes faster than the other AV factors
def fetch_av_news_sentiment(ticker: str, limit: int = 50) -> tuple[float | None, str]:
    api_key = _av_api_key()
    if not api_key:
        return None, "no ALPHAVANTAGE_API_KEY set"
    try:
        resp = requests.get(AV_URL, params={
            "function": "NEWS_SENTIMENT", "tickers": ticker, "limit": limit, "apikey": api_key,
        }, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if _av_soft_failed(data):
            return None, "Alpha Vantage rate-limited or errored"
        feed = data.get("feed", [])
        if not feed:
            return None, "no recent news found"
        num = den = 0.0
        for item in feed:
            for ts in item.get("ticker_sentiment", []):
                if ts.get("ticker") == ticker:
                    score = float(ts.get("ticker_sentiment_score", 0.0))
                    rel = float(ts.get("relevance_score", 0.0))
                    num += score * rel
                    den += rel
        if den <= 0:
            return None, "no ticker-relevant sentiment in feed"
        return num / den, f"{len(feed)} articles"
    except Exception as e:
        return None, f"request failed: {e}"


@st.cache_data(ttl=86400, show_spinner=False)  # 24h — only changes once a quarter
def fetch_av_earnings_surprise(ticker: str) -> tuple[float | None, str]:
    api_key = _av_api_key()
    if not api_key:
        return None, "no ALPHAVANTAGE_API_KEY set"
    try:
        resp = requests.get(AV_URL, params={"function": "EARNINGS", "symbol": ticker, "apikey": api_key}, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        if _av_soft_failed(data):
            return None, "Alpha Vantage rate-limited or errored"
        quarterly = data.get("quarterlyEarnings", [])
        if not quarterly:
            return None, "no quarterly earnings data"
        latest = quarterly[0]
        sp = latest.get("surprisePercentage")
        if sp in (None, "None", ""):
            return None, "surprise percentage unavailable for latest quarter"
        return float(sp), latest.get("reportedDate", "unknown date")
    except Exception as e:
        return None, f"request failed: {e}"


@st.cache_data(ttl=86400, show_spinner=False)  # 24h — next earnings date rarely changes intraday
def fetch_av_next_earnings_date(ticker: str) -> tuple[int | None, str]:
    api_key = _av_api_key()
    if not api_key:
        return None, "no ALPHAVANTAGE_API_KEY set"
    try:
        resp = requests.get(AV_URL, params={
            "function": "EARNINGS_CALENDAR", "symbol": ticker, "horizon": "3month", "apikey": api_key,
        }, timeout=15)
        resp.raise_for_status()
        text = resp.text
        if text.strip().startswith("{"):
            return None, "Alpha Vantage rate-limited or errored"  # errors come back as JSON, not CSV
        df = pd.read_csv(io.StringIO(text))
        if df.empty or "reportDate" not in df.columns:
            return None, "no upcoming earnings found"
        df["reportDate"] = pd.to_datetime(df["reportDate"])
        today = pd.Timestamp.now().normalize()
        future = df[df["reportDate"] >= today]
        if future.empty:
            return None, "no upcoming earnings found"
        next_date = future["reportDate"].min()
        return int((next_date - today).days), next_date.strftime("%Y-%m-%d")
    except Exception as e:
        return None, f"request failed: {e}"


# ---------------------------------------------------------------------------
# Pull data
# ---------------------------------------------------------------------------
tsla = fetch_price_history(TICKER)
spy = fetch_price_history(BENCHMARK, period="6mo")

if tsla.empty or len(tsla) < 200:
    st.error("Could not fetch enough TSLA price history to compute 200-day MA. "
              "Yahoo Finance may be rate-limiting or unreachable right now — try again shortly.")
    st.stop()

price = float(tsla["Close"].iloc[-1])
ma20 = float(tsla["MA_20"].iloc[-1])
ma50 = float(tsla["MA_50"].iloc[-1])
ma200 = float(tsla["MA_200"].iloc[-1])
rsi = float(tsla["RSI_14"].iloc[-1])
vol_today = float(tsla["Volume"].iloc[-1])
vol_avg20 = float(tsla["VOL_AVG_20"].iloc[-1]) if not np.isnan(tsla["VOL_AVG_20"].iloc[-1]) else None
rel_vol = (vol_today / vol_avg20) if vol_avg20 else None
realized_vol = float(tsla["REALIZED_VOL_63D"].iloc[-1]) if not np.isnan(tsla["REALIZED_VOL_63D"].iloc[-1]) else None

if not spy.empty and len(spy) > 63:
    tsla_ret_3m = tsla["Close"].iloc[-1] / tsla["Close"].iloc[-63] - 1
    spy_ret_3m = spy["Close"].iloc[-1] / spy["Close"].iloc[-63] - 1
else:
    tsla_ret_3m, spy_ret_3m = 0.0, 0.0

expirations = tuple(fetch_option_expirations(TICKER))
leaps_raw = fetch_leaps_chain(TICKER, expirations, lc.LEAPS_MIN_DAYS, lc.LEAPS_MAX_DAYS)

# Earnings date: prefer Alpha Vantage's EARNINGS_CALENDAR (more reliable), fall
# back to yfinance if no AV key is set or the AV call fails.
av_days_to_earnings, av_earnings_note = fetch_av_next_earnings_date(TICKER)
if av_days_to_earnings is not None:
    days_to_earnings = av_days_to_earnings
    earnings_source = f"Alpha Vantage (next report {av_earnings_note})"
else:
    days_to_earnings = fetch_days_to_next_earnings(TICKER)
    earnings_source = (f"yfinance (Alpha Vantage unavailable: {av_earnings_note})"
                        if days_to_earnings is not None else f"unknown ({av_earnings_note})")

put_call_ratio = fetch_near_term_put_call_ratio(TICKER)
earnings_blackout = lc.is_within_earnings_blackout(days_to_earnings)
risk_free_rate, risk_free_source = fetch_risk_free_rate()

sector_pct, sector_source = fetch_av_sector_performance(TICKER_SECTOR)
news_sentiment_raw, news_source = fetch_av_news_sentiment(TICKER)
earnings_surprise_pct, earnings_surprise_source = fetch_av_earnings_surprise(TICKER)

current_iv = None
valid_options = pd.DataFrame()
if not leaps_raw.empty:
    leaps_raw = leaps_raw.dropna(subset=["strike", "impliedVolatility"])
    leaps_raw = leaps_raw[leaps_raw["impliedVolatility"] > 0]

    greeks = leaps_raw.apply(
        lambda row: lc.bs_call_greeks(
            S=price, K=row["strike"], T=row["days_to_expiry"] / 365.0,
            r=risk_free_rate, sigma=row["impliedVolatility"],
        ),
        axis=1,
        result_type="expand",
    )
    leaps_raw[["bs_price", "delta", "gamma", "theta"]] = greeks

    # liquidity filter: require at least some open interest so we're not
    # sizing a real trade off a strike nobody trades
    liquid = leaps_raw[leaps_raw["openInterest"].fillna(0) > 0].copy()
    pool = liquid if not liquid.empty else leaps_raw

    valid_options = pool[(pool["delta"] >= lc.DELTA_MIN) & (pool["delta"] <= lc.DELTA_MAX)].copy()
    valid_options["delta_dist"] = (valid_options["delta"] - lc.DELTA_TARGET).abs()
    valid_options = valid_options.sort_values("delta_dist")

    # current IV gauge: average IV of near-the-money LEAPS in our window
    atm = pool.iloc[(pool["strike"] - price).abs().argsort()[:5]]
    if not atm.empty:
        current_iv = float(atm["impliedVolatility"].mean())

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
putcall_score = lc.score_sentiment(put_call_ratio) if put_call_ratio is not None else None
news_score = lc.score_news_sentiment(news_sentiment_raw) if news_sentiment_raw is not None else None

scores = {
    "trend": lc.score_trend(price, ma20),
    "ma_cross": lc.score_ma_cross(ma50, ma200),
    "rsi": lc.score_rsi(rsi),
    "iv": lc.score_iv_relative(current_iv, realized_vol),
    "volume": lc.score_relative_volume(rel_vol),
    "rel_strength": lc.score_relative_strength(tsla_ret_3m, spy_ret_3m),
    "sector": lc.score_sector_performance(sector_pct),
    "sentiment": lc.blended_average(putcall_score, news_score),
    "earnings_quality": lc.score_earnings_surprise(earnings_surprise_pct),
}
total = lc.compute_total_score(scores)
buy = lc.buy_signal(scores, total, earnings_blackout=earnings_blackout)
entry_grade = lc.letter_grade(total)

if not valid_options.empty:
    pool_avg_iv_for_scoring = float(valid_options["impliedVolatility"].mean())
    contract_scores = valid_options.apply(
        lambda row: lc.compute_contract_score(
            delta=row["delta"], open_interest=row["openInterest"],
            bid=row["bid"], ask=row["ask"],
            contract_iv=row["impliedVolatility"], pool_avg_iv=pool_avg_iv_for_scoring,
        ),
        axis=1,
    )
    valid_options["contract_score"] = [c["total"] for c in contract_scores]
    valid_options["contract_grade"] = [c["grade"] for c in contract_scores]
    valid_options["delta_fit_score"] = [c["delta_fit"] for c in contract_scores]
    valid_options["liquidity_score"] = [c["liquidity"] for c in contract_scores]
    valid_options["iv_cost_score"] = [c["iv_cost"] for c in contract_scores]
    valid_options = valid_options.sort_values("contract_score", ascending=False)

best_option = valid_options.iloc[0] if not valid_options.empty else None

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_overview, tab_options, tab_manage, tab_estimator = st.tabs(
    ["📈 Overview & Score", "🎯 LEAP Selection & Sizing", "🔧 Manage Position", "🧮 Value Estimator"]
)

# --- Overview tab -----------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("TSLA Price", f"${price:,.2f}")
    c2.metric("Composite Score", f"{total:.1f} / 100", help=f"Grade: {entry_grade}")
    c3.metric("RSI (14)", f"{rsi:.1f}")
    c4.metric("Realized Vol (3mo)", f"{realized_vol*100:.1f}%" if realized_vol else "n/a")
    c5.metric("Next Earnings", f"{days_to_earnings}d" if days_to_earnings is not None else "unknown")

    if earnings_blackout:
        st.error(
            f"🚫 Inside the {lc.EARNINGS_BLACKOUT_DAYS}-day earnings blackout "
            f"({days_to_earnings} days to next report, source: {earnings_source}) — "
            "no new entries regardless of score."
        )
    elif days_to_earnings is None:
        st.info(f"Couldn't determine the next earnings date ({earnings_source}) — verify manually before entering.")
    else:
        st.caption(f"Earnings date source: {earnings_source}")

    st.caption(f"Risk-free rate used in all pricing: {risk_free_rate*100:.2f}% — source: {risk_free_source}")

    if not _av_api_key():
        st.info(
            "Sector performance, news sentiment, and earnings surprise are showing as neutral (50) — "
            "set `ALPHAVANTAGE_API_KEY` to enable them. See the README for a free key."
        )

    if buy:
        st.success(f"📌 BUY SIGNAL (Grade {entry_grade}) — composite score and trend factors both confirm, "
                   "and no earnings blackout.")
    elif not earnings_blackout:
        st.warning(f"🚫 No buy signal today (Grade {entry_grade}).")
        if scores["trend"] < lc.TREND_GATE or scores["ma_cross"] < lc.TREND_GATE:
            st.caption("Trend/MA-cross gate not met, even if other factors look fine.")

    st.markdown("### Score Breakdown")
    factor_order = ["trend", "ma_cross", "rsi", "iv", "volume", "rel_strength",
                     "sector", "sentiment", "earnings_quality"]
    factor_labels = {
        "trend": "Trend (vs 20MA)", "ma_cross": "MA Cross (50/200)", "rsi": "RSI",
        "iv": "IV vs Realized Vol", "volume": "Relative Volume",
        "rel_strength": "Relative Strength vs SPY", "sector": "Sector Performance",
        "sentiment": "Sentiment (put/call + news)", "earnings_quality": "Earnings Surprise Quality",
    }
    score_df = pd.DataFrame({
        "Factor": [factor_labels[k] for k in factor_order],
        "Weight": [f"{lc.SCORE_WEIGHTS[k]*100:.0f}%" for k in factor_order],
        "Score": [round(scores[k], 1) for k in factor_order],
    })
    st.dataframe(score_df, hide_index=True, use_container_width=True)

    with st.expander("Data sources behind each factor this refresh"):
        st.markdown(
            f"- **Put/call ratio**: {f'{put_call_ratio:.2f}' if put_call_ratio is not None else 'unavailable'} "
            "(nearest options expiration, yfinance)\n"
            f"- **News sentiment**: {f'{news_sentiment_raw:+.2f}' if news_sentiment_raw is not None else 'unavailable'} "
            f"({news_source})\n"
            f"- **Sector performance ({TICKER_SECTOR})**: "
            f"{f'{sector_pct:+.2f}%' if sector_pct is not None else 'unavailable'} ({sector_source})\n"
            f"- **Earnings surprise (latest quarter)**: "
            f"{f'{earnings_surprise_pct:+.2f}%' if earnings_surprise_pct is not None else 'unavailable'} "
            f"({earnings_surprise_source})\n\n"
            "Alpha Vantage's free tier is capped at 25 requests/day — each of these calls is cached "
            "well beyond its natural refresh rate specifically to stay inside that budget."
        )

    fig = px.line(tsla.tail(400), x=tsla.tail(400).index, y="Close", title="TSLA Price with Moving Averages")
    fig.add_scatter(x=tsla.tail(400).index, y=tsla["MA_20"].tail(400), mode="lines", name="20-day MA")
    fig.add_scatter(x=tsla.tail(400).index, y=tsla["MA_50"].tail(400), mode="lines", name="50-day MA")
    fig.add_scatter(x=tsla.tail(400).index, y=tsla["MA_200"].tail(400), mode="lines", name="200-day MA")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("How each score is calculated"):
        st.markdown(
            "- **Trend**: price vs 20-day MA (graded, not just above/below).\n"
            "- **MA Cross**: 50-day MA vs 200-day MA spread (golden/death cross strength).\n"
            "- **RSI**: peaks at RSI=50; penalizes both overbought (>70) and oversold (<30) extremes.\n"
            "- **IV vs Realized Vol**: rewards *cheap* options (IV low relative to what the stock has "
            "actually been doing) — as a LEAPS **buyer** you want volatility to be inexpensive, not rich.\n"
            "- **Relative Volume**: today's volume vs its 20-day average, confirming conviction.\n"
            "- **Relative Strength**: TSLA's 3-month return minus SPY's — broad-*market* context.\n"
            f"- **Sector Performance**: real {TICKER_SECTOR} sector performance (Alpha Vantage's `SECTOR` "
            "endpoint) — distinct from Relative Strength above, which is market-wide, not sector-specific. "
            "Shows as neutral (50) without an Alpha Vantage key.\n"
            "- **Sentiment**: blend of put/call *volume* ratio (nearest options expiration, yfinance) and "
            "relevance-weighted news sentiment (Alpha Vantage `NEWS_SENTIMENT`). Uses whichever source(s) "
            "are actually available rather than penalizing a missing one.\n"
            "- **Earnings Surprise Quality**: the most recent quarter's actual-vs-estimate EPS surprise "
            "(Alpha Vantage `EARNINGS`) — a proxy for earnings quality, since forward guidance itself isn't "
            "available as clean structured data from free sources.\n\n"
            f"Separately, **earnings blackout** is a hard entry gate, not a scored factor: no new LEAP "
            f"is opened within {lc.EARNINGS_BLACKOUT_DAYS} days of a report, no matter how high the "
            "score is — a single earnings print can invalidate the whole technical picture overnight. "
            "The next-earnings date itself prefers Alpha Vantage's `EARNINGS_CALENDAR` and falls back to "
            "yfinance if unavailable."
        )

# --- Options / sizing tab ---------------------------------------------------
with tab_options:
    st.markdown(f"### Which LEAP to buy — {lc.LEAPS_MIN_DAYS}-{lc.LEAPS_MAX_DAYS} days out, "
                f"delta {lc.DELTA_MIN:.2f}-{lc.DELTA_MAX:.2f} (target ~{lc.DELTA_TARGET:.2f})")

    if valid_options.empty:
        if not expirations:
            st.error(
                "Couldn't load TSLA's options data from Yahoo Finance this refresh — likely a temporary "
                "rate limit (more common on shared cloud hosting than a home connection). This isn't a "
                "real 'no LEAPS exist' situation — click the refresh icon in the top right, or just "
                "reload the page in a minute or two."
            )
        else:
            st.error(f"No LEAPS in the {lc.LEAPS_MIN_DAYS}-{lc.LEAPS_MAX_DAYS} day / "
                     f"{lc.DELTA_MIN:.2f}-{lc.DELTA_MAX:.2f} delta window were found in the option chain "
                     "that did load. Try again shortly, or the window may need widening.")
    else:
        best = best_option
        st.markdown("#### Top-Rated Contract")
        st.success(
            f"**Grade {best['contract_grade']} ({best['contract_score']:.0f}/100)** — "
            f"**{TICKER} {best['expiration']} ${best['strike']:.0f} Call** — "
            f"Delta {best['delta']:.3f}, IV {best['impliedVolatility']*100:.1f}%, "
            f"Last ${best['lastPrice']:.2f}, {int(best['days_to_expiry'])} days to expiry, "
            f"Open Interest {int(best['openInterest']) if not np.isnan(best['openInterest']) else 'n/a'}."
        )
        bc1, bc2, bc3 = st.columns(3)
        bc1.metric("Delta Fit", f"{best['delta_fit_score']:.0f}/100")
        bc2.metric("Liquidity", f"{best['liquidity_score']:.0f}/100")
        bc3.metric("IV Cost (vs peers)", f"{best['iv_cost_score']:.0f}/100")
        st.caption(
            "Ranked by a combined contract score — 50% how close delta is to target, 30% liquidity "
            "(open interest + bid-ask spread tightness), 20% how cheap this contract's IV is relative "
            "to the other candidates in this window. This is a *different* rating from the composite "
            "entry-timing score above: that one asks 'should I buy a LEAP at all right now', this one "
            "asks 'given that I'm buying, which specific contract is the best pick'."
        )

        st.markdown("#### All Candidates, Ranked")
        display_cols = ["expiration", "days_to_expiry", "strike", "delta", "contract_score", "contract_grade",
                         "impliedVolatility", "lastPrice", "bid", "ask", "openInterest", "volume"]
        ranked_display = valid_options[display_cols].rename(columns={
            "impliedVolatility": "IV", "lastPrice": "Last", "openInterest": "OpenInt",
            "contract_score": "Score", "contract_grade": "Grade",
        }).round({"delta": 3, "IV": 3, "Score": 1})
        st.dataframe(
            ranked_display.style.background_gradient(subset=["Score"], cmap="RdYlGn", vmin=0, vmax=100),
            hide_index=True, use_container_width=True,
        )

        fig2 = px.scatter(valid_options, x="strike", y="delta", color="contract_score",
                           color_continuous_scale="RdYlGn", range_color=[0, 100],
                           size="openInterest", hover_data=["expiration", "lastPrice", "days_to_expiry"],
                           title="Candidate LEAPS — Delta vs Strike, colored by contract score")
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("### How much to buy")
        premium = float(best["lastPrice"]) if best["lastPrice"] > 0 else float(best["ask"])
        sizing = lc.position_size(portfolio_value, premium, risk_per_trade, stop_loss_pct, max_alloc_pct)

        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Contracts (risk-based)", sizing.contracts_by_risk)
        sc2.metric("Contracts (allocation cap)", sizing.contracts_by_allocation)
        sc3.metric("Recommended", sizing.contracts_recommended)
        st.caption(
            f"Risk-based: caps the position so that a {stop_loss_pct*100:.0f}% drop in the option's "
            f"value only costs {risk_per_trade*100:.1f}% of your ${portfolio_value:,.0f} portfolio "
            f"(${sizing.risk_dollars_budgeted:,.0f}). Allocation cap: never more than "
            f"{max_alloc_pct*100:.0f}% of the portfolio in this trade regardless of stop distance. "
            f"The recommendation is the smaller of the two."
        )
        if sizing.notes:
            for n in sizing.notes:
                st.warning(n)
        else:
            st.write(f"Estimated capital deployed: **${sizing.dollars_deployed:,.0f}**")

# --- Manage position tab -----------------------------------------------------
with tab_manage:
    if not have_position or held_contracts == 0:
        st.info("Check **'I already hold a TSLA LEAPS position'** in the sidebar to get "
                 "add / trim / sell / convert guidance for your specific position.")
    else:
        held_T = held_expiry_days / 365.0
        held_g = lc.bs_call_greeks(S=price, K=held_strike, T=held_T, r=risk_free_rate,
                                    sigma=(current_iv or realized_vol or 0.5))
        held_value_per_contract = held_g["price"] * 100
        held_delta = held_g["delta"]
        unrealized_gain_pct = (
            (held_value_per_contract - held_cost_basis) / held_cost_basis * 100
            if held_cost_basis > 0 else None
        )
        intrinsic = max(price - held_strike, 0.0)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Est. Value / Contract", f"${held_value_per_contract:,.0f}")
        m2.metric("Unrealized P&L", f"{unrealized_gain_pct:+.1f}%" if unrealized_gain_pct is not None else "n/a")
        m3.metric("Current Delta", f"{held_delta:.3f}")
        m4.metric("Days to Expiry", f"{held_expiry_days}")

        st.markdown("### Add to position?")
        st.caption(
            "Two distinct strategies, shown separately since they trade off differently — "
            "pick one deliberately rather than following both blindly."
        )
        pb_ok, pb_reason = lc.should_add_pullback(scores, total, price, ma50, ma200, held_contracts)
        mo_ok, mo_reason = lc.should_add_momentum(scores, held_delta, held_contracts)

        add_col1, add_col2 = st.columns(2)
        with add_col1:
            st.markdown("**Pullback add** (value-style, lower cost basis)")
            (st.success if pb_ok else st.info)(pb_reason)
        with add_col2:
            st.markdown("**Momentum add** (pyramid into strength, higher cost basis)")
            (st.success if mo_ok else st.info)(mo_reason)

        st.markdown("### Trim?")
        trim_reasons = lc.should_trim(unrealized_gain_pct, held_delta)
        if trim_reasons:
            for r in trim_reasons:
                st.warning(r)
        else:
            st.info("No trim signal right now.")

        st.markdown("### Sell / exit fully?")
        exit_reasons = lc.should_exit(scores, total, ma50, ma200, price)
        if exit_reasons:
            for r in exit_reasons:
                st.error(r)
        else:
            st.info("No exit signal right now — trend and score still support holding.")

        st.markdown("### Convert to shares?")
        convert_reasons = lc.should_convert(held_delta, held_expiry_days,
                                             held_value_per_contract / 100, intrinsic)
        if convert_reasons:
            for r in convert_reasons:
                st.warning(r)
        else:
            st.info("Still meaningful time value / optionality left — no need to convert yet.")

# --- Value estimator tab -----------------------------------------------------
with tab_estimator:
    st.markdown("### How the option's value changes as TSLA's price moves")
    default_strike = float(best_option["strike"]) if best_option is not None else round(price, -1)
    default_days = int(best_option["days_to_expiry"]) if best_option is not None else 365
    default_iv = float(best_option["impliedVolatility"]) if best_option is not None else (current_iv or 0.5)

    e1, e2, e3 = st.columns(3)
    est_strike = e1.number_input("Strike", value=default_strike, step=5.0)
    est_days = e2.number_input("Days to expiry (from today)", value=default_days, step=1, min_value=1)
    est_iv = e3.slider("Implied volatility assumption", 0.10, 1.20, float(default_iv), 0.01)

    spot_grid = np.linspace(price * 0.5, price * 1.8, 60)
    horizons = [0, 90, 180, min(360, est_days)]
    horizons = sorted(set(h for h in horizons if h <= est_days))
    surface = lc.value_surface(est_strike, est_iv, risk_free_rate, spot_grid, est_days, horizons)

    fig3 = go.Figure()
    for h in horizons:
        label = "Today" if h == 0 else f"+{h}d"
        fig3.add_trace(go.Scatter(x=spot_grid, y=surface[h], mode="lines", name=label))
    fig3.add_vline(x=price, line_dash="dot", annotation_text="Current price")
    fig3.update_layout(title=f"Theoretical value of the ${est_strike:.0f} call as TSLA price moves",
                        xaxis_title="TSLA Price", yaxis_title="Option Value ($/share, x100 per contract)")
    st.plotly_chart(fig3, use_container_width=True)
    st.caption(
        "Uses Black-Scholes with the IV assumption above, held constant across price scenarios "
        "(in reality IV itself moves with price and time — this is a simplification)."
    )
