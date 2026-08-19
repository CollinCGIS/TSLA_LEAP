"""
leaps_core.py
--------------
Pure calculation and decision-logic for the TSLA LEAPS investment system.

Deliberately has NO Streamlit and NO network calls in this file, so every
function here can be unit tested with plain numbers. app.py is the thin
Streamlit/yfinance layer that calls into this module.

This is a rules-based decision-support tool, not financial advice. All
thresholds are configurable and should be reviewed/adjusted by the user.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Constants (all overridable from the UI — these are just defaults)
# ---------------------------------------------------------------------------

# LEAPS window: 12-14 months out, expressed in calendar days with a small buffer
LEAPS_MIN_DAYS = 335   # ~11 months
LEAPS_MAX_DAYS = 440   # ~14.5 months

# Target delta band for the "which LEAP to buy" filter
DELTA_MIN = 0.65
DELTA_MAX = 0.78
DELTA_TARGET = 0.71  # preferred midpoint (0.70-0.72)

RISK_FREE_RATE = 0.045

BUY_SCORE_THRESHOLD = 70.0
TREND_GATE = 60.0       # trend + MA-cross must clear this even if total score is high
EXIT_SCORE_THRESHOLD = 30.0

DEFAULT_RISK_PER_TRADE = 0.02      # 2% of portfolio at risk per new trade
DEFAULT_STOP_LOSS_PCT = 0.35       # mental stop: exit if option loses 35% of premium
DEFAULT_MAX_ALLOCATION_PCT = 0.10  # never put more than 10% of portfolio in this trade

EARNINGS_BLACKOUT_DAYS = 30  # no new entries within this many days of an earnings report

SCORE_WEIGHTS = {
    "trend": 0.20,
    "ma_cross": 0.15,
    "rsi": 0.08,
    "iv": 0.12,
    "volume": 0.08,
    "rel_strength": 0.05,
    "sector": 0.10,
    "sentiment": 0.12,
    "earnings_quality": 0.10,
}
assert abs(sum(SCORE_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Black-Scholes (calls only — LEAPS system only trades long calls)
# ---------------------------------------------------------------------------

def bs_call_greeks(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Black-Scholes price/delta/gamma/theta for a European call.

    S: spot price, K: strike, T: time to expiry in YEARS, r: risk-free rate,
    sigma: annualized implied volatility (decimal, e.g. 0.45).

    Degenerates to intrinsic value at/after expiry or on bad inputs instead
    of raising, since live option chains occasionally have zero/NaN IV.
    """
    if S <= 0 or K <= 0:
        return {"price": 0.0, "delta": 0.0, "gamma": 0.0, "theta": 0.0}
    if T <= 0 or sigma is None or sigma <= 0 or np.isnan(sigma):
        intrinsic = max(S - K, 0.0)
        return {"price": intrinsic, "delta": 1.0 if S > K else 0.0, "gamma": 0.0, "theta": 0.0}

    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT

    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * sqrtT)
    theta = (
        -(S * norm.pdf(d1) * sigma) / (2 * sqrtT) - r * K * np.exp(-r * T) * norm.cdf(d2)
    ) / 365.0

    return {"price": float(price), "delta": float(delta), "gamma": float(gamma), "theta": float(theta)}


# ---------------------------------------------------------------------------
# Scoring functions — each returns a 0-100 score, higher = more favorable
# ---------------------------------------------------------------------------

def score_trend(price: float, ma20: float) -> float:
    """Price vs 20-day MA. +5% above MA -> 100, -5% below -> 0, linear/clipped."""
    if ma20 is None or ma20 <= 0 or np.isnan(ma20):
        return 50.0
    pct = (price - ma20) / ma20 * 100
    return float(np.clip(50 + pct * 10, 0, 100))


def score_ma_cross(ma50: float, ma200: float) -> float:
    """50/200-day MA spread. Golden-cross strength, not just a binary flag."""
    if ma200 is None or ma200 <= 0 or np.isnan(ma200):
        return 50.0
    pct = (ma50 - ma200) / ma200 * 100
    return float(np.clip(50 + pct * 15, 0, 100))


def score_rsi(rsi: Optional[float]) -> float:
    """Triangular score peaking at RSI=50; penalizes both overbought and oversold
    extremes, since LEAPS entries want steady trends, not exhaustion moves."""
    if rsi is None or np.isnan(rsi):
        return 50.0
    return float(np.clip(100 - abs(rsi - 50) * 5, 0, 100))


def score_iv_relative(current_iv: Optional[float], realized_vol: Optional[float]) -> float:
    """Cheapness of options vs recent realized volatility.
    ratio = IV / HV.  ~0.8 (cheap) -> 100,  ~1.5 (expensive) -> 0.
    NOTE: this is the opposite direction from "high IV = high score" — as a
    LEAPS *buyer* you want to pay for volatility when it's cheap relative to
    what the stock has actually been doing, not when it's rich.
    """
    if not current_iv or not realized_vol or realized_vol <= 0 or np.isnan(current_iv):
        return 50.0
    ratio = current_iv / realized_vol
    score = 100 - (ratio - 0.8) / (1.5 - 0.8) * 100
    return float(np.clip(score, 0, 100))


def score_relative_volume(rel_vol: Optional[float]) -> float:
    """Today's volume vs its 20-day average. Confirms conviction behind a move."""
    if rel_vol is None or np.isnan(rel_vol):
        return 50.0
    if rel_vol >= 1.5:
        return 100.0
    if rel_vol <= 0.5:
        return 20.0
    return float(20 + (rel_vol - 0.5) / 1.0 * 80)


def score_relative_strength(tsla_return: float, spy_return: float) -> float:
    """TSLA's trailing return vs SPY's, as a market-context / 'sector' proxy.
    (A clean free sector-peer-basket feed isn't reliably available, so relative
    strength vs the broad market is used as the systematic substitute.)"""
    spread_pts = (tsla_return - spy_return) * 100
    return float(np.clip(50 + spread_pts * 5, 0, 100))


def score_sentiment(put_call_volume_ratio: Optional[float]) -> float:
    """Near-term put/call volume ratio as a real (not placeholder) sentiment
    read. Lower ratio (more call volume than put volume) = more bullish
    positioning = higher score. ~0.5 or below -> 100 (bullish), ~1.2+ -> 0
    (bearish), linear/clipped between."""
    if put_call_volume_ratio is None or np.isnan(put_call_volume_ratio):
        return 50.0
    r = put_call_volume_ratio
    score = 100 - (r - 0.5) / (1.2 - 0.5) * 100
    return float(np.clip(score, 0, 100))


def is_within_earnings_blackout(days_to_next_earnings: Optional[int],
                                 blackout_days: int = EARNINGS_BLACKOUT_DAYS) -> bool:
    """Hard entry gate: don't open a new LEAP within `blackout_days` of an
    earnings report, regardless of how good the score looks — this is a
    binary rule, not a scored factor, since a single earnings print can
    invalidate the whole technical picture overnight."""
    if days_to_next_earnings is None:
        return False  # unknown earnings date -> don't block, but the UI should flag this
    return 0 <= days_to_next_earnings <= blackout_days


def score_sector_performance(sector_pct_points: Optional[float]) -> float:
    """Real sector performance (percentage points, e.g. -0.85 for -0.85%),
    from a source like Alpha Vantage's SECTOR endpoint. +3.3% -> 100 (sector
    running hot), -3.3% -> 0, linear/clipped between."""
    if sector_pct_points is None or np.isnan(sector_pct_points):
        return 50.0
    return float(np.clip(50 + sector_pct_points * 15, 0, 100))


def score_news_sentiment(avg_sentiment_score: Optional[float]) -> float:
    """News-based sentiment, e.g. from Alpha Vantage's NEWS_SENTIMENT
    (relevance-weighted average ticker_sentiment_score, roughly -1..1).
    Maps -1 (bearish) -> 0, +1 (bullish) -> 100."""
    if avg_sentiment_score is None or np.isnan(avg_sentiment_score):
        return 50.0
    return float(np.clip((avg_sentiment_score + 1) / 2 * 100, 0, 100))


def score_earnings_surprise(surprise_pct: Optional[float]) -> float:
    """Most recent quarter's EPS surprise (percentage points, e.g. 5.2 for a
    5.2% beat). Used as an objective proxy for 'earnings quality' since
    forward guidance text isn't available as clean structured data from free
    sources. ~+16.7% beat -> 100, ~-16.7% miss -> 0."""
    if surprise_pct is None or np.isnan(surprise_pct):
        return 50.0
    return float(np.clip(50 + surprise_pct * 3, 0, 100))


def blended_average(*scores: Optional[float]) -> float:
    """Average of whichever inputs are actually available; defaults to 50
    (neutral) only if every input is missing. Used to combine multiple
    sentiment reads (e.g. put/call positioning + news sentiment) without
    letting a missing source silently drag the blend toward 50."""
    vals = [s for s in scores if s is not None and not (isinstance(s, float) and np.isnan(s))]
    if not vals:
        return 50.0
    return float(np.mean(vals))


def compute_total_score(scores: dict) -> float:
    return round(sum(scores[k] * SCORE_WEIGHTS[k] for k in SCORE_WEIGHTS), 2)


def letter_grade(score: float) -> str:
    """Quick-scan grade for any 0-100 score (entry-timing or per-contract)."""
    if score >= 85:
        return "A"
    if score >= 70:
        return "B"
    if score >= 55:
        return "C"
    if score >= 40:
        return "D"
    return "F"


# ---------------------------------------------------------------------------
# Decision rules
# ---------------------------------------------------------------------------

def buy_signal(scores: dict, total: float, earnings_blackout: bool = False) -> bool:
    """When to buy a LEAP: composite score clears the bar AND the two trend
    factors individually confirm (a high score propped up by RSI/volume alone
    shouldn't trigger a buy), AND we're not inside the earnings blackout
    window — a hard override regardless of how good the score looks."""
    if earnings_blackout:
        return False
    return (
        total >= BUY_SCORE_THRESHOLD
        and scores["trend"] >= TREND_GATE
        and scores["ma_cross"] >= TREND_GATE
    )


# ---------------------------------------------------------------------------
# Per-contract scoring — "which one to buy" among already delta/window-
# filtered candidates, as distinct from the ticker-level "should I buy at
# all right now" composite score above.
# ---------------------------------------------------------------------------

CONTRACT_SCORE_WEIGHTS = {"delta_fit": 0.50, "liquidity": 0.30, "iv_cost": 0.20}
assert abs(sum(CONTRACT_SCORE_WEIGHTS.values()) - 1.0) < 1e-9


def score_contract_delta_fit(delta: float, target: float = DELTA_TARGET,
                              delta_min: float = DELTA_MIN, delta_max: float = DELTA_MAX) -> float:
    """How close this contract's delta is to the target, scaled to 0 at the
    edges of the acceptable band and 100 exactly at the target."""
    half_width = max(target - delta_min, delta_max - target)
    if half_width <= 0:
        return 100.0
    dist = abs(delta - target)
    return float(np.clip(100 * (1 - dist / half_width), 0, 100))


def score_contract_liquidity(open_interest: Optional[float], bid: Optional[float],
                              ask: Optional[float]) -> float:
    """Rewards deep open interest and a tight bid-ask spread — the two things
    that determine whether you can actually get filled near the quoted price.
    LEAPS are inherently less liquid than short-dated options, so the
    thresholds here are calibrated to that (500+ OI is 'plenty', not '5000+')."""
    oi_score = None
    if open_interest is not None and not np.isnan(open_interest):
        oi_score = float(np.clip(100 * min(open_interest / 500.0, 1.0), 0, 100))

    spread_score = None
    if bid is not None and ask is not None and not np.isnan(bid) and not np.isnan(ask):
        mid = (bid + ask) / 2
        if mid > 0 and ask >= bid >= 0:
            spread_pct = (ask - bid) / mid * 100
            # <=3% spread -> 100 (tight), >=15% -> 0 (wide/illiquid)
            spread_score = float(np.clip(100 - (spread_pct - 3) / (15 - 3) * 100, 0, 100))

    return blended_average(oi_score, spread_score)


def score_contract_iv_cost(contract_iv: Optional[float], pool_avg_iv: Optional[float]) -> float:
    """How this specific contract's IV compares to the other candidates in
    the same delta/expiry window — cheaper-than-peers scores higher. This is
    a *relative* read across today's candidates, separate from the ticker-
    level 'IV vs Realized Vol' entry-timing factor."""
    if not contract_iv or not pool_avg_iv or pool_avg_iv <= 0 or np.isnan(contract_iv):
        return 50.0
    ratio = contract_iv / pool_avg_iv
    score = 100 - (ratio - 0.9) / (1.3 - 0.9) * 100
    return float(np.clip(score, 0, 100))


def compute_contract_score(delta: float, open_interest: Optional[float], bid: Optional[float],
                            ask: Optional[float], contract_iv: Optional[float],
                            pool_avg_iv: Optional[float]) -> dict:
    """Combined 0-100 rating for a single candidate LEAP contract — this is
    'which one to buy' among the already-filtered candidates, distinct from
    the ticker-level composite score, which answers 'should I buy at all
    right now'. Returns the sub-scores too, for a transparent breakdown."""
    sub = {
        "delta_fit": score_contract_delta_fit(delta),
        "liquidity": score_contract_liquidity(open_interest, bid, ask),
        "iv_cost": score_contract_iv_cost(contract_iv, pool_avg_iv),
    }
    total = round(sum(sub[k] * CONTRACT_SCORE_WEIGHTS[k] for k in CONTRACT_SCORE_WEIGHTS), 1)
    return {**sub, "total": total, "grade": letter_grade(total)}


@dataclass
class PositionSizeResult:
    contracts_by_risk: int
    contracts_by_allocation: int
    contracts_recommended: int
    dollars_deployed: float
    risk_dollars_budgeted: float
    notes: list = field(default_factory=list)


def position_size(
    portfolio_value: float,
    premium: float,
    risk_per_trade: float = DEFAULT_RISK_PER_TRADE,
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT,
    max_allocation_pct: float = DEFAULT_MAX_ALLOCATION_PCT,
    contract_multiplier: int = 100,
) -> PositionSizeResult:
    """How much to buy: smaller of (a) contracts such that hitting the stop
    loses no more than risk_per_trade of the portfolio, and (b) contracts
    such that full premium never exceeds max_allocation_pct of the portfolio."""
    notes = []
    if premium <= 0 or portfolio_value <= 0:
        return PositionSizeResult(0, 0, 0, 0.0, 0.0, ["Invalid premium or portfolio value."])

    risk_dollars = portfolio_value * risk_per_trade
    by_risk = int(risk_dollars // (premium * contract_multiplier * stop_loss_pct))
    by_alloc = int((portfolio_value * max_allocation_pct) // (premium * contract_multiplier))

    recommended = max(0, min(by_risk, by_alloc))
    if recommended == 0:
        notes.append(
            "Even 1 contract exceeds your risk/allocation limits at this premium — "
            "reduce size elsewhere, raise risk tolerance, or pick a cheaper/lower-delta strike."
        )
    dollars_deployed = recommended * premium * contract_multiplier
    return PositionSizeResult(by_risk, by_alloc, recommended, dollars_deployed, risk_dollars, notes)


def should_add_pullback(scores: dict, total: float, price: float, ma50: float, ma200: float,
                         holding_contracts: int, pullback_band_pct: float = 0.03) -> tuple[bool, str]:
    """Value-style add: only ever add to a trend that's still intact, on a
    pullback toward support (near the 50-day MA), not on strength. Lower
    average cost basis, but risks adding into the start of a real breakdown
    if the pullback keeps going."""
    if holding_contracts <= 0:
        return False, "No existing position to add to."
    trend_intact = ma50 > ma200 and price > ma200
    if not trend_intact:
        return False, "Underlying trend is not intact — do not add here."
    near_ma50 = (ma50 * (1 - pullback_band_pct)) <= price <= (ma50 * (1 + pullback_band_pct))
    if not near_ma50:
        return False, "Price is not currently in the pullback/support zone near the 50-day MA."
    if total < 55:
        return False, f"Composite score ({total:.1f}) is too weak to justify adding size."
    return True, "Trend intact, price pulled back to the 50-day MA support zone, score holding up — a reasonable add point."


def should_add_momentum(scores: dict, held_delta: Optional[float],
                         holding_contracts: int, strength_gate: float = 75.0) -> tuple[bool, str]:
    """Momentum-style add: pyramid into strength once the held option's delta
    has climbed well into the money (>=0.85) and the trend/MA-cross scores
    are both strongly confirming. Higher average cost basis than a pullback
    add, but only adds when the thesis has already been validated by price."""
    if holding_contracts <= 0:
        return False, "No existing position to add to."
    if held_delta is None:
        return False, "No delta available for the held position."
    if held_delta < 0.85:
        return False, f"Held delta ({held_delta:.2f}) hasn't climbed enough yet to confirm a momentum add."
    if scores["trend"] < strength_gate or scores["ma_cross"] < strength_gate:
        return False, "Delta has climbed, but trend/MA-cross strength isn't confirming strongly enough."
    return True, (f"Held delta ({held_delta:.2f}) confirms the move is working and trend/MA-cross scores "
                  "are strong — a momentum-style add, at a higher cost basis than the original entry.")


def should_trim(unrealized_gain_pct: Optional[float], held_delta: Optional[float]) -> list[str]:
    """When to trim: profit-taking and/or delta drift toward stock-like behavior."""
    reasons = []
    if unrealized_gain_pct is not None and unrealized_gain_pct >= 100:
        reasons.append("Position value has roughly doubled — trim ~1/3 to lock in gains and reduce risk.")
    if held_delta is not None and held_delta >= 0.90:
        reasons.append("Delta is above 0.90 — the option now trades almost like stock; consider trimming.")
    return reasons


def should_exit(scores: dict, total: float, ma50: float, ma200: float, price: float) -> list[str]:
    """When to trim/sell fully: trend breakdown or thesis deterioration."""
    reasons = []
    if ma50 < ma200:
        reasons.append("Death cross (50-day MA below 200-day MA) — the primary trend has broken down.")
    if price < ma200:
        reasons.append("Price has closed below the 200-day MA — long-term trend is broken.")
    if total < EXIT_SCORE_THRESHOLD:
        reasons.append(f"Composite score has fallen below {EXIT_SCORE_THRESHOLD:.0f} — thesis has deteriorated.")
    return reasons


def should_convert(held_delta: Optional[float], days_to_expiry: Optional[int],
                    option_price: Optional[float], intrinsic_value: Optional[float]) -> list[str]:
    """When to convert the option into shares (or roll): once the option is
    deep enough ITM and close enough to expiry that you're mostly paying for
    a small sliver of time value rather than real optionality."""
    reasons = []
    if held_delta is not None and days_to_expiry is not None and held_delta >= 0.90 and days_to_expiry < 90:
        reasons.append(
            "Delta ≥ 0.90 with under 90 days left — extrinsic value is minimal. Consider exercising/"
            "converting to shares, or rolling into a new LEAP, rather than holding through fast time decay."
        )
    if option_price and intrinsic_value is not None and option_price > 0:
        extrinsic = max(option_price - intrinsic_value, 0.0)
        if extrinsic / option_price < 0.05:
            reasons.append(
                "Less than 5% of the option's value is time premium — it's economically close to owning "
                "shares outright; converting removes the residual (small) optionality you're still paying for."
            )
    return reasons


# ---------------------------------------------------------------------------
# Option value estimator (for the "how does value change with price" tool)
# ---------------------------------------------------------------------------

def value_surface(K: float, sigma: float, r: float, spot_grid: np.ndarray,
                   days_to_expiry_now: int, horizon_days: list[int]) -> dict:
    """For a fixed strike/IV, compute option value across a grid of future
    spot prices, at several forward time horizons (e.g. today, +90d, +180d,
    at expiry). Returns {horizon_days: np.array(values aligned to spot_grid)}.
    """
    out = {}
    for h in horizon_days:
        remaining_days = max(days_to_expiry_now - h, 0)
        T = remaining_days / 365.0
        vals = np.array([bs_call_greeks(S, K, T, r, sigma)["price"] for S in spot_grid])
        out[h] = vals
    return out
