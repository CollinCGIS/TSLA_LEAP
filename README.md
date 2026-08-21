# TSLA LEAPS Investment System

A rules-based decision-support dashboard for buying and managing TSLA LEAPS
calls. Not financial advice — verify everything before trading.

## Files
- `leaps_core.py` — all math/decision logic (Black-Scholes, scoring, sizing,
  add/trim/sell/convert rules). No network or UI code, so it's independently
  unit-testable.
- `app.py` — Streamlit app: fetches live data via `yfinance` (no API key
  needed) and renders the UI.
- `requirements.txt` — dependencies.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy for free (Streamlit Community Cloud)
1. Push this folder to a GitHub repo.
2. Go to https://share.streamlit.io, sign in, "New app", point it at the
   repo and `app.py`.
3. (Optional but recommended) Add a free FRED API key — see below.

## Live risk-free rate (FRED) — optional but recommended
The Black-Scholes calculations need a risk-free rate. By default this is a
static constant in `leaps_core.py`. To use a real, current 1-year Treasury
yield instead:
1. Get a free key at https://fredaccount.stlouisfed.org (instant, no
   approval wait).
2. Set it as an environment variable named `FRED_API_KEY` — locally via
   `export FRED_API_KEY=...` before running, or on Streamlit Community
   Cloud via the app's **Settings → Secrets**:
   ```toml
   FRED_API_KEY = "your_key_here"
   ```
3. The Overview tab shows which rate is actually in use (live FRED value
   with its as-of date, or the static fallback) so it's never ambiguous.

If no key is set, the app still runs fine on the fallback constant — this
is a quality upgrade, not a hard requirement.

## Two-tier scoring — timing vs. contract selection
There are two separate ratings in the dashboard, answering two different
questions:
1. **Composite entry-timing score** (Overview tab, 0-100 + letter grade) —
   "should I buy a LEAP *at all* right now?" Built from the 9 ticker-level
   factors (trend, MA cross, RSI, IV vs. realized vol, volume, relative
   strength, sector, sentiment, earnings quality).
2. **Per-contract score** (LEAP Selection tab, 0-100 + letter grade, one per
   candidate) — "*given* that I'm buying, which specific strike/expiration
   is the best pick?" Built from delta fit to target (50%), liquidity —
   open interest + bid-ask spread tightness (30%), and how cheap this
   contract's IV is relative to the other candidates in the same window
   (20%). The whole candidate table is ranked and color-coded by this score
   (green = better) so the best pick is visually obvious, not just the top
   row of an unranked table.

These intentionally don't share a scoring formula — a contract with a
perfect delta fit but a 20-wide bid-ask spread on 3 lots of open interest is
a bad pick even in a great overall setup, and that's a liquidity problem,
not a market-timing problem.

## What changed from the original draft, and why

The original script had a syntax error (unclosed bracket in the options
filter) and several logic bugs that would have made the "system" part of
the system silently wrong even if it ran:

- **Alpha Vantage integration was fictional** — `AlphaVantage(...)`,
  `.get_data()`, `.get_rsi()` aren't real methods of the `alpha_vantage`
  package. Dropped in favor of `yfinance` alone, which needs no key and is
  reliable enough for this use case.
- **`options['days_until_expiry']` was referenced but never computed**, and
  `stock.option_chain()` was called with no expiration date, so it only
  ever saw the *nearest* expiration — never anything close to 12–14 months
  out. Now the app iterates every available expiration, keeps the ones in
  the LEAPS window, and computes days-to-expiry from the real calendar date.
- **Delta/gamma used a single hardcoded IV (0.3) for every strike** instead
  of each option's own market-implied IV. Now Black-Scholes is run per-row
  using that row's actual `impliedVolatility`.
- **`score_trend = score_trend(...)` overwrote the function itself** with
  its own return value — harmless on a single run but a landmine the moment
  the app reruns (which Streamlit does constantly). Functions now live only
  in `leaps_core.py`; the app stores results in differently-named variables.
- **`score_iv` rewarded *high* IV** — backwards for a LEAPS *buyer*, who
  wants to pay for volatility when it's cheap, not rich. Replaced with an
  IV-vs-realized-volatility ratio: options cheap relative to what the stock
  has actually been doing score well; options priced rich against recent
  realized moves score poorly.
- **`score_volume` scored a raw volume figure against 0** — volume is never
  negative, so this always returned 100. Replaced with *relative* volume
  (today vs. 20-day average), which actually says something about
  conviction behind a move.
- **`score_sector` referenced a `sector_change` column that was never
  created** — an immediate `KeyError`. There's no reliable free feed for a
  clean sector-peer basket, so this was replaced with TSLA's trailing
  3-month return vs. SPY (relative strength vs. the broad market) as the
  systematic stand-in, documented as such in the UI.
- **The weighted total score was multiplied a second time by the sum of the
  weights**, silently deflating it. Weights now sum to exactly 1.0 and are
  applied once.
- **No position sizing, add/trim/sell, or convert-to-shares logic existed**
  at all, despite being half the stated goal. These are now implemented in
  `leaps_core.py` (see below) and driven from your sidebar inputs.
- **No caching** — every rerun would have hammered Yahoo Finance and Alpha
  Vantage. Data fetches are now `@st.cache_data(ttl=900)` (15 min).
- **No error handling** around option chain / history calls that fail or
  return too little data (e.g., not enough bars for a 200-day MA); the app
  now checks and shows a clear message instead of crashing.

## How each of your original questions is answered

- **When to buy**: composite score ≥ 70 **and** the trend + MA-cross
  factors individually confirm (Overview tab).
- **Which LEAP to buy**: 11–14.5 month window, delta 0.65–0.78 (target
  ~0.71), filtered to contracts with open interest, sorted by closeness to
  target delta (LEAP Selection tab).
- **How much to buy**: smaller of a risk-based size (stop-loss-implied loss
  ≤ your risk % of portfolio) and a hard allocation cap (LEAP Selection tab,
  driven by your sidebar risk settings).
- **When to add**: trend still intact, price pulled back near the 50-day
  MA, score still holding up (Manage Position tab).
- **When to trim/sell**: trim on ~2x gain or delta ≥ 0.90; full exit on a
  death cross, price below the 200-day MA, or score collapse (Manage
  Position tab).
- **When to convert to shares**: delta ≥ 0.90 with under ~90 days left, or
  extrinsic value under 5% of the option's price (Manage Position tab).
- **Estimating option value as price changes**: interactive Black-Scholes
  value curve across a spot-price grid at several time horizons (Value
  Estimator tab).
- **Objective scoring**: nine weighted 0–100 factors (trend, MA cross, RSI,
  IV vs. realized vol, relative volume, relative strength vs. SPY, sector
  performance, blended sentiment, earnings surprise quality) combined into
  one composite score (Overview tab). See `leaps_core.SCORE_WEIGHTS` for
  current weights.

## Alpha Vantage integrations — optional but recommended
Three more factors use Alpha Vantage's free API, since it has real data yfinance
doesn't: actual sector performance, news sentiment, and earnings surprise data.
1. Get a free key at https://www.alphavantage.co/support/#api-key (instant).
2. Set it as `ALPHAVANTAGE_API_KEY` — same mechanism as `FRED_API_KEY` above
   (env var locally, `Settings → Secrets` on Streamlit Community Cloud).
3. Without a key, all three factors show as neutral (50) and the app tells
   you why in the Overview tab — nothing breaks.

What each one does:
- **Sector Performance** (`SECTOR` endpoint) — TSLA's actual GICS sector
  (Consumer Discretionary) real-time performance, replacing the old
  SPY-relative-strength-as-sector-proxy with the real thing. Relative
  Strength vs. SPY is kept as its own separate factor (market-wide context
  is a different signal than sector-specific strength).
- **Sentiment** (`NEWS_SENTIMENT` endpoint) — relevance-weighted news
  sentiment, blended with the existing put/call volume ratio. If only one
  source is available in a given refresh, the blend uses just that one
  rather than diluting toward neutral.
- **Earnings Surprise Quality** (`EARNINGS` endpoint) — the most recent
  quarter's actual-vs-estimate EPS surprise, as an objective proxy for
  "earnings guidance quality." True forward guidance text isn't available
  as clean structured data from any free source, so this is the closest
  objective substitute.
- **Earnings date** (`EARNINGS_CALENDAR` endpoint) — now the *preferred*
  source for the earnings blackout gate, since it's more reliable than
  yfinance's earnings-date lookup; falls back to yfinance automatically if
  Alpha Vantage is unavailable.

**Rate limit budget**: the free tier caps out at 25 requests/day. This app
makes at most 4 Alpha Vantage calls per full data refresh (sector, news,
earnings surprise, earnings calendar), each cached well past its natural
refresh rate (2-24h depending on how often the underlying data actually
changes) specifically to stay inside that budget even with multiple people
using a shared deployment — Streamlit's cache is shared across all users of
one deployed instance, not per-visitor.

## Reconciling against the fuller spec doc

A later version of the spec added earnings timing, put/call sentiment, and a
few rules that conflicted with this implementation's original choices. Here's
what changed and what didn't, and why:

- **Earnings blackout — added.** No new LEAP entry within 30 days of a
  report, as a hard gate (`is_within_earnings_blackout`), not a scored
  factor — one earnings print can invalidate the whole technical picture
  overnight, so it shouldn't just be one of seven inputs averaged together.
  Pulled live from `yfinance`'s `get_earnings_dates()`.
- **Put/call sentiment — added as a real factor**, not a placeholder: the
  nearest-expiration put/call *volume* ratio is fetched live and scored
  (`score_sentiment`), and folded into the composite score at 15% weight
  (weights rebalanced to sum to 1.0 with the new factor included).
- **RSI philosophy — kept the original approach, not the spec doc's.** The
  spec scores oversold RSI as bullish (buy weakness). This tool instead
  peaks the RSI score at 50 and penalizes both extremes, because "buy
  oversold" is in tension with the entry rule that also requires an intact
  uptrend — you don't want to load up on a stock making fresh lows just
  because RSI is stretched, even if the longer moving averages haven't
  rolled over yet.
- **IV philosophy — kept the original approach, not the spec doc's.** The
  spec's scoring table treats high IV as bullish (100 = high). This tool
  scores IV *relative to realized volatility*, rewarding cheap options —
  as a LEAPS **buyer**, paying up for rich implied volatility works against
  you regardless of how bullish the underlying setup looks.
- **Adding to the position — both strategies exposed, not merged.** The
  spec's "add on strength" (delta rising, price already above strike) and
  this tool's original "add on a pullback near the 50-day MA" are genuinely
  different strategies with different cost-basis and risk implications, so
  rather than silently picking one, the Manage Position tab now shows both
  `should_add_pullback` and `should_add_momentum` side by side. Use the one
  that matches how you actually want to build the position.
- **Gamma > 1.0 filter — not implemented.** A single option's gamma is
  measured in fractions of a delta point per $1 move (real-world LEAPS
  gamma is typically well under 0.02), so a ">1.0" threshold as written
  would filter out every real contract. This looks like a units mix-up in
  the spec rather than an intentional rule, so it wasn't implemented as a
  hard filter — gamma is still computed and shown for reference.
- **"12-14 months (144-168 days)" — resolved in favor of the month figure.**
  144-168 days is ~4.7-5.5 months, not 12-14; the app uses 335-440 calendar
  days, matching the stated month range.

## Known limitations worth knowing about
- IV-vs-realized-vol and relative-strength-vs-SPY are reasonable free-data
  proxies, not the same as a real IV-rank history or true sector-peer
  comparison — swap in a paid data feed if you want those exactly.
- The Black-Scholes model ignores dividends (fine for TSLA, which pays
  none) and assumes constant volatility — real option prices will deviate,
  especially around earnings.
- All thresholds (score cutoffs, delta band, pullback %, stop-loss %) are
  constants/sidebar inputs you should sanity-check and adjust to your own
  risk tolerance rather than trust blindly.
- **Yahoo Finance rate limiting** (`yfinance.exceptions.YFRateLimitError`):
  yfinance is an unofficial, keyless scraper of Yahoo's endpoints, not a
  supported API. Yahoo rate-limits it more aggressively on shared cloud IPs
  (like Streamlit Community Cloud's free tier, where many apps share the
  same outbound address) than on a home connection. Every yfinance call in
  `app.py` retries a couple of times with a short backoff and fails
  gracefully into an empty result with a clear on-screen explanation rather
  than crashing the app — but if Yahoo is actively blocking that shared IP
  range, no amount of retrying inside the app will force data through. If
  you see the rate-limit message repeatedly:
  - Wait a few minutes and refresh — this is usually temporary.
  - It's typically less frequent when running locally, since your home IP
    isn't shared with other apps' traffic.
  - If it becomes a persistent problem on your deployment, the real fix is
    a paid/keyed data source for the options chain (Tradier was flagged
    earlier in this project as the best free-to-start upgrade path) rather
    than working around yfinance further.
