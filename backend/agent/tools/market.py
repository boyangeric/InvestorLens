"""
Market data tool — current quote for a single ticker via yfinance.

yfinance scrapes Yahoo Finance. No API key, no rate limit for portfolio-scale
volume. Suitable for the demo; production would swap in Alpha Vantage / IEX /
ASX direct, but the tool's interface (`get_market_quote(ticker) -> MarketQuote`)
is provider-agnostic so the swap is local to this file.

Australian tickers use the `.AX` suffix (Qantas → `QAN.AX`). Resolution from
company name → ticker is the caller's responsibility; this module deliberately
stays pure data-fetch.
"""

import logging
from datetime import datetime, timezone

import yfinance as yf
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MarketQuote(BaseModel):
    """A snapshot of a single security at a point in time."""

    ticker: str
    name: str
    price: float
    currency: str
    change_pct_day: float | None = None
    market_cap: float | None = None
    fifty_two_week_low: float | None = None
    fifty_two_week_high: float | None = None
    as_of: str  # ISO-8601 UTC — when WE fetched it, not Yahoo's quote timestamp
    source: str = "yfinance"


def get_market_quote(ticker: str) -> MarketQuote | None:
    """
    Fetch a current quote for `ticker`. Returns None on any failure — typo,
    delisted symbol, Yahoo outage, etc. — so the caller can gracefully fall
    back to "live data unavailable" rather than crashing the whole agent run.
    """
    try:
        info = yf.Ticker(ticker).info
    except Exception:
        logger.exception("yfinance: lookup failed for %s", ticker)
        return None

    if not info:
        logger.warning("yfinance: empty info for %s", ticker)
        return None

    # yfinance returns wildly inconsistent shapes across symbols; the only
    # field we treat as required is a price. Everything else is best-effort.
    price = info.get("regularMarketPrice") or info.get("currentPrice")
    if price is None:
        logger.warning("yfinance: no price for %s", ticker)
        return None

    prev_close = info.get("regularMarketPreviousClose") or info.get("previousClose")
    change_pct = None
    if prev_close:
        try:
            change_pct = round((float(price) - float(prev_close)) / float(prev_close) * 100, 2)
        except (TypeError, ValueError, ZeroDivisionError):
            change_pct = None

    return MarketQuote(
        ticker=ticker,
        name=info.get("shortName") or info.get("longName") or ticker,
        price=float(price),
        currency=info.get("currency", "USD"),
        change_pct_day=change_pct,
        market_cap=info.get("marketCap"),
        fifty_two_week_low=info.get("fiftyTwoWeekLow"),
        fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
        as_of=datetime.now(timezone.utc).isoformat(),
    )
