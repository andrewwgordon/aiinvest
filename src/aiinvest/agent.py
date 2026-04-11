"""AI Investment Agent for portfolio analysis and recommendations."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from tvscreener import FilterOperator, Market, StockField, StockScreener

load_dotenv()

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"


@dataclass
class StockSnapshot:
    """Represents a snapshot of a stock position."""
    ticker: str
    symbol: str
    price: Optional[float]
    shares: float
    position_value: Optional[float]
    performance_3m: Optional[float]
    performance_ytd: Optional[float]
    analyst_rating: Optional[float]


class OpenRouterError(RuntimeError):
    """Exception raised for OpenRouter API errors."""
    pass


class OpenRouterClient:
    """Client for interacting with the OpenRouter API."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL):
        """Initialize the OpenRouter client.

        Args:
            api_key: The OpenRouter API key.
            model: The model to use for completions.

        Raises:
            ValueError: If api_key is not provided.
        """
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")
        self.api_key = api_key
        self.model = model

    def create_completion(
        self, messages: List[Dict[str, str]], temperature: float = 0.2
    ) -> Dict[str, Any]:
        """Create a completion using the OpenRouter API.

        Args:
            messages: List of message dictionaries.
            temperature: Sampling temperature for the model.

        Returns:
            The API response as a dictionary.

        Raises:
            OpenRouterError: If the API request fails.
        """
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            OPENROUTER_URL, headers=headers, json=payload, timeout=30
        )
        if response.status_code != 200:
            raise OpenRouterError(
                f"OpenRouter request failed ({response.status_code}): "
                f"{response.text}"
            )
        return response.json()


class AIInvestmentAgent:
    """AI-powered investment agent for portfolio analysis and recommendations."""

    def __init__(
        self, openrouter_api_key: Optional[str] = None, model: str = DEFAULT_MODEL
    ):
        """Initialize the AI Investment Agent.

        Args:
            openrouter_api_key: OpenRouter API key. If None, uses environment
                variable.
            model: The LLM model to use for analysis.
        """
        self.api_key = openrouter_api_key or os.getenv("OPENROUTER_API_KEY")
        self.client = OpenRouterClient(self.api_key, model=model)

    def _parse_strategy_filters(self, strategy: str) -> Dict[str, Any]:
        """Parse natural language strategy into markets and filters.

        Args:
            strategy: Natural language trading strategy.

        Returns:
            Dictionary with 'markets' and 'filters' keys.
        """
        filters = []
        markets = []

        # Index-specific parsing (more specific before general market keywords)
        index_patterns = {
            r'ftse\s*100': (Market.UK, []),  # FTSE 100: large caps
            r'ftse\s*250': (
                Market.UK,
                [
                    (StockField.MARKET_CAPITALIZATION,
                     FilterOperator.ABOVE, 500_000_000),  # >500M
                    (StockField.MARKET_CAPITALIZATION,
                     FilterOperator.BELOW, 15_000_000_000),  # <15B
                ]
            ),
            r'ftse\s*350': (
                Market.UK,
                [(StockField.MARKET_CAPITALIZATION,
                  FilterOperator.ABOVE, 200_000_000)]  # >200M
            ),
            r's&p\s*500': (
                Market.AMERICA,
                [(StockField.MARKET_CAPITALIZATION,
                  FilterOperator.ABOVE, 2_000_000_000)]  # >2B
            ),
            r's&p\s*100': (
                Market.AMERICA,
                [(StockField.MARKET_CAPITALIZATION,
                  FilterOperator.ABOVE, 10_000_000_000)]  # >10B
            ),
            r'nasdaq\s*100': (Market.AMERICA, []),  # Tech-heavy, no cap filter
            r'dax\s*40': (
                Market.GERMANY,
                [(StockField.MARKET_CAPITALIZATION,
                  FilterOperator.ABOVE, 5_000_000_000)]  # >5B
            ),
            r'cac\s*40': (
                Market.FRANCE,
                [(StockField.MARKET_CAPITALIZATION,
                  FilterOperator.ABOVE, 2_000_000_000)]  # >2B
            ),
            r'ibex\s*35': (
                Market.SPAIN,
                [(StockField.MARKET_CAPITALIZATION,
                  FilterOperator.ABOVE, 1_000_000_000)]  # >1B
            ),
            r'nikkei\s*225': (
                Market.JAPAN,
                [(StockField.MARKET_CAPITALIZATION,
                  FilterOperator.ABOVE, 1_000_000_000)]  # >1B
            ),
        }

        strategy_lower = strategy.lower()
        for pattern, (market, additional_filters) in index_patterns.items():
            if re.search(pattern, strategy_lower, re.IGNORECASE):
                markets.append(market)
                filters.extend(additional_filters)
                break  # Take first specific index match

        # If no specific index found, fall back to general market keywords
        if not markets:
            market_keywords = {
                Market.UK: ["ftse", "london", "uk", "british", "united kingdom"],
                Market.AMERICA: ["s&p", "nasdaq", "nyse", "america", "us",
                                 "usa", "united states"],
                Market.GERMANY: ["dax", "germany", "german"],
                Market.FRANCE: ["cac", "france", "french"],
                Market.SPAIN: ["ibex", "spain", "spanish"],
                Market.ITALY: ["ftse mib", "italy", "italian"],
                Market.NETHERLANDS: ["aex", "netherlands", "dutch"],
                Market.SWEDEN: ["omx", "sweden", "swedish"],
                Market.DENMARK: ["omxc", "denmark", "danish"],
                Market.NORWAY: ["oslo", "norway", "norwegian"],
                Market.FINLAND: ["omxh", "finland", "finnish"],
                Market.JAPAN: ["nikkei", "japan", "japanese"],
                Market.CHINA: ["shanghai", "shenzhen", "china", "chinese"],
                Market.HONGKONG: ["hang seng", "hong kong"],
                Market.AUSTRALIA: ["asx", "australia", "australian"],
                Market.CANADA: ["tsx", "canada", "canadian"],
                Market.INDIA: ["nifty", "sensex", "india", "indian"],
                Market.BRAZIL: ["bovespa", "brazil", "brazilian"],
            }

            def keyword_in_text(text: str, keyword: str) -> bool:
                """Check if keyword appears as a whole word in text."""
                return re.search(rf"\b{re.escape(keyword)}\b", text) is not None

            for market, keywords in market_keywords.items():
                if any(keyword_in_text(strategy_lower, kw) for kw in keywords):
                    markets.append(market)
                    break  # For now, take the first match

        # Sector parsing and market fallback when no explicit market provided
        sector_patterns = {
            r'\btech(?:nology)?\b': (Market.AMERICA, "Technology"),
            r'\bsoftware\b': (Market.AMERICA, "Technology"),
            r'\bsemiconductor\b': (Market.AMERICA, "Technology"),
            r'\binternet\b': (Market.AMERICA, "Technology"),
            r'\bcloud\b': (Market.AMERICA, "Technology"),
            r'\bconsumer(?:\s+cyclical|\s+staples)?\b': (Market.AMERICA,
                                                         "Consumer"),
            r'\bretail\b': (Market.AMERICA, "Consumer"),
            r'\be-?commerce\b': (Market.AMERICA, "Consumer"),
            r'\bindustrial\b': (Market.UK, "Industrial"),
            r'\bmanufacturing\b': (Market.UK, "Industrial"),
            r'\benergy\b': (Market.AMERICA, "Energy"),
            r'\brenewable\b': (Market.AMERICA, "Energy"),
            r'\bhealthcare\b': (Market.AMERICA, "Healthcare"),
            r'\bbiotech\b': (Market.AMERICA, "Healthcare"),
            r'\bpharma\b': (Market.AMERICA, "Healthcare"),
            r'\bfinancial\b': (Market.UK, "Financial"),
            r'\bbank(?:ing)?\b': (Market.UK, "Financial"),
            r'\binsurance\b': (Market.UK, "Financial"),
            r'\butilities\b': (Market.AMERICA, "Utilities"),
            r'\bmaterial(?:s)?\b': (Market.AMERICA, "Materials"),
            r'\breal estate\b': (Market.AMERICA, "Real Estate"),
            r'\bproperty\b': (Market.AMERICA, "Real Estate"),
            r'\btelecom(?:munications)?\b': (Market.AMERICA,
                                             "Communication Services"),
            r'\bcommunications?\b': (Market.AMERICA, "Communication Services"),
        }
        sector_match = None
        for pattern, (sector_market, sector_name) in sector_patterns.items():
            if re.search(pattern, strategy_lower):
                sector_match = (sector_market, sector_name)
                break

        if sector_match:
            sector_market, sector_name = sector_match
            if not markets:
                markets.append(sector_market)
            filters.append((StockField.SECTOR, FilterOperator.MATCH,
                           sector_name))

        # Valuation filters
        pe_match = re.search(
            r'(?:price.to.earnings|P/E).*below\s+(\d+(?:\.\d+)?)',
            strategy, re.IGNORECASE
        )
        if pe_match:
            filters.append((StockField.PRICE_TO_EARNINGS_RATIO_TTM,
                           FilterOperator.BELOW, float(pe_match.group(1))))

        pb_match = re.search(
            r'(?:price.to.book|P/B).*below\s+(\d+(?:\.\d+)?)',
            strategy, re.IGNORECASE
        )
        if pb_match:
            filters.append((StockField.PRICE_TO_BOOK_MRQ,
                           FilterOperator.BELOW, float(pb_match.group(1))))

        dy_match = re.search(
            r'dividend.yields?.*above\s+(\d+(?:\.\d+)?)%',
            strategy, re.IGNORECASE
        )
        if dy_match:
            filters.append((StockField.DIVIDENDS_YIELD,
                           FilterOperator.ABOVE, float(dy_match.group(1))))

        # Additional filters
        mc_match = re.search(
            r'market.cap.*above\s+(\d+(?:\.\d+)?)([bB]|[mM]|[kK])?',
            strategy, re.IGNORECASE
        )
        if mc_match:
            value = float(mc_match.group(1))
            unit = mc_match.group(2)
            if unit:
                multiplier = {'b': 1e9, 'm': 1e6, 'k': 1e3}.get(unit.lower(), 1)
                value *= multiplier
            filters.append((StockField.MARKET_CAPITALIZATION,
                           FilterOperator.ABOVE, value))

        vol_match = re.search(
            r'(?:average.volume|volume).*above\s+(\d+(?:\.\d+)?)([mM]|[kK])?',
            strategy, re.IGNORECASE
        )
        if vol_match:
            value = float(vol_match.group(1))
            unit = vol_match.group(2)
            if unit:
                multiplier = {'m': 1e6, 'k': 1e3}.get(unit.lower(), 1)
                value *= multiplier
            filters.append((StockField.AVERAGE_VOLUME_30_DAY,
                           FilterOperator.ABOVE, value))

        perf_match = re.search(
            r'(?:1.?year|yearly).performance.*above\s+(-?\d+(?:\.\d+)?)%',
            strategy, re.IGNORECASE
        )
        if perf_match:
            filters.append((StockField.YEARLY_PERFORMANCE,
                           FilterOperator.ABOVE, float(perf_match.group(1))))

        roe_match = re.search(
            r'ROE.*above\s+(\d+(?:\.\d+)?)%', strategy, re.IGNORECASE
        )
        if roe_match:
            filters.append((StockField.RETURN_ON_EQUITY_TTM,
                           FilterOperator.ABOVE, float(roe_match.group(1))))

        roa_match = re.search(
            r'ROA.*above\s+(\d+(?:\.\d+)?)%', strategy, re.IGNORECASE
        )
        if roa_match:
            filters.append((StockField.RETURN_ON_ASSETS_TTM,
                           FilterOperator.ABOVE, float(roa_match.group(1))))
        return {"markets": markets, "filters": filters}

    def _fetch_buy_candidates(
        self, parsed_filters: Dict[str, Any], max_candidates: int
    ) -> List[Dict[str, Any]]:
        """Fetch buy candidates from tvscreener based on parsed filters.

        Args:
            parsed_filters: Dictionary with markets and filters from parsing.
            max_candidates: Maximum number of candidates to return.

        Returns:
            List of candidate stock dictionaries.
        """
        screener = StockScreener()
        if parsed_filters["markets"]:
            screener.set_markets(*parsed_filters["markets"])
        for field, op, value in parsed_filters["filters"]:
            screener.add_filter(field, op, value)
        screener.select(
            StockField.PRICE,
            StockField.PRICE_TO_EARNINGS_RATIO_TTM,
            StockField.PRICE_TO_BOOK_MRQ,
            StockField.DIVIDENDS_YIELD,
            StockField.YEARLY_PERFORMANCE,
            StockField.MARKET_CAPITALIZATION,
            StockField.AVERAGE_VOLUME_30_DAY,
            StockField.RETURN_ON_EQUITY_TTM,
            StockField.RETURN_ON_ASSETS_TTM,
            StockField.NAME,
            StockField.SECTOR,
            StockField.DESCRIPTION,
        )
        screener.sort_by(StockField.MARKET_CAPITALIZATION, False)  # Descending
        results = screener.get().to_dict("records")
        return results[:max_candidates]

    def _fetch_stock_snapshot(self, ticker: str, shares: float) -> StockSnapshot:
        """Fetch current stock data for a ticker.

        Args:
            ticker: Stock ticker symbol.
            shares: Number of shares held.

        Returns:
            StockSnapshot with current data.

        Raises:
            ValueError: If no data found for ticker.
        """
        screener = StockScreener()
        screener.search(ticker)
        screener.select(
            StockField.PRICE,
            StockField.MONTH_PERFORMANCE_3,
            StockField.YTD_PERFORMANCE,
            StockField.RECOMMENDATION_MARK,
        )
        results = screener.get().to_dict("records")
        if not results:
            raise ValueError(f"No data found for ticker {ticker}")

        symbol_record = self._choose_best_match(results, ticker)
        price = self._safe_float(symbol_record.get("Price"))
        position_value = price * shares if price is not None else None

        return StockSnapshot(
            ticker=ticker.upper(),
            symbol=symbol_record.get("Symbol", ticker),
            price=price,
            shares=shares,
            position_value=position_value,
            performance_3m=self._safe_float(symbol_record.get("3-Month Performance")),
            performance_ytd=self._safe_float(symbol_record.get("YTD Performance")),
            analyst_rating=self._safe_float(symbol_record.get("Analyst Rating")),
        )

    @staticmethod
    def _choose_best_match(
        records: List[Dict[str, Any]], ticker: str
    ) -> Dict[str, Any]:
        """Choose the best matching record for a ticker from search results.

        Args:
            records: List of stock records from search.
            ticker: The ticker to match.

        Returns:
            The best matching record.
        """
        normalized = ticker.upper()
        exact_matches = [
            r for r in records
            if isinstance(r.get("Symbol"), str) and
            r["Symbol"].upper().endswith(f":{normalized}")
        ]
        if exact_matches:
            return exact_matches[0]
        return records[0]

    @staticmethod
    def _safe_float(value: Any) -> Optional[float]:
        """Safely convert value to float, returning None on failure.

        Args:
            value: Value to convert.

        Returns:
            Float value or None if conversion fails.
        """
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _build_prompt(
        self,
        portfolio: Dict[str, float],
        cash_balance: float,
        strategy: str,
        snapshots: List[StockSnapshot],
        buy_candidates: List[Dict[str, Any]],
    ) -> str:
        """Build the LLM prompt with portfolio and candidate data.

        Args:
            portfolio: Dictionary of ticker to shares.
            cash_balance: Available cash.
            strategy: Original strategy text.
            snapshots: Current portfolio snapshots.
            buy_candidates: Filtered buy candidates.

        Returns:
            Formatted prompt string for LLM.
        """
        holdings_summary = []
        for snapshot in snapshots:
            holdings_summary.append(
                f"- {snapshot.ticker}: shares={snapshot.shares}, "
                f"price={snapshot.price or 'N/A'}, "
                f"position_value={snapshot.position_value or 'N/A'}, "
                f"3M_perf={snapshot.performance_3m or 'N/A'}, "
                f"YTD_perf={snapshot.performance_ytd or 'N/A'}, "
                f"analyst_rating={snapshot.analyst_rating or 'N/A'}"
            )

        if not holdings_summary:
            holdings_summary = ["No existing holdings."]

        candidates_summary = []
        for candidate in buy_candidates:
            candidates_summary.append(
                f"- {candidate.get('Symbol', 'Unknown')}: "
                f"price={candidate.get('Price', 'N/A')}, "
                f"P/E={candidate.get('Price to Earnings Ratio (TTM)', 'N/A')}, "
                f"P/B={candidate.get('Price to Book (MRQ)', 'N/A')}, "
                f"Div Yield={candidate.get('Dividends Yield', 'N/A')}%, "
                f"1Y Perf={candidate.get('Yearly Performance', 'N/A')}%, "
                f"Market Cap={candidate.get('Market Capitalization', 'N/A')}, "
                f"Avg Vol={candidate.get('Average Volume (30 day)', 'N/A')}, "
                f"ROE={candidate.get('Return on Equity (TTM)', 'N/A')}%, "
                f"ROA={candidate.get('Return on Assets (TTM)', 'N/A')}%, "
                f"Company={candidate.get('Description', 'N/A')[:50]}, "
                f"Sector={candidate.get('Sector', 'N/A')}"
            )

        return (
            "You are an AI investment analyst. Analyze the current portfolio, "
            "cash balance, and natural-language trading strategy. "
            "Provide clear recommendations for each existing holding: sell, "
            "hold, or trim. Then provide buy recommendations from the provided "
            "candidate stocks when there is available cash. "
            "Base recommendations on the provided market snapshot and strategy, "
            "and keep the advice practical and portfolio-aware.\n\n"
            "Trading strategy:\n"
            f"{strategy}\n\n"
            "Portfolio snapshot:\n"
            f"Cash available: ${cash_balance:,.2f}\n"
            "Holdings:\n"
            + "\n".join(holdings_summary)
            + "\n\n"
            "Buy candidates (filtered by strategy):\n"
            + "\n".join(candidates_summary)
            + "\n\n"
            "Respond with a concise recommendation list. Use a structure with "
            "sections for SELL, HOLD, and BUY, and mention reasons where appropriate."
        )

    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the LLM response into structured recommendations.

        Args:
            response: Raw LLM API response.

        Returns:
            Parsed response with sell/hold/buy lists.

        Raises:
            OpenRouterError: If response structure is invalid.
        """
        message = ""
        try:
            message = response["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError):
            raise OpenRouterError("Invalid OpenRouter response structure")

        parsed = {
            "text": message.strip(),
            "sell": [],
            "hold": [],
            "buy": [],
            "raw": response,
        }

        for line in message.splitlines():
            normalized = line.strip().lower()
            if (normalized.startswith("sell") or
                    normalized.startswith("- sell") or
                    normalized.startswith("* sell")):
                parsed["sell"].append(line.strip())
            elif (normalized.startswith("hold") or
                  normalized.startswith("- hold") or
                  normalized.startswith("* hold")):
                parsed["hold"].append(line.strip())
            elif (normalized.startswith("buy") or
                  normalized.startswith("- buy") or
                  normalized.startswith("* buy")):
                parsed["buy"].append(line.strip())

        return parsed

    def recommend(
        self,
        portfolio: Dict[str, float],
        cash_balance: float,
        strategy: str,
        max_buy_candidates: int = 10,
    ) -> Dict[str, Any]:
        """Generate investment recommendations based on strategy.

        Args:
            portfolio: Dictionary mapping tickers to share counts.
            cash_balance: Available cash for investments.
            strategy: Natural language trading strategy.
            max_buy_candidates: Maximum buy candidates to consider.

        Returns:
            Dictionary with recommendations and metadata.

        Raises:
            ValueError: If cash_balance is negative.
        """
        if cash_balance < 0:
            raise ValueError("cash_balance must be zero or positive")

        snapshots = [
            self._fetch_stock_snapshot(ticker, shares)
            for ticker, shares in portfolio.items()
        ]
        parsed_filters = self._parse_strategy_filters(strategy)
        buy_candidates = self._fetch_buy_candidates(parsed_filters, max_buy_candidates)
        prompt = self._build_prompt(
            portfolio, cash_balance, strategy, snapshots, buy_candidates
        )
        response = self.client.create_completion([
            {
                "role": "system",
                "content": (
                    "You are a financial assistant that produces trade recommendations "
                    "based on a natural language strategy."
                )
            },
            {"role": "user", "content": prompt},
        ])
        parsed = self._parse_response(response)
        parsed["snapshots"] = [snapshot.__dict__ for snapshot in snapshots]
        parsed["buy_candidates"] = buy_candidates
        parsed["cash_balance"] = cash_balance
        parsed["strategy"] = strategy
        parsed["portfolio"] = portfolio
        # Ensure buy recommendations equal max_buy_candidates
        parsed["buy"] = [
            candidate.get("Symbol", "Unknown") for candidate in buy_candidates
        ]
        return parsed
