"""Unit tests for the AI Investment Agent."""

import pytest
from tvscreener import FilterOperator, Market, StockField

from src.aiinvest.agent import AIInvestmentAgent


class TestParseStrategyFilters:
    """Test suite for strategy filter parsing functionality."""

    def setup_method(self):
        """Set up test fixtures before each test method."""
        self.agent = AIInvestmentAgent()

    def test_market_parsing_ftse(self):
        """Test parsing of FTSE market references."""
        strategy = "Focus on FTSE 100 with low P/E"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]

    def test_market_parsing_sp500(self):
        """Test parsing of S&P 500 market references."""
        strategy = "Invest in S&P 500 stocks"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.AMERICA in parsed["markets"]

    def test_market_parsing_dax(self):
        """Test parsing of DAX market references."""
        strategy = "DAX index companies"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.GERMANY in parsed["markets"]

    def test_market_parsing_cac(self):
        """Test parsing of CAC market references."""
        strategy = "CAC 40 French stocks"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.FRANCE in parsed["markets"]

    def test_market_parsing_ibex(self):
        """Test parsing of IBEX market references."""
        strategy = "IBEX 35 Spain"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.SPAIN in parsed["markets"]

    def test_sector_parsing_technology(self):
        """Test parsing of technology sector references."""
        strategy = "technology stocks"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.AMERICA in parsed["markets"]
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.SECTOR
        assert op == FilterOperator.MATCH
        assert value == "Technology"

    def test_sector_parsing_industrial(self):
        """Test parsing of industrial sector references."""
        strategy = "industrial sector opportunities"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.SECTOR
        assert op == FilterOperator.MATCH
        assert value == "Industrial"

    def test_sector_with_explicit_market(self):
        """Test sector parsing combined with explicit market/index."""
        strategy = "FTSE 250 technology stocks"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        assert any(
            f[0] == StockField.SECTOR and
            f[1] == FilterOperator.MATCH and
            f[2] == "Technology"
            for f in parsed["filters"]
        )

    def test_index_parsing_ftse100(self):
        """Test parsing of FTSE 100 index references."""
        strategy = "FTSE 100 companies"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        assert len(parsed["filters"]) == 0  # FTSE 100 has no additional filters

    def test_index_parsing_ftse250(self):
        """Test parsing of FTSE 250 index references."""
        strategy = "FTSE 250 mid-caps"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        assert len(parsed["filters"]) == 2  # Market cap filters
        # Check the filters
        mc_filters = [
            f for f in parsed["filters"]
            if f[0] == StockField.MARKET_CAPITALIZATION
        ]
        assert len(mc_filters) == 2
        above_filter = next(f for f in mc_filters if f[1] == FilterOperator.ABOVE)
        below_filter = next(f for f in mc_filters if f[1] == FilterOperator.BELOW)
        assert above_filter[2] == 500_000_000
        assert below_filter[2] == 15_000_000_000

    def test_index_parsing_sp500(self):
        """Test parsing of S&P 500 index references."""
        strategy = "S&P 500 large caps"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.AMERICA in parsed["markets"]
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.MARKET_CAPITALIZATION
        assert op == FilterOperator.ABOVE
        assert value == 2_000_000_000

    def test_index_parsing_nasdaq100(self):
        """Test parsing of NASDAQ 100 index references."""
        strategy = "NASDAQ 100 tech stocks"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.AMERICA in parsed["markets"]
        assert len(parsed["filters"]) == 1  # NASDAQ 100 has sector filter
        field, op, value = parsed["filters"][0]
        assert field == StockField.SECTOR
        assert op == FilterOperator.MATCH
        assert value == "Technology"

    def test_index_parsing_dax40(self):
        """Test parsing of DAX 40 index references."""
        strategy = "DAX 40 German companies"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.GERMANY in parsed["markets"]
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.MARKET_CAPITALIZATION
        assert op == FilterOperator.ABOVE
        assert value == 5_000_000_000

    def test_pe_filter_below(self):
        """Test parsing of P/E ratio filters."""
        strategy = "P/E below 15"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.PRICE_TO_EARNINGS_RATIO_TTM
        assert op == FilterOperator.BELOW
        assert value == 15.0

    def test_pe_filter_price_to_earnings(self):
        """Test parsing of price-to-earnings filters."""
        strategy = "price-to-earnings below 20"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.PRICE_TO_EARNINGS_RATIO_TTM
        assert op == FilterOperator.BELOW
        assert value == 20.0

    def test_pb_filter_below(self):
        """Test parsing of P/B ratio filters."""
        strategy = "P/B below 1.5"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.PRICE_TO_BOOK_MRQ
        assert op == FilterOperator.BELOW
        assert value == 1.5

    def test_pb_filter_price_to_book(self):
        """Test parsing of price-to-book filters."""
        strategy = "price-to-book below 1.2"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.PRICE_TO_BOOK_MRQ
        assert op == FilterOperator.BELOW
        assert value == 1.2

    def test_dividend_yield_above(self):
        """Test parsing of dividend yield filters."""
        strategy = "dividend yield above 3%"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.DIVIDENDS_YIELD
        assert op == FilterOperator.ABOVE
        assert value == 3.0

    def test_market_cap_above_billion(self):
        """Test parsing of market cap filters in billions."""
        strategy = "market cap above 10B"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.MARKET_CAPITALIZATION
        assert op == FilterOperator.ABOVE
        assert value == 10_000_000_000.0

    def test_market_cap_above_million(self):
        """Test parsing of market cap filters in millions."""
        strategy = "market cap above 500M"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.MARKET_CAPITALIZATION
        assert op == FilterOperator.ABOVE
        assert value == 500_000_000.0

    def test_average_volume_above_million(self):
        """Test parsing of average volume filters."""
        strategy = "average volume above 1M"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.AVERAGE_VOLUME_30_DAY
        assert op == FilterOperator.ABOVE
        assert value == 1_000_000.0

    def test_yearly_performance_above(self):
        """Test parsing of yearly performance filters."""
        strategy = "1 year performance above 5%"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.YEARLY_PERFORMANCE
        assert op == FilterOperator.ABOVE
        assert value == 5.0

    def test_roe_above(self):
        """Test parsing of ROE (Return on Equity) filters."""
        strategy = "ROE above 10%"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.RETURN_ON_EQUITY_TTM
        assert op == FilterOperator.ABOVE
        assert value == 10.0

    def test_roa_above(self):
        """Test parsing of ROA (Return on Assets) filters."""
        strategy = "ROA above 8%"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.RETURN_ON_ASSETS_TTM
        assert op == FilterOperator.ABOVE
        assert value == 8.0

    def test_combined_filters(self):
        """Test parsing of multiple combined filters."""
        strategy = (
            "FTSE 100 with P/E below 15, P/B below 1.5, "
            "dividend yield above 3%, market cap above 1B"
        )
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        assert len(parsed["filters"]) == 4
        # Check each filter
        fields = [f[0] for f in parsed["filters"]]
        assert StockField.PRICE_TO_EARNINGS_RATIO_TTM in fields
        assert StockField.PRICE_TO_BOOK_MRQ in fields
        assert StockField.DIVIDENDS_YIELD in fields
        assert StockField.MARKET_CAPITALIZATION in fields

    def test_index_with_additional_filters(self):
        """Test parsing of index references with additional filters."""
        strategy = "FTSE 250 companies with P/E below 20"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        # Should have FTSE 250 filters (2) + P/E filter (1) = 3 total
        assert len(parsed["filters"]) == 3
        pe_filter = next(
            f for f in parsed["filters"]
            if f[0] == StockField.PRICE_TO_EARNINGS_RATIO_TTM
        )
        assert pe_filter[1] == FilterOperator.BELOW
        assert pe_filter[2] == 20.0

    def test_fallback_to_general_market(self):
        """Test fallback to general market when no specific index is mentioned."""
        strategy = "UK stocks with high ROE"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        # Should not have index-specific filters since no specific index mentioned
        mc_filters = [
            f for f in parsed["filters"]
            if f[0] == StockField.MARKET_CAPITALIZATION
        ]
        assert len(mc_filters) == 0

    def test_case_insensitive_index(self):
        """Test case insensitive parsing of index names."""
        strategy = "ftse 250 MID-CAPS"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        assert len(parsed["filters"]) == 2  # Market cap filters

    def test_index_with_typos(self):
        """Test parsing of index names with minor variations."""
        # Test that minor variations still work
        strategy = "FTSE250 companies"  # No space
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        assert len(parsed["filters"]) == 2

    def test_multiple_indices_priority(self):
        """Test priority handling when multiple indices are mentioned."""
        # Should take first match
        strategy = "FTSE 100 and FTSE 250 companies"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        assert len(parsed["filters"]) == 0  # FTSE 100 has no additional filters

    def test_index_with_units_in_strategy(self):
        """Test parsing of index with additional market cap filters."""
        strategy = "FTSE 250 with market cap above 1B"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        # Should have FTSE 250 filters (2) + market cap filter (1) = 3 total
        assert len(parsed["filters"]) == 3
        mc_filters = [
            f for f in parsed["filters"]
            if f[0] == StockField.MARKET_CAPITALIZATION
        ]
        assert len(mc_filters) == 3  # Two from FTSE 250, one from strategy

    def test_no_matches(self):
        """Test parsing of strategies with no recognizable filters."""
        strategy = "Some random strategy without filters"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert parsed["markets"] == []
        assert parsed["filters"] == []

    def test_case_insensitive(self):
        """Test case insensitive parsing of strategies."""
        strategy = "focus on ftse 100 with p/e below 20"
        parsed = self.agent._parse_strategy_filters(strategy)
        assert Market.UK in parsed["markets"]
        assert len(parsed["filters"]) == 1
        field, op, value = parsed["filters"][0]
        assert field == StockField.PRICE_TO_EARNINGS_RATIO_TTM
        assert value == 20.0

    def test_empty_portfolio_allowed(self):
        """Test that empty portfolio scenarios are handled gracefully."""
        # Should not raise error for empty portfolio
        try:
            parsed_filters = self.agent._parse_strategy_filters("FTSE 100 value stocks")
            buy_candidates = self.agent._fetch_buy_candidates(parsed_filters, 5)
            # Just check that it doesn't crash
            assert isinstance(buy_candidates, list)
        except Exception as e:
            pytest.fail(f"Empty portfolio handling failed: {e}")
