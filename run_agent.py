"""Sample runner script for the AI Investment Agent."""

from src.aiinvest.agent import AIInvestmentAgent


EXAMPLE_PORTFOLIO = {}

# Alternative: Empty portfolio for pure cash deployment
# EXAMPLE_PORTFOLIO = {}

EXAMPLE_STRATEGY = (
    """
Value investing strategy focused on UK mid-cap stocks, specifically those in the FTSE 250 index.
The strategy aims to identify undervalued stocks with strong fundamentals, 
such as low price-to-earnings ratios, solid dividend yields, and reasonable price-to-book ratios. 
The agent should analyze the current portfolio, cash balance, 
and the specified strategy to provide actionable recommendations on which stocks to buy, hold, or sell. 
The recommendations should be based on the latest market data and the defined criteria for value investing within the UK mid-cap segment.
    """
)


def main() -> None:
    """Run the AI Investment Agent with sample data and print recommendations."""
    agent = AIInvestmentAgent()
    recommendation = agent.recommend(
        portfolio=EXAMPLE_PORTFOLIO,
        cash_balance=10000.0,
        strategy=EXAMPLE_STRATEGY,
    )

    print("=== AI Investment Recommendation ===")
    print(recommendation["text"])
    print("\n--- Parsed recommendations ---")
    for category in ("sell", "hold", "buy"):
        print(f"{category.upper()}: {recommendation[category]}")


if __name__ == "__main__":
    main()
