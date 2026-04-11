# aiinvest

AI Investment Agent powered by `tvscreener` and OpenRouter.

## What it does

- Reads a portfolio of tickers and a cash balance (portfolio can be empty for pure cash deployment)
- Fetches live quote and performance data using `tvscreener`
- **Robustly parses natural-language trading strategies** to extract:
  - **Specific indices**: FTSE 100/250/350, S&P 500/100, NASDAQ 100, DAX 40, CAC 40, IBEX 35, NIKKEI 225
  - **Sector targeting**: recognizes sectors like technology, industrial, consumer, energy, healthcare, financials, utilities, materials, real estate, and communication services
  - **Market selection by sector**: selects a default market when strategy includes sector keywords without an explicit index
  - **Valuation filters**: P/E, P/B ratios, dividend yields, market caps, volumes
  - **Performance metrics**: ROE, ROA, yearly performance
  - **General markets**: UK, US, Germany, France, Spain, etc. (fallback when no specific index mentioned)
- Uses `tvscreener` to research and filter buy candidates based on the strategy
- Sends a strategy prompt with portfolio data and buy candidates to OpenRouter (default model: gpt-4o-mini)
- Returns sell/hold/buy guidance and supporting context

## Setup

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Create a `.env` file from the example:

```bash
cp env.example .env
```

3. Set your OpenRouter API key in `.env`:

```bash
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_MODEL=gpt-4o-mini  # optional, defaults to gpt-4o-mini
```

## Run the sample

```bash
python3 run_agent.py
```

## Testing

Run the test suite:

```bash
PYTHONPATH=src python3 -m pytest tests/ -v
```

## Using the Agent Programmatically

```python
from aiinvest.agent import AIInvestmentAgent

agent = AIInvestmentAgent()
recommendation = agent.recommend(
    portfolio={"AAPL": 10, "MSFT": 5},
    cash_balance=10000.0,
    strategy="Focus on S&P 500 with P/E below 20 and dividend yield above 2%",
    max_buy_candidates=10  # optional, default 10
)

print(recommendation["text"])  # Full LLM response
print(recommendation["sell"])  # List of sell recommendations
print(recommendation["hold"])  # List of hold recommendations
print(recommendation["buy"])   # List of buy recommendations
```

## Strategy Parsing

The agent automatically parses the trading strategy for:

- **Markets**: Recognizes major indices like FTSE 100/250, S&P 500, NASDAQ, DAX, CAC 40, IBEX 35, etc., and filters to the corresponding market
- **Valuation filters**:
  - "P/E below X" or "price-to-earnings below X"
  - "P/B below X" or "price-to-book below X"  
  - "dividend yield above X%"
- **Additional filters**:
  - "market cap above X[B|M|K]" (e.g., 10B for 10 billion)
  - "average volume above X[M|K]" (e.g., 1M for 1 million)
  - "1 year performance above X%"
  - "ROE above X%" (Return on Equity)
  - "ROA above X%" (Return on Assets)

Buy candidates are retrieved from `tvscreener`, sorted by market cap descending, and limited to 10 by default (configurable via `max_buy_candidates`).

## Output Structure

The `recommend()` method returns a dictionary with:

- `text`: Full natural language response from the LLM
- `sell`, `hold`: Parsed lists of recommendations from the LLM
- `buy`: List of all buy candidate symbols (length = `max_buy_candidates`)
- `snapshots`: Portfolio stock data
- `buy_candidates`: Detailed candidate stock data
- `cash_balance`, `strategy`, `portfolio`: Input parameters

## Files

- `src/aiinvest/agent.py`: Main agent implementation
- `run_agent.py`: Sample runner script
- `requirements.txt`: Python dependencies
- `env.example`: Environment variable template
- `tests/test_agent.py`: Comprehensive test suite
