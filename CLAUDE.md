# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI-powered hedge fund proof of concept for educational purposes. The system uses multiple AI agents (famous investors' personas like Warren Buffett, Charlie Munger, Cathie Wood, etc.) working together with specialized analyst agents to make trading decisions. The project has both a CLI interface and a web application (FastAPI backend + React frontend).

**Important**: This is for educational/research purposes only - not for real trading.

## Development Commands

### Environment Setup
```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Install dependencies
poetry install

# Set up environment variables
cp .env.example .env
# Edit .env to add API keys (OPENAI_API_KEY, FINANCIAL_DATASETS_API_KEY, etc.)
```

### Running the CLI

```bash
# Run the hedge fund for specific tickers
poetry run python src/main.py --ticker AAPL,MSFT,NVDA

# Run with date range
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --start-date 2024-01-01 --end-date 2024-03-01

# Run with local LLMs via Ollama
poetry run python src/main.py --ticker AAPL,MSFT,NVDA --ollama

# Run backtester
poetry run python src/backtester.py --ticker AAPL,MSFT,NVDA

# Alternative backtester command
poetry run backtester --ticker AAPL,MSFT,NVDA
```

### Running the Web Application

```bash
# Quick start (recommended)
./run.sh  # Mac/Linux
run.bat   # Windows

# Or manually:
cd app/backend
poetry run uvicorn main:app --reload  # Backend at http://localhost:8000

# In separate terminal:
cd app/frontend
npm install
npm run dev  # Frontend at http://localhost:5173
```

### Testing

```bash
# Run tests
poetry run pytest

# Run specific test file
poetry run pytest tests/test_api_rate_limiting.py
```

### Code Quality

```bash
# Format code (line length: 420)
poetry run black .

# Sort imports
poetry run isort .

# Lint code
poetry run flake8
```

## Architecture

### Agent System (LangGraph Multi-Agent)

The core of the system is a **LangGraph state machine** that orchestrates multiple AI agents. Agents communicate through a shared `AgentState` (defined in `src/graph/state.py`) that contains:
- `messages`: Conversation history
- `data`: Shared financial data (prices, metrics, news)
- `metadata`: Additional context

**Agent Workflow:**
1. **Analyst Agents** (investor personas) analyze the stock and provide recommendations
2. **Risk Manager** (`src/agents/risk_manager.py`) assesses portfolio risk and sets limits
3. **Portfolio Manager** (`src/agents/portfolio_manager.py`) makes final trading decisions

All analyst agents are configured in `src/utils/analysts.py` which defines:
- Agent order/execution sequence
- Display names and descriptions
- Agent functions

### Key Agent Types

**Investor Persona Agents** (in `src/agents/`):
- Famous investor personas: `warren_buffett.py`, `charlie_munger.py`, `cathie_wood.py`, `michael_burry.py`, `bill_ackman.py`, etc.
- Each has unique investment philosophy and analysis approach

**Analytical Agents**:
- `valuation.py`: Intrinsic value calculations and trading signals
- `fundamentals.py`: Fundamental data analysis
- `technicals.py`: Technical indicator analysis
- `sentiment.py` / `news_sentiment.py`: Market/news sentiment analysis
- `growth_agent.py`: Growth-focused analysis

**Decision-Making Agents**:
- `risk_manager.py`: Risk metrics and position limits
- `portfolio_manager.py`: Final trading decisions and order generation

### Data Layer

**Financial Data API** (`src/tools/api.py`):
- Fetches prices, financial metrics, news, insider trades
- Uses Financial Datasets API (free for AAPL, GOOGL, MSFT, NVDA, TSLA)
- Implements caching (`src/data/cache.py`) and rate limiting (429 handling with linear backoff)
- Data models defined in `src/data/models.py`

### Backtesting System

Located in `src/backtesting/`:
- `engine.py`: Core backtesting engine
- `portfolio.py`: Portfolio tracking and management
- `trader.py`: Trade execution simulation
- `metrics.py`: Performance metrics calculation (Sharpe ratio, max drawdown, etc.)
- `cli.py`: CLI interface for backtester

### LLM Configuration

Located in `src/llm/`:
- `models.py`: LLM provider configuration and initialization
- `api_models.json`: API-based model configurations (OpenAI, Anthropic, Groq, etc.)
- `ollama_models.json`: Local Ollama model configurations

Supports multiple providers: OpenAI, Anthropic, Groq, DeepSeek, Google Gemini, Ollama (local).

### Web Application

**Backend** (`app/backend/`):
- FastAPI application
- Main entry: `main.py`
- Database: SQLAlchemy with Alembic migrations
- CORS configured for localhost:5173 (frontend)
- API routes in `routes/`

**Frontend** (`app/frontend/`):
- React + Vite application
- Communicates with backend at localhost:8000

## Important Implementation Notes

### Adding New Agents

1. Create agent file in `src/agents/` (follow existing patterns)
2. Register in `src/utils/analysts.py` in `ANALYST_CONFIG`:
   - Set unique `order` for execution sequence
   - Define `agent_func`, `display_name`, `description`, `investing_style`
   - Set `type` as "analyst"
3. Agent function should accept `AgentState` and return updated state
4. Use shared state for data access and communication

### LLM Model Selection

- Default models defined in `src/llm/models.py`
- Use `--ollama` flag for local models
- API keys required for cloud providers (set in `.env`)
- Model configuration in JSON files (`api_models.json`, `ollama_models.json`)

### Date Handling

- Default behavior: analyzes recent period (e.g., last month)
- Use `--start-date` and `--end-date` for specific ranges
- Format: YYYY-MM-DD
- Both main.py and backtester.py support date flags

### Data Caching

- Financial data is cached to reduce API calls
- Cache implementation in `src/data/cache.py`
- Rate limiting handles 429 errors with linear backoff (60s, 90s, 120s...)

### Python Version

- Requires Python 3.11+
- Python 3.13+ may have compatibility issues
- Use pyenv or conda to manage Python versions if needed

## API Keys Required

Set in `.env` file:
- At least one LLM provider key: `OPENAI_API_KEY`, `GROQ_API_KEY`, `ANTHROPIC_API_KEY`, or `DEEPSEEK_API_KEY`
- `FINANCIAL_DATASETS_API_KEY` (only for tickers beyond AAPL, GOOGL, MSFT, NVDA, TSLA)

## Code Style

- Black formatter with **line length 420** (non-standard)
- isort for import sorting (black profile)
- Target Python version: 3.11
