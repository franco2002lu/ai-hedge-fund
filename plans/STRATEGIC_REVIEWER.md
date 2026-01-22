# Implementation Plan: Strategic Reviewer Agent & Decision Book Frameworks

## Overview

Add a "Strategic Reviewer" agent that acts as a Devil's Advocate, reviewing analyst signals and applying decision-making frameworks from "The Decision Book" before signals reach the Risk Manager.

**Workflow Change:**
```
BEFORE: start → [analysts in parallel] → risk_manager → portfolio_manager → END
AFTER:  start → [analysts in parallel] → strategic_reviewer → risk_manager → portfolio_manager → END
```

---

## Implementation Status

| Phase | Status | Notes |
|-------|--------|-------|
| Phase 0: Prerequisites | ✅ Complete | 43 tests passing |
| Phase 1: Framework Infrastructure | ✅ Complete | All files created |
| Phase 2: Strategic Reviewer Agent | ✅ Complete | Core agent implemented |
| Phase 3: Workflow Integration | ✅ Complete | CLI and main.py updated |
| Phase 4: Portfolio Manager Enhancement | ✅ Complete | strategic_review_impact field added |
| Phase 5: Output Visualization | ✅ Complete | Display functions added |
| Phase 6: Web Backend | ⏳ Post-MVP | Not yet implemented |

### Files Created

| File | Status |
|------|--------|
| `src/frameworks/__init__.py` | ✅ Created |
| `src/frameworks/models.py` | ✅ Created |
| `src/frameworks/loader.py` | ✅ Created |
| `src/frameworks/definitions/pre_mortem.json` | ✅ Created |
| `src/frameworks/definitions/swot.json` | ✅ Created |
| `src/frameworks/definitions/rubber_band.json` | ✅ Created |
| `src/frameworks/definitions/rumsfeld_matrix.json` | ✅ Created |
| `src/frameworks/definitions/black_swan.json` | ✅ Created |
| `src/agents/strategic_reviewer.py` | ✅ Created |

### Files Modified

| File | Status |
|------|--------|
| `src/utils/analysts.py` | ✅ Modified - Added strategic_reviewer to ANALYST_CONFIG |
| `src/cli/input.py` | ✅ Modified - Added CLI flags and CLIInputs fields |
| `src/main.py` | ✅ Modified - Updated create_workflow() and run_hedge_fund() |
| `src/agents/portfolio_manager.py` | ✅ Modified - Added strategic_review_impact field |
| `src/utils/display.py` | ✅ Modified - Added visualization functions |

---

## Phase 0: Prerequisites - Verify Current System

Before implementing the Strategic Reviewer, verify the current system works correctly. This establishes a baseline and ensures our test setup can be reused for verification.

### 0.1 Environment Setup

```bash
# Ensure dependencies are installed
poetry install

# Verify environment variables are configured
cat .env | grep -E "(OPENAI_API_KEY|FINANCIAL_DATASETS_API_KEY)" | head -2
# Should show API keys are set (values redacted)
```

### 0.2 Run Existing Tests

```bash
# Run all existing tests to establish baseline
poetry run pytest tests/ -v

# Note any existing failures for comparison after implementation
```

### 0.3 Verify CLI Workflow (Minimal Run)

```bash
# Run with a single analyst and single ticker to verify basic flow
# Using free tickers (AAPL, GOOGL, MSFT, NVDA, TSLA) to avoid API key requirement
poetry run python src/main.py \
    --ticker AAPL \
    --analysts warren_buffett \
    --model gpt-4.1 \
    --show-reasoning

# Expected output:
# - Warren Buffett agent analyzes AAPL
# - Risk Manager calculates position limits
# - Portfolio Manager makes trading decision
# - Final output shows trading decision with reasoning
```

### 0.4 Verify Multi-Analyst Workflow

```bash
# Run with multiple analysts to verify parallel execution
poetry run python src/main.py \
    --ticker AAPL,MSFT \
    --analysts warren_buffett,charlie_munger,ben_graham \
    --model gpt-4.1 \
    --show-reasoning

# Expected output:
# - All 3 analysts run in parallel
# - Each analyst produces signals for both tickers
# - Risk Manager aggregates signals
# - Portfolio Manager makes final decisions
```

### 0.5 Verify Workflow Graph Generation

```bash
# Generate and inspect the workflow graph
poetry run python src/main.py \
    --ticker AAPL \
    --analysts warren_buffett,charlie_munger \
    --show-agent-graph

# Should produce a graph image showing:
# start_node → [warren_buffett_agent, charlie_munger_agent] → risk_management_agent → portfolio_manager → END
```

### 0.6 Document Baseline Behavior

Record the following for comparison after implementation:

| Metric | Before Implementation |
|--------|----------------------|
| Tests passing | `pytest tests/ -v` output |
| CLI run time (single analyst) | ~X seconds |
| CLI run time (3 analysts) | ~X seconds |
| Workflow nodes | start → analysts → risk → portfolio → END |
| Output format | Trading decisions JSON |

### 0.7 Success Criteria

- [x] All existing tests pass (43 tests passing)
- [x] CLI runs successfully with single analyst
- [x] CLI runs successfully with multiple analysts
- [x] Workflow graph generates correctly
- [x] Output includes analyst signals, risk metrics, and trading decisions

---

## Phase 1: Framework Infrastructure

### Create `/src/frameworks/` directory structure

```
src/frameworks/
├── __init__.py
├── models.py           # Pydantic models for framework inputs/outputs
├── loader.py           # Load and select frameworks based on triggers
└── definitions/
    ├── rubber_band.json
    ├── swot.json
    ├── bcg_matrix.json
    ├── rumsfeld_matrix.json
    └── pre_mortem.json
```

### 1.1 Framework Models (`src/frameworks/models.py`)

```python
from pydantic import BaseModel, Field
from typing import Literal

class FrameworkDefinition(BaseModel):
    """Base schema for framework JSON definitions."""
    name: str
    description: str
    prompt_template: str
    output_fields: list[str]

class SWOTAnalysis(BaseModel):
    """Structured SWOT output."""
    strengths: list[str] = Field(description="Internal advantages")
    weaknesses: list[str] = Field(description="Internal disadvantages")
    opportunities: list[str] = Field(description="External favorable factors")
    threats: list[str] = Field(description="External unfavorable factors")

class PreMortemAnalysis(BaseModel):
    """Pre-mortem failure analysis output."""
    failure_scenarios: list[dict] = Field(description="List of {scenario, probability, impact}")
    black_swan_candidates: list[str] = Field(description="Low-probability, high-impact events")
    risk_level: Literal["low", "medium", "high", "critical"]

class RubberBandAnalysis(BaseModel):
    """Mean-reversion analysis output."""
    deviation_score: int = Field(ge=1, le=10, description="How far from historical mean (1-10)")
    reversion_probability: int = Field(ge=0, le=100, description="Likelihood of mean reversion")
    stretched_direction: Literal["bullish", "bearish", "neutral"]
    historical_context: str

class RumsfeldMatrixAnalysis(BaseModel):
    """Known/unknown categorization output."""
    known_knowns: list[str] = Field(description="Facts we're confident about")
    known_unknowns: list[str] = Field(description="Questions we know we need answered")
    unknown_unknowns: list[str] = Field(description="Potential blind spots identified")
```

### 1.2 Framework Loader (`src/frameworks/loader.py`)

```python
import json
from pathlib import Path
from typing import Literal

FRAMEWORK_DIR = Path(__file__).parent / "definitions"

def load_framework(framework_type: str) -> dict:
    """Load a framework definition from JSON."""
    path = FRAMEWORK_DIR / f"{framework_type}.json"
    if not path.exists():
        raise FileNotFoundError(f"Framework definition not found: {framework_type}")
    with open(path) as f:
        return json.load(f)

def get_triggered_frameworks(
    consensus_type: str,
    avg_confidence: float,
    mode_config: dict,
) -> list[str]:
    """
    Select applicable frameworks based on signal distribution and review mode.

    Args:
        consensus_type: "unanimous_bullish", "unanimous_bearish", "mixed", "high_conflict"
        avg_confidence: Average confidence across analysts
        mode_config: Review mode configuration with thresholds

    Returns:
        List of framework names to apply (max 3)
    """
    frameworks = []

    # Consensus-based triggers
    if consensus_type == "unanimous_bullish":
        frameworks = ["pre_mortem", "rubber_band"]
    elif consensus_type == "unanimous_bearish":
        frameworks = ["swot", "rubber_band"]
    elif consensus_type == "high_conflict":
        frameworks = ["swot", "rumsfeld_matrix"]

    # High confidence adds black swan analysis
    if avg_confidence > mode_config["high_confidence_threshold"]:
        if "pre_mortem" not in frameworks:
            frameworks.append("pre_mortem")
        frameworks.append("black_swan")

    return frameworks[:3]  # Cap at 3 to avoid prompt bloat
```

### 1.3 Framework JSON Definitions

**`src/frameworks/definitions/pre_mortem.json`:**
```json
{
    "name": "Pre-Mortem Analysis",
    "description": "Imagine the investment has failed. What went wrong?",
    "prompt_template": "PRE-MORTEM ANALYSIS: Imagine this investment has failed spectacularly in 12 months. What went wrong? List 3-5 specific failure scenarios with estimated probability (low/medium/high) and potential impact.",
    "output_fields": ["failure_scenarios", "black_swan_candidates", "risk_level"]
}
```

**`src/frameworks/definitions/swot.json`:**
```json
{
    "name": "SWOT Analysis",
    "description": "Strengths, Weaknesses, Opportunities, Threats",
    "prompt_template": "SWOT ANALYSIS: Provide a structured analysis:\n- Strengths: What internal advantages does this investment have?\n- Weaknesses: What internal factors could hurt it?\n- Opportunities: What external factors could help?\n- Threats: What external factors could hurt it?",
    "output_fields": ["strengths", "weaknesses", "opportunities", "threats"]
}
```

**`src/frameworks/definitions/rubber_band.json`:**
```json
{
    "name": "Rubber Band Analysis",
    "description": "Mean-reversion risk assessment",
    "prompt_template": "RUBBER BAND ANALYSIS: How far has sentiment/valuation deviated from historical norms? Consider:\n- Deviation severity (1-10 scale)\n- Probability of mean reversion (0-100%)\n- Direction of stretch (bullish/bearish/neutral)\n- Historical context for comparison",
    "output_fields": ["deviation_score", "reversion_probability", "stretched_direction", "historical_context"]
}
```

**`src/frameworks/definitions/rumsfeld_matrix.json`:**
```json
{
    "name": "Rumsfeld Matrix",
    "description": "Categorize knowns and unknowns",
    "prompt_template": "RUMSFELD MATRIX: Categorize the information landscape:\n- Known Knowns: Facts we're confident about\n- Known Unknowns: Questions we know we need answered\n- Unknown Unknowns: What blind spots might we have? What are we not even thinking about?",
    "output_fields": ["known_knowns", "known_unknowns", "unknown_unknowns"]
}
```

**`src/frameworks/definitions/black_swan.json`:**
```json
{
    "name": "Black Swan Analysis",
    "description": "Identify tail-risk events",
    "prompt_template": "BLACK SWAN ANALYSIS: What low-probability, high-impact events could invalidate the thesis? List 2-3 tail-risk scenarios that analysts may have overlooked. Consider geopolitical, technological, regulatory, and market structure risks.",
    "output_fields": ["tail_risk_scenarios", "probability_estimate", "potential_impact"]
}
```

### 1.4 Framework Trigger Logic

| Scenario | Frameworks Triggered |
|----------|---------------------|
| Unanimous Bullish (≥ threshold) | Pre-Mortem, Rubber Band |
| Unanimous Bearish (≥ threshold) | SWOT, Rubber Band |
| High Conflict (≥ threshold split) | SWOT, Rumsfeld Matrix |
| High Avg Confidence (> threshold) | Pre-Mortem, Black Swan |

---

## Phase 2: Strategic Reviewer Agent

### Create `/src/agents/strategic_reviewer.py`

**Core Design Principles:**
- Does NOT fetch price data (reviews analyst conclusions only)
- Acts as Devil's Advocate (questions unanimous consensus)
- Applies Decision Book frameworks based on signal distribution
- Outputs metadata/warnings (not modified signals)
- Uses configurable review modes for different skepticism levels

### 2.1 Review Mode Configuration

```python
REVIEW_MODE_CONFIG = {
    "conservative": {
        "description": "Maximum skepticism, questions everything",
        "confidence_range": (-20, 5),
        "unanimity_threshold": 0.85,   # 85% agreement triggers unanimity frameworks
        "conflict_threshold": 0.30,    # 30% disagreement triggers conflict frameworks
        "high_confidence_threshold": 75,
    },
    "balanced": {
        "description": "Standard devil's advocate (default)",
        "confidence_range": (-15, 10),
        "unanimity_threshold": 0.95,   # 95% agreement
        "conflict_threshold": 0.40,    # 40% disagreement
        "high_confidence_threshold": 85,
    },
    "aggressive": {
        "description": "Light touch, only flags major red flags",
        "confidence_range": (-10, 10),
        "unanimity_threshold": 1.0,    # 100% agreement only
        "conflict_threshold": 0.50,    # 50% disagreement
        "high_confidence_threshold": 90,
    },
}
```

### 2.2 Output Structure

```python
class StrategicReviewSignal(BaseModel):
    signal: Literal["validated", "caution", "warning", "critical"]
    confidence: int  # 0-100 (review confidence)
    reasoning: str
    # Extended fields:
    confidence_adjustment: int = Field(ge=-20, le=10, description="Suggestion for portfolio manager")
    consensus_type: str  # "unanimous_bullish", "unanimous_bearish", "mixed", "high_conflict"
    frameworks_applied: list[str]
    key_concerns: list[str]
    contrarian_thesis: str  # "What if we're wrong?"
    # Optional framework-specific outputs:
    swot_analysis: SWOTAnalysis | None = None
    pre_mortem_analysis: PreMortemAnalysis | None = None
    rubber_band_analysis: RubberBandAnalysis | None = None
    rumsfeld_analysis: RumsfeldMatrixAnalysis | None = None
```

### 2.3 Agent Implementation

```python
def strategic_reviewer_agent(state: AgentState, agent_id: str = "strategic_reviewer_agent"):
    """
    Reviews analyst signals and applies decision-making frameworks.
    Does NOT fetch price data - reviews analyst conclusions only.
    Acts as Devil's Advocate, questioning consensus.
    """
    data = state["data"]
    tickers = data["tickers"]
    analyst_signals = data["analyst_signals"]
    review_mode = state["metadata"].get("review_mode", "balanced")
    mode_config = REVIEW_MODE_CONFIG[review_mode]

    strategic_review = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Aggregating analyst signals")

        # 1. Aggregate signals
        signal_summary = aggregate_signals(analyst_signals, ticker)

        # 2. Determine consensus type
        consensus_type = determine_consensus_type(signal_summary, mode_config)

        # 3. Select triggered frameworks
        frameworks = get_triggered_frameworks(consensus_type, signal_summary["avg_confidence"], mode_config)

        progress.update_status(agent_id, ticker, f"Applying {len(frameworks)} frameworks")

        # 4. Generate strategic review via LLM
        review = generate_strategic_review(
            ticker=ticker,
            signal_summary=signal_summary,
            consensus_type=consensus_type,
            frameworks=frameworks,
            review_mode=review_mode,
            mode_config=mode_config,
            state=state,
            agent_id=agent_id,
        )

        strategic_review[ticker] = {
            "signal": review.signal,
            "confidence": review.confidence,
            "reasoning": review.reasoning,
            "confidence_adjustment": review.confidence_adjustment,
            "consensus_type": consensus_type,
            "frameworks_applied": frameworks,
            "key_concerns": review.key_concerns,
            "contrarian_thesis": review.contrarian_thesis,
            # Include framework-specific outputs if generated
            "swot_analysis": review.swot_analysis.model_dump() if review.swot_analysis else None,
            "pre_mortem_analysis": review.pre_mortem_analysis.model_dump() if review.pre_mortem_analysis else None,
            "rubber_band_analysis": review.rubber_band_analysis.model_dump() if review.rubber_band_analysis else None,
            "rumsfeld_analysis": review.rumsfeld_analysis.model_dump() if review.rumsfeld_analysis else None,
        }

        progress.update_status(agent_id, ticker, "Done", analysis=review.reasoning)

    message = HumanMessage(content=json.dumps(strategic_review), name=agent_id)

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(strategic_review, "Strategic Reviewer")

    state["data"]["analyst_signals"][agent_id] = strategic_review
    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}
```

### 2.4 Helper Functions

```python
def aggregate_signals(analyst_signals: dict, ticker: str) -> dict:
    """Aggregate all analyst signals for a ticker."""
    bullish = bearish = neutral = 0
    confidences = []
    signals_detail = {}

    for agent_id, signals in analyst_signals.items():
        # Skip non-analyst agents
        if agent_id.startswith("risk_management") or agent_id.startswith("strategic_reviewer"):
            continue
        if ticker not in signals:
            continue

        sig = signals[ticker].get("signal")
        conf = signals[ticker].get("confidence", 50)

        if sig == "bullish":
            bullish += 1
        elif sig == "bearish":
            bearish += 1
        else:
            neutral += 1

        confidences.append(conf)
        signals_detail[agent_id] = {"signal": sig, "confidence": conf}

    total = bullish + bearish + neutral
    return {
        "bullish_count": bullish,
        "bearish_count": bearish,
        "neutral_count": neutral,
        "total": total,
        "avg_confidence": sum(confidences) / len(confidences) if confidences else 50,
        "bullish_pct": bullish / total if total > 0 else 0,
        "bearish_pct": bearish / total if total > 0 else 0,
        "signals_detail": signals_detail,
    }


def determine_consensus_type(summary: dict, config: dict) -> str:
    """Determine the consensus type based on signal distribution."""
    unanimity_threshold = config["unanimity_threshold"]
    conflict_threshold = config["conflict_threshold"]

    if summary["bullish_pct"] >= unanimity_threshold:
        return "unanimous_bullish"
    elif summary["bearish_pct"] >= unanimity_threshold:
        return "unanimous_bearish"
    elif min(summary["bullish_pct"], summary["bearish_pct"]) >= conflict_threshold:
        return "high_conflict"
    else:
        return "mixed"


def generate_strategic_review(
    ticker: str,
    signal_summary: dict,
    consensus_type: str,
    frameworks: list[str],
    review_mode: str,
    mode_config: dict,
    state: AgentState,
    agent_id: str,
) -> StrategicReviewSignal:
    """Generate the strategic review using LLM with framework prompts."""

    # Load framework prompts
    framework_prompts = []
    for fw in frameworks:
        try:
            fw_def = load_framework(fw)
            framework_prompts.append(fw_def["prompt_template"])
        except FileNotFoundError:
            continue

    # Build the prompt
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a Strategic Reviewer acting as Devil's Advocate.
Review mode: {review_mode} ({mode_config['description']})

Your role is to question consensus and identify risks that analysts may have missed.
Apply the requested frameworks to stress-test the investment thesis.

Confidence adjustment range: {mode_config['confidence_range'][0]} to {mode_config['confidence_range'][1]}
- Negative values suggest reducing conviction
- Positive values suggest the thesis is robust

Output a review_status:
- "validated": Thesis holds up well under scrutiny
- "caution": Minor concerns identified
- "warning": Significant risks that warrant attention
- "critical": Major red flags that could invalidate the thesis"""
        ),
        (
            "human",
            """Ticker: {ticker}

ANALYST SIGNAL SUMMARY:
- Bullish: {bullish_count} ({bullish_pct:.0%})
- Bearish: {bearish_count} ({bearish_pct:.0%})
- Neutral: {neutral_count}
- Average Confidence: {avg_confidence:.1f}%
- Consensus Type: {consensus_type}

ANALYST DETAILS:
{signals_detail}

FRAMEWORKS TO APPLY:
{framework_prompts}

Provide your strategic review with:
1. Overall review status and reasoning
2. Confidence adjustment recommendation
3. Key concerns (list of specific issues)
4. Contrarian thesis (what if we're wrong?)
5. Framework-specific analysis for each applied framework"""
        ),
    ])

    prompt = template.invoke({
        "ticker": ticker,
        "bullish_count": signal_summary["bullish_count"],
        "bearish_count": signal_summary["bearish_count"],
        "neutral_count": signal_summary["neutral_count"],
        "bullish_pct": signal_summary["bullish_pct"],
        "bearish_pct": signal_summary["bearish_pct"],
        "avg_confidence": signal_summary["avg_confidence"],
        "consensus_type": consensus_type,
        "signals_detail": json.dumps(signal_summary["signals_detail"], indent=2),
        "framework_prompts": "\n\n".join(framework_prompts) if framework_prompts else "No specific frameworks triggered.",
    })

    def _default():
        return StrategicReviewSignal(
            signal="caution",
            confidence=50,
            reasoning="Unable to complete strategic review",
            confidence_adjustment=0,
            consensus_type=consensus_type,
            frameworks_applied=frameworks,
            key_concerns=["Review could not be completed"],
            contrarian_thesis="Insufficient data for contrarian analysis",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=StrategicReviewSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )
```

---

## Phase 3: Workflow Integration

### 3.1 CLI Integration (`src/main.py`)

**Modify `create_workflow()`:**

```python
def create_workflow(selected_analysts=None, include_strategic_review=True):
    """Create the workflow with selected analysts."""
    workflow = StateGraph(AgentState)
    workflow.add_node("start_node", start)

    analyst_nodes = get_analyst_nodes()

    if selected_analysts is None:
        selected_analysts = list(analyst_nodes.keys())

    # Add analyst nodes (exclude reviewer type from parallel phase)
    for analyst_key in selected_analysts:
        if ANALYST_CONFIG.get(analyst_key, {}).get("type") == "reviewer":
            continue
        node_name, node_func = analyst_nodes[analyst_key]
        workflow.add_node(node_name, node_func)
        workflow.add_edge("start_node", node_name)

    # Always add risk and portfolio management
    workflow.add_node("risk_management_agent", risk_management_agent)
    workflow.add_node("portfolio_manager", portfolio_management_agent)

    if include_strategic_review:
        from src.agents.strategic_reviewer import strategic_reviewer_agent
        workflow.add_node("strategic_reviewer_agent", strategic_reviewer_agent)

        # Connect: analysts → strategic_reviewer (instead of → risk_manager)
        for analyst_key in selected_analysts:
            if ANALYST_CONFIG.get(analyst_key, {}).get("type") != "reviewer":
                node_name = analyst_nodes[analyst_key][0]
                workflow.add_edge(node_name, "strategic_reviewer_agent")

        workflow.add_edge("strategic_reviewer_agent", "risk_management_agent")
    else:
        # Original flow: analysts → risk_manager directly
        for analyst_key in selected_analysts:
            if ANALYST_CONFIG.get(analyst_key, {}).get("type") != "reviewer":
                workflow.add_edge(analyst_nodes[analyst_key][0], "risk_management_agent")

    workflow.add_edge("risk_management_agent", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    workflow.set_entry_point("start_node")
    return workflow
```

**Modify `run_hedge_fund()`:**

```python
def run_hedge_fund(
    tickers: list[str],
    start_date: str,
    end_date: str,
    portfolio: dict,
    show_reasoning: bool = False,
    selected_analysts: list[str] = [],
    model_name: str = "gpt-4.1",
    model_provider: str = "OpenAI",
    include_strategic_review: bool = True,  # NEW
    review_mode: str = "balanced",           # NEW
):
    progress.start()
    try:
        workflow = create_workflow(
            selected_analysts if selected_analysts else None,
            include_strategic_review=include_strategic_review,
        )
        agent = workflow.compile()

        final_state = agent.invoke({
            "messages": [HumanMessage(content="Make trading decisions based on the provided data.")],
            "data": {
                "tickers": tickers,
                "portfolio": portfolio,
                "start_date": start_date,
                "end_date": end_date,
                "analyst_signals": {},
            },
            "metadata": {
                "show_reasoning": show_reasoning,
                "model_name": model_name,
                "model_provider": model_provider,
                "review_mode": review_mode,  # NEW
            },
        })
        # ... rest unchanged ...
```

**Add CLI flags in `src/cli/input.py`:**

```python
parser.add_argument(
    "--no-strategic-review",
    action="store_true",
    help="Bypass the strategic reviewer agent"
)
parser.add_argument(
    "--review-mode",
    type=str,
    choices=["conservative", "balanced", "aggressive"],
    default="balanced",
    help="Strategic review intensity: conservative (max skepticism), balanced (default), aggressive (light touch)"
)
```

**Update `CLIInputs` dataclass:**

```python
@dataclass
class CLIInputs:
    # ... existing fields ...
    include_strategic_review: bool = True
    review_mode: str = "balanced"
```

### 3.2 Agent Registration (`src/utils/analysts.py`)

Add to `ANALYST_CONFIG`:

```python
from src.agents.strategic_reviewer import strategic_reviewer_agent

ANALYST_CONFIG = {
    # ... existing analysts ...

    "strategic_reviewer": {
        "display_name": "Strategic Reviewer",
        "description": "Devil's Advocate",
        "investing_style": "Applies contrarian analysis and decision frameworks to question analyst consensus.",
        "agent_func": strategic_reviewer_agent,
        "type": "reviewer",  # New type (distinct from "analyst")
        "order": 99,  # Runs after all analysts
    },
}
```

### 3.3 Web Backend (`app/backend/services/graph.py`)

Modify `create_graph()` to handle reviewer type:

```python
def create_graph(graph_nodes: list, graph_edges: list) -> StateGraph:
    # ... existing setup ...

    # Track reviewer nodes separately
    reviewer_nodes = set()

    for unique_agent_id in agent_ids:
        base_agent_key = extract_base_agent_key(unique_agent_id)
        if base_agent_key in ANALYST_CONFIG:
            agent_type = ANALYST_CONFIG[base_agent_key].get("type", "analyst")

            if agent_type == "reviewer":
                reviewer_nodes.add(unique_agent_id)
                reviewer_function = create_agent_function(
                    ANALYST_CONFIG[base_agent_key]["agent_func"],
                    unique_agent_id
                )
                graph.add_node(unique_agent_id, reviewer_function)
                continue

            # ... existing analyst handling ...

    # Route: analysts → reviewer_node → risk_manager
    # (Similar pattern to how portfolio_manager is handled)
```

---

## Phase 4: Portfolio Manager Enhancement

### Modify `src/agents/portfolio_manager.py`

**Update `PortfolioDecision` model:**

```python
class PortfolioDecision(BaseModel):
    action: Literal["buy", "sell", "short", "cover", "hold"]
    quantity: int = Field(description="Number of shares to trade")
    confidence: int = Field(description="Confidence 0-100")
    reasoning: str = Field(description="Reasoning for the decision")
    strategic_review_impact: str = Field(
        default="",
        description="How the strategic review influenced this decision (required if review data present)"
    )
```

**Update `generate_trading_decision()`:**

```python
def generate_trading_decision(..., analyst_signals: dict, ...):
    # ... existing code ...

    # Extract strategic review data
    strategic_review = {}
    for agent_id, signals in analyst_signals.items():
        if agent_id.startswith("strategic_reviewer"):
            strategic_review = signals
            break

    # Build strategic review context for prompt
    review_context = ""
    if strategic_review:
        review_context = json.dumps({
            ticker: {
                "status": data.get("signal"),
                "confidence_adjustment": data.get("confidence_adjustment", 0),
                "key_concerns": data.get("key_concerns", []),
                "contrarian_thesis": data.get("contrarian_thesis", ""),
            }
            for ticker, data in strategic_review.items()
            if ticker in tickers_for_llm
        }, separators=(",", ":"))

    # Update prompt template
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            "You are a portfolio manager.\n"
            "Inputs per ticker: analyst signals, allowed actions with max qty, and strategic review (if provided).\n"
            "Pick one allowed action per ticker and a quantity <= the max.\n"
            "Keep reasoning concise (max 100 chars).\n"
            "IMPORTANT: If strategic review data is provided, you MUST explain how it influenced your decision "
            "in the 'strategic_review_impact' field. Consider the concerns and confidence adjustments.\n"
            "Return JSON only."
        ),
        (
            "human",
            "Signals:\n{signals}\n\n"
            "Allowed:\n{allowed}\n\n"
            + ("Strategic Review:\n{review}\n\n" if review_context else "")
            + "Format:\n{{\n"
            '  "decisions": {{\n'
            '    "TICKER": {{"action":"...","quantity":int,"confidence":int,"reasoning":"...","strategic_review_impact":"..."}}\n'
            "  }}\n"
            "}}"
        ),
    ])

    prompt_data = {
        "signals": json.dumps(compact_signals, separators=(",", ":")),
        "allowed": json.dumps(compact_allowed, separators=(",", ":")),
    }
    if review_context:
        prompt_data["review"] = review_context

    # ... rest of function ...
```

---

## Phase 5: Output Visualization

### Modify `src/utils/display.py`

Add functions for framework visualization:

```python
from colorama import Fore, Style

def print_strategic_review_output(review_data: dict, ticker: str):
    """Print strategic review with color-coded status."""
    status_colors = {
        "validated": Fore.GREEN,
        "caution": Fore.YELLOW,
        "warning": Fore.RED,
        "critical": Fore.MAGENTA + Style.BRIGHT,
    }

    review = review_data.get(ticker, {})
    status = review.get("signal", "unknown")
    color = status_colors.get(status, Fore.WHITE)

    print(f"\n{'='*60}")
    print(f"STRATEGIC REVIEW: {ticker}")
    print(f"{'='*60}")
    print(f"Status: {color}{status.upper()}{Style.RESET_ALL}")
    print(f"Confidence Adjustment: {review.get('confidence_adjustment', 0):+d}")
    print(f"Consensus Type: {review.get('consensus_type', 'unknown')}")
    print(f"Frameworks Applied: {', '.join(review.get('frameworks_applied', []))}")

    print(f"\n{Fore.CYAN}Key Concerns:{Style.RESET_ALL}")
    for concern in review.get("key_concerns", []):
        print(f"  • {concern}")

    print(f"\n{Fore.YELLOW}Contrarian Thesis:{Style.RESET_ALL}")
    print(f"  {review.get('contrarian_thesis', 'N/A')}")

    print(f"\n{Fore.WHITE}Reasoning:{Style.RESET_ALL}")
    print(f"  {review.get('reasoning', 'N/A')}")


def print_swot_matrix(swot: dict):
    """Print SWOT analysis as a 2x2 grid with colored quadrants."""
    if not swot:
        return

    print(f"\n{Fore.CYAN}SWOT ANALYSIS{Style.RESET_ALL}")
    print("┌" + "─"*30 + "┬" + "─"*30 + "┐")

    # Strengths (Green) | Weaknesses (Red)
    print(f"│ {Fore.GREEN}STRENGTHS{Style.RESET_ALL}" + " "*20 + f"│ {Fore.RED}WEAKNESSES{Style.RESET_ALL}" + " "*19 + "│")
    max_len = max(len(swot.get("strengths", [])), len(swot.get("weaknesses", [])), 1)
    for i in range(max_len):
        s = swot.get("strengths", [])[i] if i < len(swot.get("strengths", [])) else ""
        w = swot.get("weaknesses", [])[i] if i < len(swot.get("weaknesses", [])) else ""
        print(f"│ • {s[:26]:<26} │ • {w[:26]:<26} │")

    print("├" + "─"*30 + "┼" + "─"*30 + "┤")

    # Opportunities (Blue) | Threats (Yellow)
    print(f"│ {Fore.BLUE}OPPORTUNITIES{Style.RESET_ALL}" + " "*16 + f"│ {Fore.YELLOW}THREATS{Style.RESET_ALL}" + " "*22 + "│")
    max_len = max(len(swot.get("opportunities", [])), len(swot.get("threats", [])), 1)
    for i in range(max_len):
        o = swot.get("opportunities", [])[i] if i < len(swot.get("opportunities", [])) else ""
        t = swot.get("threats", [])[i] if i < len(swot.get("threats", [])) else ""
        print(f"│ • {o[:26]:<26} │ • {t[:26]:<26} │")

    print("└" + "─"*30 + "┴" + "─"*30 + "┘")


def print_pre_mortem_analysis(pre_mortem: dict):
    """Print pre-mortem analysis with risk indicators."""
    if not pre_mortem:
        return

    risk_colors = {
        "low": Fore.GREEN,
        "medium": Fore.YELLOW,
        "high": Fore.RED,
        "critical": Fore.MAGENTA + Style.BRIGHT,
    }

    risk_level = pre_mortem.get("risk_level", "unknown")
    color = risk_colors.get(risk_level, Fore.WHITE)

    print(f"\n{Fore.CYAN}PRE-MORTEM ANALYSIS{Style.RESET_ALL}")
    print(f"Risk Level: {color}{risk_level.upper()}{Style.RESET_ALL}")

    print(f"\n{Fore.RED}Failure Scenarios:{Style.RESET_ALL}")
    for scenario in pre_mortem.get("failure_scenarios", []):
        if isinstance(scenario, dict):
            print(f"  [{scenario.get('probability', '?')}] {scenario.get('scenario', 'Unknown')}")
        else:
            print(f"  • {scenario}")

    print(f"\n{Fore.MAGENTA}Black Swan Candidates:{Style.RESET_ALL}")
    for swan in pre_mortem.get("black_swan_candidates", []):
        print(f"  ⚠ {swan}")


def print_rubber_band_analysis(rubber_band: dict):
    """Print rubber band analysis with deviation meter."""
    if not rubber_band:
        return

    print(f"\n{Fore.CYAN}RUBBER BAND ANALYSIS{Style.RESET_ALL}")

    deviation = rubber_band.get("deviation_score", 5)
    direction = rubber_band.get("stretched_direction", "neutral")
    reversion_prob = rubber_band.get("reversion_probability", 50)

    # Visual deviation meter
    meter = "["
    for i in range(1, 11):
        if i <= deviation:
            meter += "█"
        else:
            meter += "░"
    meter += "]"

    direction_colors = {
        "bullish": Fore.GREEN,
        "bearish": Fore.RED,
        "neutral": Fore.YELLOW,
    }
    color = direction_colors.get(direction, Fore.WHITE)

    print(f"Deviation: {meter} {deviation}/10")
    print(f"Direction: {color}{direction.upper()}{Style.RESET_ALL}")
    print(f"Reversion Probability: {reversion_prob}%")
    print(f"Context: {rubber_band.get('historical_context', 'N/A')}")
```

---

## Critical Files Summary

| File | Action | Priority |
|------|--------|----------|
| `src/frameworks/__init__.py` | Create | P0 - Core |
| `src/frameworks/models.py` | Create - Pydantic models | P0 - Core |
| `src/frameworks/loader.py` | Create - Framework loading | P0 - Core |
| `src/frameworks/definitions/*.json` | Create - 5 framework definitions | P0 - Core |
| `src/agents/strategic_reviewer.py` | Create - Main agent | P0 - Core |
| `src/utils/analysts.py` | Modify - Add to ANALYST_CONFIG | P0 - Core |
| `src/main.py` | Modify - Update create_workflow() | P0 - Core |
| `src/cli/input.py` | Modify - Add CLI flags | P0 - Core |
| `src/agents/portfolio_manager.py` | Modify - Add review impact | P1 - Enhancement |
| `src/utils/display.py` | Modify - Add visualization functions | P1 - Enhancement |
| `app/backend/services/graph.py` | Modify - Handle reviewer type | P2 - Post-MVP |

---

## Verification Plan

### 1. Unit Tests

```bash
# Test framework loading
pytest tests/frameworks/test_loader.py -v

# Test signal aggregation
pytest tests/agents/test_strategic_reviewer.py::test_aggregate_signals -v

# Test consensus determination
pytest tests/agents/test_strategic_reviewer.py::test_determine_consensus -v

# Test framework selection
pytest tests/agents/test_strategic_reviewer.py::test_framework_triggers -v
```

### 2. Integration Tests - CLI

```bash
# Default run (strategic review enabled, balanced mode)
poetry run python src/main.py --ticker AAPL,MSFT --analysts-all --show-reasoning

# Verify output shows:
# - Strategic Reviewer section between analysts and risk manager
# - Key concerns, contrarian thesis visible
# - SWOT/Pre-Mortem/Rubber Band visualizations
# - Portfolio manager includes "strategic_review_impact" in reasoning

# Test bypass flag
poetry run python src/main.py --ticker AAPL --analysts-all --no-strategic-review
# Verify: No strategic reviewer output

# Test review modes
poetry run python src/main.py --ticker AAPL --analysts-all --review-mode conservative
poetry run python src/main.py --ticker AAPL --analysts-all --review-mode aggressive
```

### 3. Workflow Visualization

```bash
# Generate workflow graph to verify edge routing
poetry run python src/main.py --ticker AAPL --show-agent-graph
# Verify: Graph shows analysts → strategic_reviewer → risk_manager → portfolio_manager
```

### 4. Web Backend Test

- Start web app
- Create flow with Strategic Reviewer node between analysts and portfolio manager
- Run flow and verify strategic review output appears

---

## Implementation Order

1. **Phase 1: Framework Infrastructure**
   - Create `src/frameworks/` directory structure
   - Create `models.py` with Pydantic schemas
   - Create JSON framework definitions
   - Create `loader.py`

2. **Phase 2: Strategic Reviewer Agent**
   - Implement `strategic_reviewer.py` with review modes
   - Add to `ANALYST_CONFIG`

3. **Phase 3: CLI Workflow**
   - Modify `create_workflow()` in `main.py`
   - Add `--no-strategic-review` and `--review-mode` flags
   - Update `run_hedge_fund()` with new parameters

4. **Phase 4: Portfolio Manager Enhancement**
   - Update `PortfolioDecision` model
   - Modify `generate_trading_decision()` to include review context
   - Require `strategic_review_impact` explanation

5. **Phase 5: Output Visualization**
   - Add display functions for framework outputs
   - Integrate with `print_trading_output()`

6. **Phase 6: Web Backend** (Post-MVP)
   - Modify `create_graph()` in `graph.py`
   - Handle reviewer node type

---

## Design Rationale

### Why separate framework infrastructure?
- Enables adding new frameworks without modifying agent code
- JSON definitions are easy to version and review
- Pydantic models ensure type safety for framework outputs
- Loader can be extended for dynamic framework selection

### Why metadata-only output (not signal modification)?
- Preserves original analyst signals for transparency
- Allows portfolio manager to decide how much weight to give the review
- Easier to debug and audit decision flow

### Why review mode presets?
- Different users have different risk tolerances
- Avoids overwhelming users with framework configuration
- Provides sensible defaults while allowing customization

### Why require portfolio manager to explain review impact?
- Ensures the review is actually considered, not ignored
- Provides accountability and transparency
- Helps users understand how contrarian analysis affected decisions

---

## Usage Guide

### Basic Usage (Strategic Review Enabled by Default)

```bash
# Run with all analysts (strategic review enabled by default)
poetry run python src/main.py --tickers AAPL --analysts-all --model gpt-4.1 --show-reasoning

# Run with specific analysts
poetry run python src/main.py --tickers AAPL,MSFT --analysts warren_buffett,charlie_munger --model gpt-4.1
```

### Review Mode Options

```bash
# Conservative mode - maximum skepticism, questions everything
poetry run python src/main.py --tickers AAPL --analysts-all --review-mode conservative

# Balanced mode (default) - standard devil's advocate
poetry run python src/main.py --tickers AAPL --analysts-all --review-mode balanced

# Aggressive mode - light touch, only flags major red flags
poetry run python src/main.py --tickers AAPL --analysts-all --review-mode aggressive
```

### Bypass Strategic Review

```bash
# Skip strategic review entirely (original workflow)
poetry run python src/main.py --tickers AAPL --analysts-all --no-strategic-review
```

### Expected Output

When `--show-reasoning` is enabled, the output includes:

1. **Analyst Signals** - Individual analyst recommendations (bullish/bearish/neutral)
2. **Strategic Review** - Devil's advocate analysis including:
   - Review status (validated/caution/warning/critical)
   - Confidence adjustment recommendation
   - Key concerns identified
   - Contrarian thesis ("What if we're wrong?")
   - Framework-specific analysis (SWOT, Pre-Mortem, etc.)
3. **Risk Manager** - Position limits and risk metrics
4. **Portfolio Manager** - Final trading decisions with `strategic_review_impact` field explaining how the review influenced the decision

### Review Mode Comparison

| Mode | Unanimity Threshold | Conflict Threshold | Confidence Threshold | Use Case |
|------|--------------------|--------------------|---------------------|----------|
| Conservative | 85% | 30% | 75% | Risk-averse, thorough analysis |
| Balanced | 95% | 40% | 85% | Default, standard scrutiny |
| Aggressive | 100% | 50% | 90% | Fast decisions, minimal friction |

---

## Remaining Work

### Phase 6: Web Backend (Post-MVP)

The web backend (`app/backend/services/graph.py`) has not been modified yet. To enable strategic reviewer in the web UI:

1. Modify `create_graph()` to detect `type: "reviewer"` in ANALYST_CONFIG
2. Route edges: analysts → reviewer → risk_manager
3. Add UI controls for `--review-mode` selection

This is not required for CLI usage but needed for full web app integration.