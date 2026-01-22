# Implementation Plan: Strategic Reviewer Agent & Decision Book Frameworks

## Overview

Add a "Strategic Reviewer" agent that acts as a Devil's Advocate, reviewing analyst signals and applying decision-making frameworks from "The Decision Book" before signals reach the Risk Manager.

**Workflow Change:**
```
BEFORE: start → [analysts in parallel] → risk_manager → portfolio_manager → END
AFTER:  start → [analysts in parallel] → strategic_reviewer → risk_manager → portfolio_manager → END
```

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

### Framework Trigger Logic

| Scenario | Frameworks Triggered |
|----------|---------------------|
| 100% Bullish | Pre-Mortem, Rubber Band |
| 100% Bearish | SWOT, Rubber Band |
| High Conflict (split signals) | SWOT, Rumsfeld Matrix |
| High Confidence (>90%) | Pre-Mortem, Black Swan |

### Key Files to Create

1. **`src/frameworks/models.py`** - Pydantic models:
   - `FrameworkDefinition` - Base framework schema
   - `SWOTAnalysis` - strengths/weaknesses/opportunities/threats
   - `PreMortemAnalysis` - failure_scenarios, black_swan_candidates, risk_level
   - `RubberBandAnalysis` - deviation, reversion_probability
   - `RumsfeldMatrixAnalysis` - known_knowns, known_unknowns, unknown_unknowns

2. **`src/frameworks/loader.py`** - Functions:
   - `load_framework(framework_type)` - Load JSON definition
   - `get_triggered_frameworks(bullish_count, bearish_count, ...)` - Select applicable frameworks

---

## Phase 2: Strategic Reviewer Agent

### Create `/src/agents/strategic_reviewer.py`

**Core Design:**
- Does NOT fetch price data (reviews analyst conclusions only)
- Acts as Devil's Advocate (questions unanimous consensus)
- Applies Decision Book frameworks based on signal distribution
- Outputs metadata/warnings (not modified signals)

**Output Structure:**
```python
class StrategicReviewSignal(BaseModel):
    review_status: Literal["validated", "caution", "warning", "critical"]
    confidence_adjustment: int  # Range: -20 to +10
    frameworks_applied: list[str]
    key_concerns: list[str]
    contrarian_thesis: str
    reasoning: str
```

**Key Functions:**
1. `strategic_reviewer_agent(state, agent_id)` - Main agent function
2. `aggregate_signals(analyst_signals, ticker)` - Collect/summarize analyst signals
3. `generate_strategic_review(...)` - LLM call with framework prompts

**Agent Pattern (follows existing convention):**
```python
def strategic_reviewer_agent(state: AgentState, agent_id: str = "strategic_reviewer_agent"):
    # 1. Get analyst signals from state["data"]["analyst_signals"]
    # 2. Aggregate signals per ticker (count bullish/bearish/neutral)
    # 3. Determine triggered frameworks based on signal distribution
    # 4. Call LLM with framework-specific prompts
    # 5. Store review in state["data"]["analyst_signals"][agent_id]
    # 6. Return {"messages": [...], "data": state["data"]}
```

---

## Phase 3: Workflow Integration

### 3.1 CLI Integration (`src/main.py`)

Modify `create_workflow()`:

```python
def create_workflow(selected_analysts=None, include_strategic_review=True):
    # ... existing analyst node setup ...

    if include_strategic_review:
        from src.agents.strategic_reviewer import strategic_reviewer_agent
        workflow.add_node("strategic_reviewer_agent", strategic_reviewer_agent)

        # Connect: analysts → strategic_reviewer (instead of → risk_manager)
        for analyst_key in selected_analysts:
            node_name = analyst_nodes[analyst_key][0]
            workflow.add_edge(node_name, "strategic_reviewer_agent")

        workflow.add_edge("strategic_reviewer_agent", "risk_management_agent")
    else:
        # Original flow: analysts → risk_manager directly
        for analyst_key in selected_analysts:
            workflow.add_edge(analyst_nodes[analyst_key][0], "risk_management_agent")
```

Add CLI flag: `--no-strategic-review` to bypass

### 3.2 Agent Registration (`src/utils/analysts.py`)

Add to `ANALYST_CONFIG`:

```python
"strategic_reviewer": {
    "display_name": "Strategic Reviewer",
    "description": "Devil's Advocate",
    "investing_style": "Applies contrarian analysis and decision frameworks to question analyst consensus.",
    "agent_func": strategic_reviewer_agent,
    "type": "reviewer",  # New type (distinct from "analyst")
    "order": 99,  # Runs after all analysts
},
```

### 3.3 Web Backend (`app/backend/services/graph.py`)

Modify `create_graph()` to handle reviewer type:

```python
# Track reviewer nodes separately
reviewer_nodes = set()

for unique_agent_id in agent_ids:
    base_agent_key = extract_base_agent_key(unique_agent_id)
    if base_agent_key in ANALYST_CONFIG:
        if ANALYST_CONFIG[base_agent_key].get("type") == "reviewer":
            reviewer_nodes.add(unique_agent_id)
            # Add node but handle edge routing specially
```

Route: `analysts → reviewer_node → risk_manager` (similar to how portfolio_manager is handled)

---

## Phase 4: Output Visualization

### Modify `src/utils/display.py`

Add functions for framework visualization:

1. **`print_strategic_review_output(review_data, ticker)`**
   - Status table with color-coded review_status
   - Key concerns list
   - Contrarian thesis

2. **`print_swot_matrix(swot)`**
   - 2x2 grid with colored quadrants (GREEN=strengths, RED=weaknesses, BLUE=opportunities, YELLOW=threats)

3. **`print_pre_mortem_analysis(pre_mortem)`**
   - Risk level indicator
   - Failure scenarios with probability
   - Black swan candidates

4. **`print_rubber_band_analysis(rubber_band)`**
   - Deviation from mean
   - Reversion probability

---

## Critical Files Summary

| File | Action |
|------|--------|
| `src/frameworks/__init__.py` | Create |
| `src/frameworks/models.py` | Create - Pydantic models |
| `src/frameworks/loader.py` | Create - Framework loading |
| `src/frameworks/definitions/*.json` | Create - 5 framework definitions |
| `src/agents/strategic_reviewer.py` | Create - Main agent |
| `src/utils/analysts.py` | Modify - Add to ANALYST_CONFIG |
| `src/main.py` | Modify - Update create_workflow() |
| `app/backend/services/graph.py` | Modify - Handle reviewer type |
| `src/utils/display.py` | Modify - Add visualization functions |

---

## Verification Plan

### 1. Unit Tests
```bash
# Test framework loading
pytest tests/frameworks/test_loader.py

# Test signal aggregation
pytest tests/agents/test_strategic_reviewer.py
```

### 2. Integration Test
```bash
# Run with strategic review enabled (default)
poetry run python src/main.py --ticker AAPL,MSFT --show-reasoning

# Verify strategic_reviewer_agent appears in output between analysts and risk_manager
```

### 3. Workflow Test
```bash
# Test bypass flag
poetry run python src/main.py --ticker AAPL --no-strategic-review

# Verify strategic reviewer is skipped
```

### 4. Web Backend Test
- Start web app
- Create flow with Strategic Reviewer node between analysts and portfolio manager
- Run flow and verify strategic review output appears

---

## Implementation Order

1. **Framework Infrastructure** (Phase 1)
   - Create models.py with Pydantic schemas
   - Create JSON framework definitions
   - Create loader.py

2. **Strategic Reviewer Agent** (Phase 2)
   - Implement strategic_reviewer.py
   - Add to ANALYST_CONFIG

3. **CLI Workflow** (Phase 3a)
   - Modify create_workflow() in main.py
   - Add --no-strategic-review flag

4. **Web Backend** (Phase 3b)
   - Modify create_graph() in graph.py
   - Handle reviewer node type

5. **Output Visualization** (Phase 4)
   - Add display functions
   - Integrate with print_trading_output()
