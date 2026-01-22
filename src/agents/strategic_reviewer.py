"""
Strategic Reviewer Agent

Acts as a Devil's Advocate, reviewing analyst signals and applying
decision-making frameworks from "The Decision Book" before signals
reach the Risk Manager.

Core Design Principles:
- Does NOT fetch price data (reviews analyst conclusions only)
- Acts as Devil's Advocate (questions unanimous consensus)
- Applies Decision Book frameworks based on signal distribution
- Outputs metadata/warnings (not modified signals)
- Uses configurable review modes for different skepticism levels
"""

import json
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal

from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.frameworks.loader import load_framework, get_triggered_frameworks
from src.frameworks.models import (
    SWOTAnalysis,
    PreMortemAnalysis,
    RubberBandAnalysis,
    RumsfeldMatrixAnalysis,
)


# Review mode configuration presets
REVIEW_MODE_CONFIG = {
    "conservative": {
        "description": "Maximum skepticism, questions everything",
        "confidence_range": (-20, 5),
        "unanimity_threshold": 0.85,  # 85% agreement triggers unanimity frameworks
        "conflict_threshold": 0.30,  # 30% disagreement triggers conflict frameworks
        "high_confidence_threshold": 75,
    },
    "balanced": {
        "description": "Standard devil's advocate (default)",
        "confidence_range": (-15, 10),
        "unanimity_threshold": 0.95,  # 95% agreement
        "conflict_threshold": 0.40,  # 40% disagreement
        "high_confidence_threshold": 85,
    },
    "aggressive": {
        "description": "Light touch, only flags major red flags",
        "confidence_range": (-10, 10),
        "unanimity_threshold": 1.0,  # 100% agreement only
        "conflict_threshold": 0.50,  # 50% disagreement
        "high_confidence_threshold": 90,
    },
}


class StrategicReviewSignal(BaseModel):
    """Output model for strategic review."""

    signal: Literal["validated", "caution", "warning", "critical"]
    confidence: int = Field(ge=0, le=100, description="Review confidence (0-100)")
    reasoning: str = Field(description="Overall review reasoning")
    confidence_adjustment: int = Field(
        ge=-20, le=10, description="Suggestion for portfolio manager confidence adjustment"
    )
    key_concerns: list[str] = Field(description="List of specific concerns identified")
    contrarian_thesis: str = Field(description="What if we're wrong?")
    # Optional framework-specific outputs
    swot_analysis: SWOTAnalysis | None = None
    pre_mortem_analysis: PreMortemAnalysis | None = None
    rubber_band_analysis: RubberBandAnalysis | None = None
    rumsfeld_analysis: RumsfeldMatrixAnalysis | None = None


def aggregate_signals(analyst_signals: dict, ticker: str) -> dict:
    """
    Aggregate all analyst signals for a ticker.

    Args:
        analyst_signals: Dict of agent_id -> signals per ticker
        ticker: Stock ticker to aggregate

    Returns:
        Dict with signal counts, percentages, and details
    """
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
    """
    Determine the consensus type based on signal distribution.

    Args:
        summary: Signal aggregation summary from aggregate_signals()
        config: Review mode configuration

    Returns:
        One of: "unanimous_bullish", "unanimous_bearish", "mixed", "high_conflict"
    """
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
    """
    Generate the strategic review using LLM with framework prompts.

    Args:
        ticker: Stock ticker being reviewed
        signal_summary: Aggregated signal data
        consensus_type: Determined consensus type
        frameworks: List of framework names to apply
        review_mode: Review mode name
        mode_config: Review mode configuration
        state: Agent state for LLM config
        agent_id: Agent identifier for logging

    Returns:
        StrategicReviewSignal with review results
    """
    # Load framework prompts
    framework_prompts = []
    for fw in frameworks:
        try:
            fw_def = load_framework(fw)
            framework_prompts.append(f"### {fw_def['name']}\n{fw_def['prompt_template']}")
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

Output a signal (review_status):
- "validated": Thesis holds up well under scrutiny
- "caution": Minor concerns identified
- "warning": Significant risks that warrant attention
- "critical": Major red flags that could invalidate the thesis

Return your analysis as JSON matching the required schema."""
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
1. Overall review signal (validated/caution/warning/critical) and reasoning
2. Confidence adjustment recommendation (integer within the allowed range)
3. Key concerns (list of specific issues)
4. Contrarian thesis (what if we're wrong?)"""
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
        "framework_prompts": "\n\n".join(framework_prompts) if framework_prompts else "No specific frameworks triggered - provide general contrarian analysis.",
    })

    def _default():
        return StrategicReviewSignal(
            signal="caution",
            confidence=50,
            reasoning="Unable to complete strategic review due to insufficient data or LLM error",
            confidence_adjustment=0,
            key_concerns=["Review could not be completed - recommend manual review"],
            contrarian_thesis="Insufficient data for contrarian analysis",
        )

    return call_llm(
        prompt=prompt,
        pydantic_model=StrategicReviewSignal,
        agent_name=agent_id,
        state=state,
        default_factory=_default,
    )


def strategic_reviewer_agent(state: AgentState, agent_id: str = "strategic_reviewer_agent"):
    """
    Strategic Reviewer Agent - Reviews analyst signals and applies decision-making frameworks.

    This agent acts as a Devil's Advocate, questioning analyst consensus and applying
    structured frameworks to identify potential risks and blind spots.

    Args:
        state: Current agent state with analyst signals
        agent_id: Unique identifier for this agent instance

    Returns:
        Updated state with strategic review added to analyst_signals
    """
    data = state["data"]
    tickers = data["tickers"]
    analyst_signals = data["analyst_signals"]
    review_mode = state["metadata"].get("review_mode", "balanced")
    mode_config = REVIEW_MODE_CONFIG.get(review_mode, REVIEW_MODE_CONFIG["balanced"])

    strategic_review = {}

    for ticker in tickers:
        progress.update_status(agent_id, ticker, "Aggregating analyst signals")

        # 1. Aggregate signals
        signal_summary = aggregate_signals(analyst_signals, ticker)

        # Skip if no analyst signals for this ticker
        if signal_summary["total"] == 0:
            strategic_review[ticker] = {
                "signal": "caution",
                "confidence": 0,
                "reasoning": "No analyst signals available for review",
                "confidence_adjustment": 0,
                "consensus_type": "none",
                "frameworks_applied": [],
                "key_concerns": ["No analyst data to review"],
                "contrarian_thesis": "Cannot provide contrarian view without analyst input",
            }
            progress.update_status(agent_id, ticker, "Done - No signals")
            continue

        # 2. Determine consensus type
        consensus_type = determine_consensus_type(signal_summary, mode_config)

        # 3. Select triggered frameworks
        frameworks = get_triggered_frameworks(
            consensus_type, signal_summary["avg_confidence"], mode_config
        )

        progress.update_status(
            agent_id, ticker, f"Applying {len(frameworks)} framework(s): {', '.join(frameworks)}"
        )

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

        # 5. Store review results
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

    # Create message for workflow
    message = HumanMessage(content=json.dumps(strategic_review), name=agent_id)

    # Show reasoning if requested
    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(strategic_review, "Strategic Reviewer")

    # Store in state
    state["data"]["analyst_signals"][agent_id] = strategic_review

    progress.update_status(agent_id, None, "Done")

    return {"messages": [message], "data": state["data"]}
