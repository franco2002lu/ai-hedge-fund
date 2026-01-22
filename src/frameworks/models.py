"""Pydantic models for decision-making framework outputs."""

from pydantic import BaseModel, Field
from typing import Literal


class FrameworkDefinition(BaseModel):
    """Base schema for framework JSON definitions."""

    name: str
    description: str
    prompt_template: str
    output_fields: list[str]


class SWOTAnalysis(BaseModel):
    """Structured SWOT analysis output."""

    strengths: list[str] = Field(description="Internal advantages")
    weaknesses: list[str] = Field(description="Internal disadvantages")
    opportunities: list[str] = Field(description="External favorable factors")
    threats: list[str] = Field(description="External unfavorable factors")


class PreMortemAnalysis(BaseModel):
    """Pre-mortem failure analysis output."""

    failure_scenarios: list[dict] = Field(
        description="List of {scenario, probability, impact}"
    )
    black_swan_candidates: list[str] = Field(
        description="Low-probability, high-impact events"
    )
    risk_level: Literal["low", "medium", "high", "critical"]


class RubberBandAnalysis(BaseModel):
    """Mean-reversion analysis output."""

    deviation_score: int = Field(
        ge=1, le=10, description="How far from historical mean (1-10)"
    )
    reversion_probability: int = Field(
        ge=0, le=100, description="Likelihood of mean reversion"
    )
    stretched_direction: Literal["bullish", "bearish", "neutral"]
    historical_context: str


class RumsfeldMatrixAnalysis(BaseModel):
    """Known/unknown categorization output."""

    known_knowns: list[str] = Field(description="Facts we're confident about")
    known_unknowns: list[str] = Field(
        description="Questions we know we need answered"
    )
    unknown_unknowns: list[str] = Field(description="Potential blind spots identified")
