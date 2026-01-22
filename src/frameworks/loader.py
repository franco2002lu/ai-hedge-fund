"""Framework loading and selection utilities."""

import json
from pathlib import Path

FRAMEWORK_DIR = Path(__file__).parent / "definitions"


def load_framework(framework_type: str) -> dict:
    """
    Load a framework definition from JSON.

    Args:
        framework_type: Name of the framework (e.g., "swot", "pre_mortem")

    Returns:
        Dictionary containing framework definition

    Raises:
        FileNotFoundError: If framework definition file doesn't exist
    """
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
        consensus_type: One of "unanimous_bullish", "unanimous_bearish", "mixed", "high_conflict"
        avg_confidence: Average confidence across analysts (0-100)
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

    # Cap at 3 frameworks to avoid prompt bloat
    return frameworks[:3]
