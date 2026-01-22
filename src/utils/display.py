from colorama import Fore, Style
from tabulate import tabulate
from .analysts import ANALYST_ORDER
import os
import json


def sort_agent_signals(signals):
    """Sort agent signals in a consistent order."""
    # Create order mapping from ANALYST_ORDER
    analyst_order = {display: idx for idx, (display, _) in enumerate(ANALYST_ORDER)}
    analyst_order["Risk Management"] = len(ANALYST_ORDER)  # Add Risk Management at the end

    return sorted(signals, key=lambda x: analyst_order.get(x[0], 999))


def print_trading_output(result: dict) -> None:
    """
    Print formatted trading results with colored tables for multiple tickers.

    Args:
        result (dict): Dictionary containing decisions and analyst signals for multiple tickers
    """
    decisions = result.get("decisions")
    if not decisions:
        print(f"{Fore.RED}No trading decisions available{Style.RESET_ALL}")
        return

    # Print decisions for each ticker
    for ticker, decision in decisions.items():
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Analysis for {Fore.CYAN}{ticker}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{Style.BRIGHT}{'=' * 50}{Style.RESET_ALL}")

        # Prepare analyst signals table for this ticker
        table_data = []
        for agent, signals in result.get("analyst_signals", {}).items():
            if ticker not in signals:
                continue
                
            # Skip Risk Management agent in the signals section
            if agent == "risk_management_agent":
                continue

            signal = signals[ticker]
            agent_name = agent.replace("_agent", "").replace("_", " ").title()
            signal_type = signal.get("signal", "").upper()
            confidence = signal.get("confidence", 0)

            signal_color = {
                "BULLISH": Fore.GREEN,
                "BEARISH": Fore.RED,
                "NEUTRAL": Fore.YELLOW,
            }.get(signal_type, Fore.WHITE)
            
            # Get reasoning if available
            reasoning_str = ""
            if "reasoning" in signal and signal["reasoning"]:
                reasoning = signal["reasoning"]
                
                # Handle different types of reasoning (string, dict, etc.)
                if isinstance(reasoning, str):
                    reasoning_str = reasoning
                elif isinstance(reasoning, dict):
                    # Convert dict to string representation
                    reasoning_str = json.dumps(reasoning, indent=2)
                else:
                    # Convert any other type to string
                    reasoning_str = str(reasoning)
                
                # Wrap long reasoning text to make it more readable
                wrapped_reasoning = ""
                current_line = ""
                # Use a fixed width of 60 characters to match the table column width
                max_line_length = 60
                for word in reasoning_str.split():
                    if len(current_line) + len(word) + 1 > max_line_length:
                        wrapped_reasoning += current_line + "\n"
                        current_line = word
                    else:
                        if current_line:
                            current_line += " " + word
                        else:
                            current_line = word
                if current_line:
                    wrapped_reasoning += current_line
                
                reasoning_str = wrapped_reasoning

            table_data.append(
                [
                    f"{Fore.CYAN}{agent_name}{Style.RESET_ALL}",
                    f"{signal_color}{signal_type}{Style.RESET_ALL}",
                    f"{Fore.WHITE}{confidence}%{Style.RESET_ALL}",
                    f"{Fore.WHITE}{reasoning_str}{Style.RESET_ALL}",
                ]
            )

        # Sort the signals according to the predefined order
        table_data = sort_agent_signals(table_data)

        print(f"\n{Fore.WHITE}{Style.BRIGHT}AGENT ANALYSIS:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        print(
            tabulate(
                table_data,
                headers=[f"{Fore.WHITE}Agent", "Signal", "Confidence", "Reasoning"],
                tablefmt="grid",
                colalign=("left", "center", "right", "left"),
            )
        )

        # Print Trading Decision Table
        action = decision.get("action", "").upper()
        action_color = {
            "BUY": Fore.GREEN,
            "SELL": Fore.RED,
            "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN,
            "SHORT": Fore.RED,
        }.get(action, Fore.WHITE)

        # Get reasoning and format it
        reasoning = decision.get("reasoning", "")
        # Wrap long reasoning text to make it more readable
        wrapped_reasoning = ""
        if reasoning:
            current_line = ""
            # Use a fixed width of 60 characters to match the table column width
            max_line_length = 60
            for word in reasoning.split():
                if len(current_line) + len(word) + 1 > max_line_length:
                    wrapped_reasoning += current_line + "\n"
                    current_line = word
                else:
                    if current_line:
                        current_line += " " + word
                    else:
                        current_line = word
            if current_line:
                wrapped_reasoning += current_line

        decision_data = [
            ["Action", f"{action_color}{action}{Style.RESET_ALL}"],
            ["Quantity", f"{action_color}{decision.get('quantity')}{Style.RESET_ALL}"],
            [
                "Confidence",
                f"{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}",
            ],
            ["Reasoning", f"{Fore.WHITE}{wrapped_reasoning}{Style.RESET_ALL}"],
        ]
        
        print(f"\n{Fore.WHITE}{Style.BRIGHT}TRADING DECISION:{Style.RESET_ALL} [{Fore.CYAN}{ticker}{Style.RESET_ALL}]")
        print(tabulate(decision_data, tablefmt="grid", colalign=("left", "left")))

    # Print Portfolio Summary
    print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")
    portfolio_data = []
    
    # Extract portfolio manager reasoning (common for all tickers)
    portfolio_manager_reasoning = None
    for ticker, decision in decisions.items():
        if decision.get("reasoning"):
            portfolio_manager_reasoning = decision.get("reasoning")
            break
            
    analyst_signals = result.get("analyst_signals", {})
    for ticker, decision in decisions.items():
        action = decision.get("action", "").upper()
        action_color = {
            "BUY": Fore.GREEN,
            "SELL": Fore.RED,
            "HOLD": Fore.YELLOW,
            "COVER": Fore.GREEN,
            "SHORT": Fore.RED,
        }.get(action, Fore.WHITE)

        # Calculate analyst signal counts
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        if analyst_signals:
            for agent, signals in analyst_signals.items():
                if ticker in signals:
                    signal = signals[ticker].get("signal", "").upper()
                    if signal == "BULLISH":
                        bullish_count += 1
                    elif signal == "BEARISH":
                        bearish_count += 1
                    elif signal == "NEUTRAL":
                        neutral_count += 1

        portfolio_data.append(
            [
                f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
                f"{action_color}{action}{Style.RESET_ALL}",
                f"{action_color}{decision.get('quantity')}{Style.RESET_ALL}",
                f"{Fore.WHITE}{decision.get('confidence'):.1f}%{Style.RESET_ALL}",
                f"{Fore.GREEN}{bullish_count}{Style.RESET_ALL}",
                f"{Fore.RED}{bearish_count}{Style.RESET_ALL}",
                f"{Fore.YELLOW}{neutral_count}{Style.RESET_ALL}",
            ]
        )

    headers = [
        f"{Fore.WHITE}Ticker",
        f"{Fore.WHITE}Action",
        f"{Fore.WHITE}Quantity",
        f"{Fore.WHITE}Confidence",
        f"{Fore.WHITE}Bullish",
        f"{Fore.WHITE}Bearish",
        f"{Fore.WHITE}Neutral",
    ]
    
    # Print the portfolio summary table
    print(
        tabulate(
            portfolio_data,
            headers=headers,
            tablefmt="grid",
            colalign=("left", "center", "right", "right", "center", "center", "center"),
        )
    )
    
    # Print Portfolio Manager's reasoning if available
    if portfolio_manager_reasoning:
        # Handle different types of reasoning (string, dict, etc.)
        reasoning_str = ""
        if isinstance(portfolio_manager_reasoning, str):
            reasoning_str = portfolio_manager_reasoning
        elif isinstance(portfolio_manager_reasoning, dict):
            # Convert dict to string representation
            reasoning_str = json.dumps(portfolio_manager_reasoning, indent=2)
        else:
            # Convert any other type to string
            reasoning_str = str(portfolio_manager_reasoning)
            
        # Wrap long reasoning text to make it more readable
        wrapped_reasoning = ""
        current_line = ""
        # Use a fixed width of 60 characters to match the table column width
        max_line_length = 60
        for word in reasoning_str.split():
            if len(current_line) + len(word) + 1 > max_line_length:
                wrapped_reasoning += current_line + "\n"
                current_line = word
            else:
                if current_line:
                    current_line += " " + word
                else:
                    current_line = word
        if current_line:
            wrapped_reasoning += current_line
            
        print(f"\n{Fore.WHITE}{Style.BRIGHT}Portfolio Strategy:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{wrapped_reasoning}{Style.RESET_ALL}")


def print_backtest_results(table_rows: list) -> None:
    """Print the backtest results in a nicely formatted table"""
    # Clear the screen
    os.system("cls" if os.name == "nt" else "clear")

    # Split rows into ticker rows and summary rows
    ticker_rows = []
    summary_rows = []

    for row in table_rows:
        if isinstance(row[1], str) and "PORTFOLIO SUMMARY" in row[1]:
            summary_rows.append(row)
        else:
            ticker_rows.append(row)

    # Display latest portfolio summary
    if summary_rows:
        # Pick the most recent summary by date (YYYY-MM-DD)
        latest_summary = max(summary_rows, key=lambda r: r[0])
        print(f"\n{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY:{Style.RESET_ALL}")

        # Adjusted indexes after adding Long/Short Shares
        position_str = latest_summary[7].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        cash_str     = latest_summary[8].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")
        total_str    = latest_summary[9].split("$")[1].split(Style.RESET_ALL)[0].replace(",", "")

        print(f"Cash Balance: {Fore.CYAN}${float(cash_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Position Value: {Fore.YELLOW}${float(position_str):,.2f}{Style.RESET_ALL}")
        print(f"Total Value: {Fore.WHITE}${float(total_str):,.2f}{Style.RESET_ALL}")
        print(f"Portfolio Return: {latest_summary[10]}")
        if len(latest_summary) > 14 and latest_summary[14]:
            print(f"Benchmark Return: {latest_summary[14]}")

        # Display performance metrics if available
        if latest_summary[11]:  # Sharpe ratio
            print(f"Sharpe Ratio: {latest_summary[11]}")
        if latest_summary[12]:  # Sortino ratio
            print(f"Sortino Ratio: {latest_summary[12]}")
        if latest_summary[13]:  # Max drawdown
            print(f"Max Drawdown: {latest_summary[13]}")

    # Add vertical spacing
    print("\n" * 2)

    # Print the table with just ticker rows
    print(
        tabulate(
            ticker_rows,
            headers=[
                "Date",
                "Ticker",
                "Action",
                "Quantity",
                "Price",
                "Long Shares",
                "Short Shares",
                "Position Value",
            ],
            tablefmt="grid",
            colalign=(
                "left",    # Date
                "left",    # Ticker
                "center",  # Action
                "right",   # Quantity
                "right",   # Price
                "right",   # Long Shares
                "right",   # Short Shares
                "right",   # Position Value
            ),
        )
    )

    # Add vertical spacing
    print("\n" * 4)


def format_backtest_row(
    date: str,
    ticker: str,
    action: str,
    quantity: float,
    price: float,
    long_shares: float = 0,
    short_shares: float = 0,
    position_value: float = 0,
    is_summary: bool = False,
    total_value: float = None,
    return_pct: float = None,
    cash_balance: float = None,
    total_position_value: float = None,
    sharpe_ratio: float = None,
    sortino_ratio: float = None,
    max_drawdown: float = None,
    benchmark_return_pct: float | None = None,
) -> list[any]:
    """Format a row for the backtest results table"""
    # Color the action
    action_color = {
        "BUY": Fore.GREEN,
        "COVER": Fore.GREEN,
        "SELL": Fore.RED,
        "SHORT": Fore.RED,
        "HOLD": Fore.WHITE,
    }.get(action.upper(), Fore.WHITE)

    if is_summary:
        return_color = Fore.GREEN if return_pct >= 0 else Fore.RED
        benchmark_str = ""
        if benchmark_return_pct is not None:
            bench_color = Fore.GREEN if benchmark_return_pct >= 0 else Fore.RED
            benchmark_str = f"{bench_color}{benchmark_return_pct:+.2f}%{Style.RESET_ALL}"
        return [
            date,
            f"{Fore.WHITE}{Style.BRIGHT}PORTFOLIO SUMMARY{Style.RESET_ALL}",
            "",  # Action
            "",  # Quantity
            "",  # Price
            "",  # Long Shares
            "",  # Short Shares
            f"{Fore.YELLOW}${total_position_value:,.2f}{Style.RESET_ALL}",  # Total Position Value
            f"{Fore.CYAN}${cash_balance:,.2f}{Style.RESET_ALL}",  # Cash Balance
            f"{Fore.WHITE}${total_value:,.2f}{Style.RESET_ALL}",  # Total Value
            f"{return_color}{return_pct:+.2f}%{Style.RESET_ALL}",  # Return
            f"{Fore.YELLOW}{sharpe_ratio:.2f}{Style.RESET_ALL}" if sharpe_ratio is not None else "",  # Sharpe Ratio
            f"{Fore.YELLOW}{sortino_ratio:.2f}{Style.RESET_ALL}" if sortino_ratio is not None else "",  # Sortino Ratio
            f"{Fore.RED}{max_drawdown:.2f}%{Style.RESET_ALL}" if max_drawdown is not None else "",  # Max Drawdown (signed)
            benchmark_str,  # Benchmark (S&P 500)
        ]
    else:
        return [
            date,
            f"{Fore.CYAN}{ticker}{Style.RESET_ALL}",
            f"{action_color}{action.upper()}{Style.RESET_ALL}",
            f"{action_color}{quantity:,.0f}{Style.RESET_ALL}",
            f"{Fore.WHITE}{price:,.2f}{Style.RESET_ALL}",
            f"{Fore.GREEN}{long_shares:,.0f}{Style.RESET_ALL}",   # Long Shares
            f"{Fore.RED}{short_shares:,.0f}{Style.RESET_ALL}",    # Short Shares
            f"{Fore.YELLOW}{position_value:,.2f}{Style.RESET_ALL}",
        ]


def print_strategic_review_output(review_data: dict, ticker: str) -> None:
    """
    Print strategic review with color-coded status.

    Args:
        review_data: Dictionary containing strategic review for all tickers
        ticker: The specific ticker to display
    """
    status_colors = {
        "validated": Fore.GREEN,
        "caution": Fore.YELLOW,
        "warning": Fore.RED,
        "critical": Fore.MAGENTA + Style.BRIGHT,
    }

    review = review_data.get(ticker, {})
    status = review.get("signal", "unknown")
    color = status_colors.get(status, Fore.WHITE)

    print(f"\n{'=' * 60}")
    print(f"{Fore.WHITE}{Style.BRIGHT}STRATEGIC REVIEW: {Fore.CYAN}{ticker}{Style.RESET_ALL}")
    print(f"{'=' * 60}")
    print(f"Status: {color}{status.upper()}{Style.RESET_ALL}")
    print(f"Confidence Adjustment: {review.get('confidence_adjustment', 0):+d}")
    print(f"Consensus Type: {review.get('consensus_type', 'unknown')}")
    frameworks = review.get("frameworks_applied", [])
    print(f"Frameworks Applied: {', '.join(frameworks) if frameworks else 'None'}")

    print(f"\n{Fore.CYAN}Key Concerns:{Style.RESET_ALL}")
    for concern in review.get("key_concerns", []):
        print(f"  {Fore.YELLOW}\u2022{Style.RESET_ALL} {concern}")

    print(f"\n{Fore.YELLOW}Contrarian Thesis:{Style.RESET_ALL}")
    print(f"  {review.get('contrarian_thesis', 'N/A')}")

    print(f"\n{Fore.WHITE}Reasoning:{Style.RESET_ALL}")
    print(f"  {review.get('reasoning', 'N/A')}")

    # Print framework-specific analyses if available
    if review.get("swot_analysis"):
        print_swot_matrix(review["swot_analysis"])
    if review.get("pre_mortem_analysis"):
        print_pre_mortem_analysis(review["pre_mortem_analysis"])
    if review.get("rubber_band_analysis"):
        print_rubber_band_analysis(review["rubber_band_analysis"])


def print_swot_matrix(swot: dict) -> None:
    """
    Print SWOT analysis as a 2x2 grid with colored quadrants.

    Args:
        swot: Dictionary with strengths, weaknesses, opportunities, threats
    """
    if not swot:
        return

    print(f"\n{Fore.CYAN}SWOT ANALYSIS{Style.RESET_ALL}")
    print("\u250c" + "\u2500" * 30 + "\u252c" + "\u2500" * 30 + "\u2510")

    # Strengths (Green) | Weaknesses (Red)
    print(f"\u2502 {Fore.GREEN}STRENGTHS{Style.RESET_ALL}" + " " * 20 +
          f"\u2502 {Fore.RED}WEAKNESSES{Style.RESET_ALL}" + " " * 19 + "\u2502")
    strengths = swot.get("strengths", []) or []
    weaknesses = swot.get("weaknesses", []) or []
    max_len = max(len(strengths), len(weaknesses), 1)
    for i in range(max_len):
        s = strengths[i][:26] if i < len(strengths) else ""
        w = weaknesses[i][:26] if i < len(weaknesses) else ""
        print(f"\u2502 \u2022 {s:<26} \u2502 \u2022 {w:<26} \u2502")

    print("\u251c" + "\u2500" * 30 + "\u253c" + "\u2500" * 30 + "\u2524")

    # Opportunities (Blue) | Threats (Yellow)
    print(f"\u2502 {Fore.BLUE}OPPORTUNITIES{Style.RESET_ALL}" + " " * 16 +
          f"\u2502 {Fore.YELLOW}THREATS{Style.RESET_ALL}" + " " * 22 + "\u2502")
    opportunities = swot.get("opportunities", []) or []
    threats = swot.get("threats", []) or []
    max_len = max(len(opportunities), len(threats), 1)
    for i in range(max_len):
        o = opportunities[i][:26] if i < len(opportunities) else ""
        t = threats[i][:26] if i < len(threats) else ""
        print(f"\u2502 \u2022 {o:<26} \u2502 \u2022 {t:<26} \u2502")

    print("\u2514" + "\u2500" * 30 + "\u2534" + "\u2500" * 30 + "\u2518")


def print_pre_mortem_analysis(pre_mortem: dict) -> None:
    """
    Print pre-mortem analysis with risk indicators.

    Args:
        pre_mortem: Dictionary with failure_scenarios, black_swan_candidates, risk_level
    """
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
            prob = scenario.get("probability", "?")
            desc = scenario.get("scenario", "Unknown")
            print(f"  [{prob}] {desc}")
        else:
            print(f"  \u2022 {scenario}")

    print(f"\n{Fore.MAGENTA}Black Swan Candidates:{Style.RESET_ALL}")
    for swan in pre_mortem.get("black_swan_candidates", []):
        print(f"  \u26a0 {swan}")


def print_rubber_band_analysis(rubber_band: dict) -> None:
    """
    Print rubber band analysis with deviation meter.

    Args:
        rubber_band: Dictionary with deviation_score, reversion_probability, stretched_direction
    """
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
            meter += "\u2588"  # Full block
        else:
            meter += "\u2591"  # Light shade
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
    context = rubber_band.get("historical_context", "N/A")
    print(f"Context: {context}")
