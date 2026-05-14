import pandas as pd


def remove_vig(decimal_odds: dict) -> dict:
    """
    Converts bookmaker odds into no-vig implied probabilities.

    Bookmakers build in a margin, also called the vig.
    This means their implied probabilities usually add up to more than 100%.

    This function removes that margin so we can compare our model
    to the market more fairly.
    """

    # Convert each decimal odd into an implied probability.
    # Example: odds of 2.00 means implied probability of 1 / 2.00 = 50%.
    implied = {
        outcome: 1 / odds
        for outcome, odds in decimal_odds.items()
    }

    # This total is usually greater than 1 because of bookmaker margin.
    total = sum(implied.values())

    # Divide by the total to force probabilities to add up to 1.
    no_vig_probs = {
        outcome: prob / total
        for outcome, prob in implied.items()
    }

    return no_vig_probs


def kelly_fraction(
    model_prob: float,
    decimal_odds: float,
    fraction: float = 0.25,
    cap: float = 0.05
) -> float:
    """
    Calculates the Kelly bet size.

    Kelly tells you what fraction of your bankroll to bet
    when you believe you have an edge.

    Full Kelly can be too aggressive, so we use fractional Kelly.
    """

    # b is the net profit per $1 bet.
    # Example: decimal odds 2.50 means b = 1.50.
    b = decimal_odds - 1

    # q is the probability of losing.
    q = 1 - model_prob

    # Full Kelly formula.
    full_kelly = (b * model_prob - q) / b

    # Fractional Kelly makes the bet smaller and safer.
    fractional_kelly = full_kelly * fraction

    # Never bet negative.
    # Also cap the bet so the model does not suggest huge positions.
    return max(0, min(fractional_kelly, cap))


def find_edges(
    model_probs: dict,
    market_odds: dict,
    bankroll: float = 1000
) -> pd.DataFrame:
    """
    Compares our model probabilities to market probabilities.

    Positive edge means our model thinks the outcome is more likely
    than the market thinks.

    That is the kind of situation a quant betting model looks for.
    """

    # Convert bookmaker odds into no-vig probabilities.
    market_probs = remove_vig(market_odds)

    rows = []

    for outcome in model_probs:
        model_prob = model_probs[outcome]
        market_prob = market_probs[outcome]
        odds = market_odds[outcome]

        # Edge is the difference between our probability and the market probability.
        edge = model_prob - market_prob

        # Use Kelly to decide how much to bet.
        kelly = kelly_fraction(
            model_prob=model_prob,
            decimal_odds=odds
        )

        rows.append(
            {
                "outcome": outcome,
                "model_prob": model_prob,
                "market_prob_no_vig": market_prob,
                "edge": edge,
                "market_odds": odds,
                "fair_odds": 1 / model_prob,
                "kelly_fraction": kelly,
                "suggested_stake": bankroll * kelly,
            }
        )

    result = pd.DataFrame(rows)

    # Put the best opportunities at the top.
    result = result.sort_values("edge", ascending=False)

    return result