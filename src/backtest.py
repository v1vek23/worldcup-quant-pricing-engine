import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_results
from src.model import WorldCupPricer


OUTCOMES = ["home_win", "draw", "away_win"]


def get_actual_outcome(home_score: int, away_score: int) -> str:
    """
    Converts the real match score into one of three outcomes.
    """

    if home_score > away_score:
        return "home_win"

    if home_score == away_score:
        return "draw"

    return "away_win"


def brier_score(probabilities: dict, actual_outcome: str) -> float:
    """
    Measures how close the predicted probabilities were to reality.

    Lower is better.

    Example:
    If the model says home_win = 90% and home_win happens,
    that is a good Brier score.

    If the model says home_win = 90% and away_win happens,
    that is a bad Brier score.
    """

    total = 0

    for outcome in OUTCOMES:
        predicted_prob = probabilities[outcome]
        actual_value = 1 if outcome == actual_outcome else 0

        total += (predicted_prob - actual_value) ** 2

    return total


def log_loss_score(probabilities: dict, actual_outcome: str) -> float:
    """
    Measures how much probability the model assigned to the true outcome.

    Lower is better.

    If the model gives a very low probability to what actually happens,
    log loss punishes the model heavily.
    """

    actual_prob = probabilities[actual_outcome]

    # Prevent log(0), which would crash.
    actual_prob = max(actual_prob, 1e-12)

    return -np.log(actual_prob)


def run_world_cup_backtest(
    year: int = 2022,
    start_year: int = 2000,
    output_dir: str = "reports"
) -> tuple[pd.DataFrame, dict]:
    """
    Backtests the model on one past World Cup.

    Important:
    The model trains only on matches before that World Cup starts.
    Then it predicts the World Cup matches one by one.
    """

    print("Loading data...")
    df = load_results(start_year=start_year)

    # Only use actual FIFA World Cup tournament matches.
    world_cup_matches = df[
        (df["tournament"].str.lower() == "fifa world cup")
        & (df["date"].dt.year == year)
    ].copy()

    if world_cup_matches.empty:
        raise ValueError(f"No FIFA World Cup matches found for {year}.")

    world_cup_matches = world_cup_matches.sort_values("date").reset_index(drop=True)

    tournament_start = world_cup_matches["date"].min()

    # This prevents data leakage.
    # The model cannot train on matches from the tournament it is predicting.
    train_df = df[df["date"] < tournament_start].copy()

    print(f"Training on {len(train_df):,} matches before {tournament_start.date()}...")
    print(f"Testing on {len(world_cup_matches):,} matches from the {year} World Cup...")

    pricer = WorldCupPricer()
    pricer.fit(train_df)

    rows = []

    for match in world_cup_matches.itertuples(index=False):
        prediction = pricer.predict_match(
            home_team=match.home_team,
            away_team=match.away_team,
            tournament=match.tournament,
            neutral=match.neutral
        )

        probabilities = prediction["probabilities"]

        actual_outcome = get_actual_outcome(
            home_score=match.home_score,
            away_score=match.away_score
        )

        predicted_outcome = max(probabilities, key=probabilities.get)

        actual_prob = probabilities[actual_outcome]

        row = {
            "date": match.date,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "home_score": match.home_score,
            "away_score": match.away_score,
            "actual_outcome": actual_outcome,
            "predicted_outcome": predicted_outcome,
            "correct": predicted_outcome == actual_outcome,
            "home_win_prob": probabilities["home_win"],
            "draw_prob": probabilities["draw"],
            "away_win_prob": probabilities["away_win"],
            "actual_outcome_prob": actual_prob,
            "brier_score": brier_score(probabilities, actual_outcome),
            "log_loss": log_loss_score(probabilities, actual_outcome),
            "home_fair_odds": prediction["fair_odds"]["home_win"],
            "draw_fair_odds": prediction["fair_odds"]["draw"],
            "away_fair_odds": prediction["fair_odds"]["away_win"],
        }

        rows.append(row)

    results = pd.DataFrame(rows)

    summary = {
        "year": year,
        "matches": len(results),
        "accuracy": results["correct"].mean(),
        "avg_brier_score": results["brier_score"].mean(),
        "avg_log_loss": results["log_loss"].mean(),
        "avg_actual_outcome_prob": results["actual_outcome_prob"].mean(),
        "avg_model_confidence": results[
            ["home_win_prob", "draw_prob", "away_win_prob"]
        ].max(axis=1).mean(),
    }

    save_backtest_outputs(results, summary, output_dir)

    return results, summary


def save_backtest_outputs(
    results: pd.DataFrame,
    summary: dict,
    output_dir: str = "reports"
):
    """
    Saves the backtest results to CSV and markdown files.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    year = summary["year"]

    csv_path = Path(output_dir) / f"backtest_{year}_world_cup.csv"
    summary_path = Path(output_dir) / f"backtest_{year}_summary.md"

    results.to_csv(csv_path, index=False)

    with open(summary_path, "w") as file:
        file.write(f"# {year} World Cup Backtest Summary\n\n")

        file.write("The model was trained only on matches before the tournament began, then tested on the World Cup matches from that year.\n\n")

        file.write("## Summary Metrics\n\n")
        file.write(f"- Matches tested: {summary['matches']}\n")
        file.write(f"- Prediction accuracy: {summary['accuracy']:.2%}\n")
        file.write(f"- Average Brier score: {summary['avg_brier_score']:.4f}\n")
        file.write(f"- Average log loss: {summary['avg_log_loss']:.4f}\n")
        file.write(f"- Average probability assigned to actual outcome: {summary['avg_actual_outcome_prob']:.2%}\n")
        file.write(f"- Average model confidence: {summary['avg_model_confidence']:.2%}\n\n")

        file.write("## Interpretation\n\n")
        file.write("Accuracy measures how often the model's most likely outcome happened.\n\n")
        file.write("Brier score and log loss measure probability quality. These are more useful than accuracy because the model is trying to estimate probabilities, not just pick winners.\n\n")
        file.write("Average probability assigned to the actual outcome shows how much confidence the model placed on what really happened.\n")

    print(f"\nSaved detailed results to {csv_path}")
    print(f"Saved summary to {summary_path}")


def print_summary(summary: dict):
    """
    Prints the backtest summary in the terminal.
    """

    print("\n==============================")
    print(f"{summary['year']} WORLD CUP BACKTEST")
    print("==============================")

    print(f"Matches tested: {summary['matches']}")
    print(f"Prediction accuracy: {summary['accuracy']:.2%}")
    print(f"Average Brier score: {summary['avg_brier_score']:.4f}")
    print(f"Average log loss: {summary['avg_log_loss']:.4f}")
    print(f"Average probability assigned to actual outcome: {summary['avg_actual_outcome_prob']:.2%}")
    print(f"Average model confidence: {summary['avg_model_confidence']:.2%}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Backtest the World Cup Quant Pricing Engine"
    )

    parser.add_argument(
        "--year",
        type=int,
        default=2022,
        help="World Cup year to backtest"
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="First year of historical data to use"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    _, summary = run_world_cup_backtest(
        year=args.year,
        start_year=args.start_year
    )

    print_summary(summary)


if __name__ == "__main__":
    main()