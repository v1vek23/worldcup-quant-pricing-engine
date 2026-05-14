import argparse
from pathlib import Path

from src.data import load_results
from src.model import WorldCupPricer
from src.market import find_edges
from src.visuals import (
    save_probability_chart,
    save_fair_odds_chart,
    save_score_matrix_heatmap,
    save_top_elo_ratings,
)


def print_prediction(prediction: dict):
    """
    Prints the model output for one match.
    """

    print("\n==============================")
    print("MATCH PRICING")
    print("==============================")

    print(f"{prediction['home_team']} vs {prediction['away_team']}")

    print("\nExpected goals:")
    print(f"{prediction['home_team']}: {prediction['lambda_home']:.3f}")
    print(f"{prediction['away_team']}: {prediction['lambda_away']:.3f}")

    print("\nModel probabilities:")
    for outcome, prob in prediction["probabilities"].items():
        print(f"{outcome}: {prob:.3%}")

    print("\nFair odds:")
    for outcome, odds in prediction["fair_odds"].items():
        print(f"{outcome}: {odds:.2f}")


def save_example_output(prediction: dict, edges, output_path: str = "reports/example_output.txt"):
    """
    Saves the terminal-style output to a text file for GitHub.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as file:
        file.write("==============================\n")
        file.write("MATCH PRICING\n")
        file.write("==============================\n")

        file.write(f"{prediction['home_team']} vs {prediction['away_team']}\n\n")

        file.write("Expected goals:\n")
        file.write(f"{prediction['home_team']}: {prediction['lambda_home']:.3f}\n")
        file.write(f"{prediction['away_team']}: {prediction['lambda_away']:.3f}\n\n")

        file.write("Model probabilities:\n")
        for outcome, prob in prediction["probabilities"].items():
            file.write(f"{outcome}: {prob:.3%}\n")

        file.write("\nFair odds:\n")
        for outcome, odds in prediction["fair_odds"].items():
            file.write(f"{outcome}: {odds:.2f}\n")

        file.write("\n==============================\n")
        file.write("MARKET COMPARISON\n")
        file.write("==============================\n")
        file.write(edges.to_string(index=False))


def parse_args():
    """
    Lets the user run the model from the command line.

    Example:
    python main.py --home Argentina --away France --home-odds 2.65 --draw-odds 3.20 --away-odds 2.80
    """

    parser = argparse.ArgumentParser(
        description="World Cup Quant Pricing Engine"
    )

    parser.add_argument("--home", type=str, default="Argentina", help="Home team name")
    parser.add_argument("--away", type=str, default="France", help="Away team name")

    parser.add_argument("--home-odds", type=float, default=2.65, help="Market odds for home win")
    parser.add_argument("--draw-odds", type=float, default=3.20, help="Market odds for draw")
    parser.add_argument("--away-odds", type=float, default=2.80, help="Market odds for away win")

    parser.add_argument("--bankroll", type=float, default=1000, help="Bankroll size")
    parser.add_argument("--start-year", type=int, default=2000, help="First year of historical data")

    parser.add_argument(
        "--no-charts",
        action="store_true",
        help="Skip saving charts"
    )

    return parser.parse_args()


def main():
    """
    Full project pipeline:

    1. Load international match data
    2. Train Elo and Poisson pricing model
    3. Price one match
    4. Compare model probabilities to market odds
    5. Calculate fractional Kelly sizing
    6. Save charts and example output
    """

    args = parse_args()

    print("Loading data...")
    df = load_results(start_year=args.start_year)

    print(f"Loaded {len(df):,} matches.")

    print("Training World Cup pricing model...")
    pricer = WorldCupPricer()
    pricer.fit(df)

    prediction = pricer.predict_match(
        home_team=args.home,
        away_team=args.away,
        tournament="FIFA World Cup",
        neutral=1
    )

    print_prediction(prediction)

    market_odds = {
        "home_win": args.home_odds,
        "draw": args.draw_odds,
        "away_win": args.away_odds,
    }

    edges = find_edges(
        model_probs=prediction["probabilities"],
        market_odds=market_odds,
        bankroll=args.bankroll
    )

    print("\n==============================")
    print("MARKET COMPARISON")
    print("==============================")
    print(edges.to_string(index=False))

    save_example_output(prediction, edges)

    if not args.no_charts:
        print("\nSaving charts to reports/...")
        save_probability_chart(prediction)
        save_fair_odds_chart(prediction, market_odds)
        save_score_matrix_heatmap(prediction)
        save_top_elo_ratings(pricer.ratings)

    print("\nDone.")


if __name__ == "__main__":
    main()