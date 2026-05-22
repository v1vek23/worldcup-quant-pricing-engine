from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_probability_chart(prediction: dict, output_path: str = "reports/probability_chart.png"):
    """
    Saves a bar chart of the model's win/draw/loss probabilities.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    probabilities = prediction["probabilities"]

    labels = list(probabilities.keys())
    values = list(probabilities.values())

    plt.figure(figsize=(8, 5))
    plt.bar(labels, values)

    plt.title(f"Model Probabilities: {prediction['home_team']} vs {prediction['away_team']}")
    plt.ylabel("Probability")
    plt.ylim(0, 1)

    for i, value in enumerate(values):
        plt.text(i, value + 0.02, f"{value:.1%}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_fair_odds_chart(
    prediction: dict,
    market_odds: dict,
    output_path: str = "reports/fair_odds_vs_market.png"
):
    """
    Saves a chart comparing the model's fair odds to market odds.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    fair_odds = prediction["fair_odds"]

    labels = list(fair_odds.keys())
    model_values = [fair_odds[label] for label in labels]
    market_values = [market_odds[label] for label in labels]

    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 5))
    plt.bar(x - width / 2, model_values, width, label="Model Fair Odds")
    plt.bar(x + width / 2, market_values, width, label="Market Odds")

    plt.title("Fair Odds vs Market Odds")
    plt.ylabel("Decimal Odds")
    plt.xticks(x, labels)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_score_matrix_heatmap(
    prediction: dict,
    output_path: str = "reports/score_matrix_heatmap.png"
):
    """
    Saves a heatmap of possible scoreline probabilities.

    Row = home goals.
    Column = away goals.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    score_matrix = prediction["score_matrix"]

    plt.figure(figsize=(8, 6))
    plt.imshow(score_matrix, origin="lower")

    plt.title(f"Scoreline Probability Matrix: {prediction['home_team']} vs {prediction['away_team']}")
    plt.xlabel(f"{prediction['away_team']} Goals")
    plt.ylabel(f"{prediction['home_team']} Goals")

    plt.colorbar(label="Probability")

    max_goals = score_matrix.shape[0] - 1
    plt.xticks(range(max_goals + 1))
    plt.yticks(range(max_goals + 1))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_top_elo_ratings(
    ratings: dict,
    output_path: str = "reports/top_teams_elo.png",
    top_n: int = 20
):
    """
    Saves a bar chart of the top Elo-rated teams.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    ratings_df = pd.DataFrame(
        {
            "team": list(ratings.keys()),
            "elo": list(ratings.values()),
        }
    )

    ratings_df = ratings_df.sort_values("elo", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(ratings_df["team"], ratings_df["elo"])

    plt.title(f"Top {top_n} International Teams by Elo Rating")
    plt.xlabel("Elo Rating")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_calibration_plot(
    calibration_table: pd.DataFrame,
    output_path: str = "reports/calibration_plot.png"
):
    """
    Saves a calibration plot.

    The closer the model line is to the diagonal line,
    the better calibrated the model is.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    valid = calibration_table.dropna(
        subset=["avg_confidence", "accuracy"]
    )

    plt.figure(figsize=(7, 6))

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="Perfect Calibration"
    )

    plt.plot(
        valid["avg_confidence"],
        valid["accuracy"],
        marker="o",
        label="Model"
    )

    plt.title("Backtest Calibration Plot")
    plt.xlabel("Average Model Confidence")
    plt.ylabel("Actual Accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_confidence_histogram(
    results: pd.DataFrame,
    output_path: str = "reports/confidence_histogram.png"
):
    """
    Saves a histogram showing how confident the model usually is.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.hist(results["model_confidence"], bins=10)

    plt.title("Distribution of Model Confidence")
    plt.xlabel("Model Confidence")
    plt.ylabel("Number of Matches")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_outcome_metric_chart(
    outcome_metrics: pd.DataFrame,
    output_path: str = "reports/outcome_metrics.png"
):
    """
    Saves a chart showing the average probability assigned
    to each actual outcome type.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))

    plt.bar(
        outcome_metrics["actual_outcome"],
        outcome_metrics["avg_actual_outcome_prob"]
    )

    plt.title("Average Probability Assigned to Actual Outcome")
    plt.xlabel("Actual Outcome")
    plt.ylabel("Average Assigned Probability")
    plt.ylim(0, 1)

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_champion_probability_chart(
    simulation_summary: pd.DataFrame,
    output_path: str = "reports/tournament_champion_probabilities.png",
    top_n: int = 20
):
    """
    Saves a chart of the top teams by simulated champion probability.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    top = simulation_summary.sort_values(
        "champion_prob",
        ascending=False
    ).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(top["team"], top["champion_prob"])

    plt.title(f"Top {top_n} Simulated Champion Probabilities")
    plt.xlabel("Champion Probability")

    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_stage_probability_chart(
    simulation_summary: pd.DataFrame,
    output_path: str = "reports/tournament_stage_probabilities.png",
    top_n: int = 12
):
    """
    Saves a grouped chart showing how likely top teams are
    to reach later tournament stages.
    """

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    top = simulation_summary.sort_values(
        "champion_prob",
        ascending=False
    ).head(top_n)

    stages = [
        "round_16_prob",
        "quarterfinal_prob",
        "semifinal_prob",
        "final_prob",
        "champion_prob",
    ]

    x = np.arange(len(top))
    width = 0.15

    plt.figure(figsize=(13, 6))

    for i, stage in enumerate(stages):
        plt.bar(
            x + i * width,
            top[stage],
            width,
            label=stage.replace("_prob", "").replace("_", " ").title()
        )

    plt.title(f"Stage Reach Probabilities for Top {top_n} Teams")
    plt.xlabel("Team")
    plt.ylabel("Probability")
    plt.xticks(x + width * 2, top["team"], rotation=45, ha="right")
    plt.ylim(0, 1)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()