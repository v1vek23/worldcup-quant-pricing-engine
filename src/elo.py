from collections import defaultdict
import numpy as np
import pandas as pd


def get_match_result(home_score: int, away_score: int) -> float:
    """
    Converts a match score into the home team's result.

    1.0 means home team won.
    0.5 means draw.
    0.0 means home team lost.
    """

    if home_score > away_score:
        return 1.0

    if home_score == away_score:
        return 0.5

    return 0.0


def tournament_k_factor(tournament: str) -> float:
    """
    Chooses how much a match should affect a team's Elo rating.

    Important games should move ratings more.
    Friendlies should move ratings less.
    """

    tournament = tournament.lower()

    # World Cup games should matter a lot.
    if "fifa world cup" in tournament:
        return 45

    # Major continental competitions should also matter a lot.
    if "uefa euro" in tournament or "copa américa" in tournament:
        return 40

    # Qualifiers matter, but usually less than the actual tournament.
    if "qualification" in tournament:
        return 30

    # Friendlies should not change ratings too much.
    if "friendly" in tournament:
        return 15

    # Default value for tournaments we did not specifically name.
    return 25


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Standard Elo formula.

    It returns the probability that Team A gets a positive result
    against Team B.

    Higher rating difference means higher expected score.
    """

    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def add_elo_features(
    df: pd.DataFrame,
    base_rating: float = 1500,
    home_advantage: float = 50
) -> tuple[pd.DataFrame, dict]:
    """
    Goes through every match in date order and updates team Elo ratings.

    It also adds pre-match Elo features to the dataframe.

    Important idea:
    The model should only know each team's rating BEFORE the match,
    not after the match. Otherwise, we would accidentally leak future info.
    """

    # defaultdict gives every new team a starting rating of 1500.
    ratings = defaultdict(lambda: base_rating)

    home_elos = []
    away_elos = []
    elo_diffs = []

    # Loop through matches in chronological order.
    for row in df.itertuples(index=False):
        home_team = row.home_team
        away_team = row.away_team

        # These are the ratings before the match is played.
        home_rating = ratings[home_team]
        away_rating = ratings[away_team]

        # If the match is not neutral, give the home team a rating boost.
        # If neutral = 1, there is no home advantage.
        home_boost = 0 if row.neutral else home_advantage

        # This is one of the most important features for the model.
        # Positive means home team is stronger.
        # Negative means away team is stronger.
        rating_diff = home_rating + home_boost - away_rating

        # Store the pre-match ratings as features.
        home_elos.append(home_rating)
        away_elos.append(away_rating)
        elo_diffs.append(rating_diff)

        # Expected home result based on Elo.
        expected_home = expected_score(home_rating + home_boost, away_rating)

        # Actual home result based on the real score.
        actual_home = get_match_result(row.home_score, row.away_score)

        # Bigger wins should move Elo ratings more than narrow wins.
        goal_diff = abs(row.home_score - row.away_score)
        margin_multiplier = np.log(goal_diff + 1) + 1

        # K factor controls how strongly the match changes the ratings.
        k = tournament_k_factor(row.tournament) * margin_multiplier

        # Elo update:
        # If actual result is better than expected, rating goes up.
        # If actual result is worse than expected, rating goes down.
        ratings[home_team] += k * (actual_home - expected_home)
        ratings[away_team] -= k * (actual_home - expected_home)

    # Add the Elo features to a copy of the dataframe.
    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = elo_diffs

    # Return both:
    # 1. the dataframe with Elo features
    # 2. the final team ratings after all matches
    return df, dict(ratings)