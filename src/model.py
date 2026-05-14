import numpy as np
import pandas as pd
from scipy.stats import poisson

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.elo import add_elo_features


def make_encoder():
    """
    Creates a OneHotEncoder.

    Different versions of sklearn use slightly different argument names.
    This try/except makes the code work on more computers.
    """

    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def make_poisson_pipeline():
    """
    Creates the machine learning pipeline.

    The model has two types of inputs:
    1. Categorical features, like team names and tournament names
    2. Numeric features, like Elo ratings

    The ColumnTransformer handles both types correctly.
    """

    categorical_features = ["home_team", "away_team", "tournament"]
    numeric_features = ["neutral", "home_elo", "away_elo", "elo_diff"]

    # One-hot encoding turns names like "Argentina" and "France"
    # into numbers the model can understand.
    #
    # StandardScaler puts numeric features on a similar scale.
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", make_encoder(), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    # Poisson regression is useful for count data.
    # Goals are count data: 0, 1, 2, 3, etc.
    model = PoissonRegressor(alpha=0.01, max_iter=1000)

    # Pipeline means:
    # first transform the data,
    # then fit the Poisson model.
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )


class WorldCupPricer:
    """
    Main pricing model.

    It does three big things:

    1. Builds Elo ratings for team strength
    2. Predicts expected goals for each team
    3. Converts expected goals into win/draw/loss probabilities
    """

    def __init__(self, base_rating: float = 1500, home_advantage: float = 50):
        self.base_rating = base_rating
        self.home_advantage = home_advantage

        # We train two separate models:
        # one for home team goals
        # one for away team goals
        self.home_goal_model = make_poisson_pipeline()
        self.away_goal_model = make_poisson_pipeline()

        self.ratings = None
        self.is_fitted = False

    def fit(self, df: pd.DataFrame):
        """
        Trains the model on historical match data.
        """

        # Add Elo ratings to every match before training.
        df_with_elo, ratings = add_elo_features(
            df,
            base_rating=self.base_rating,
            home_advantage=self.home_advantage
        )

        # These are the features the model uses to predict goals.
        features = [
            "home_team",
            "away_team",
            "tournament",
            "neutral",
            "home_elo",
            "away_elo",
            "elo_diff",
        ]

        X = df_with_elo[features]

        # The home model learns to predict home goals.
        y_home = df_with_elo["home_score"]

        # The away model learns to predict away goals.
        y_away = df_with_elo["away_score"]

        self.home_goal_model.fit(X, y_home)
        self.away_goal_model.fit(X, y_away)

        # Save the final Elo ratings so we can price future matches.
        self.ratings = ratings
        self.is_fitted = True

        return self

    def _make_match_row(
        self,
        home_team: str,
        away_team: str,
        tournament: str = "FIFA World Cup",
        neutral: int = 1
    ) -> pd.DataFrame:
        """
        Creates a one-row dataframe for a future match.

        The trained model expects the same columns it saw during training,
        so we build them here.
        """

        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction.")

        # Get each team's current Elo rating.
        # If a team is missing, use the base rating.
        home_elo = self.ratings.get(home_team, self.base_rating)
        away_elo = self.ratings.get(away_team, self.base_rating)

        # Neutral-site games get no home advantage.
        home_boost = 0 if neutral else self.home_advantage

        elo_diff = home_elo + home_boost - away_elo

        return pd.DataFrame(
            {
                "home_team": [home_team],
                "away_team": [away_team],
                "tournament": [tournament],
                "neutral": [neutral],
                "home_elo": [home_elo],
                "away_elo": [away_elo],
                "elo_diff": [elo_diff],
            }
        )

    def predict_match(
        self,
        home_team: str,
        away_team: str,
        tournament: str = "FIFA World Cup",
        neutral: int = 1,
        max_goals: int = 10
    ) -> dict:
        """
        Prices one match.

        It predicts expected goals, then uses a Poisson distribution
        to estimate every possible scoreline.
        """

        match_row = self._make_match_row(
            home_team=home_team,
            away_team=away_team,
            tournament=tournament,
            neutral=neutral
        )

        # These are the expected goals for each team.
        # Example: Argentina 1.22, France 1.09
        lambda_home = float(self.home_goal_model.predict(match_row)[0])
        lambda_away = float(self.away_goal_model.predict(match_row)[0])

        # Prevent impossible or tiny expected goal values.
        lambda_home = max(lambda_home, 0.05)
        lambda_away = max(lambda_away, 0.05)

        # We calculate probabilities for scores from 0 goals to max_goals.
        home_goals = np.arange(0, max_goals + 1)
        away_goals = np.arange(0, max_goals + 1)

        # Poisson probability of each possible goal count.
        # Example: P(home scores 0), P(home scores 1), etc.
        home_probs = poisson.pmf(home_goals, lambda_home)
        away_probs = poisson.pmf(away_goals, lambda_away)

        # Score matrix:
        # rows = home goals
        # columns = away goals
        #
        # Example:
        # score_matrix[2][1] = probability home wins 2-1
        score_matrix = np.outer(home_probs, away_probs)

        # Normalize so all scoreline probabilities add up to 1.
        # This handles the tiny probability beyond max_goals.
        score_matrix = score_matrix / score_matrix.sum()

        # Lower triangle means home goals > away goals.
        # That is a home win.
        home_win_prob = np.tril(score_matrix, -1).sum()

        # Diagonal means home goals = away goals.
        # That is a draw.
        draw_prob = np.trace(score_matrix)

        # Upper triangle means away goals > home goals.
        # That is an away win.
        away_win_prob = np.triu(score_matrix, 1).sum()

        probabilities = {
            "home_win": home_win_prob,
            "draw": draw_prob,
            "away_win": away_win_prob,
        }

        # Fair odds are just 1 divided by probability.
        # Example: 40% probability means fair odds of 2.50.
        fair_odds = {
            outcome: 1 / prob
            for outcome, prob in probabilities.items()
        }

        return {
            "home_team": home_team,
            "away_team": away_team,
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "probabilities": probabilities,
            "fair_odds": fair_odds,
            "score_matrix": score_matrix,
        }