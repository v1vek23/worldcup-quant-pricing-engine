from pathlib import Path
import pandas as pd

# This is the public dataset we are using for international football results.
RAW_RESULTS_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def load_results(
    raw_path: str = "data/raw/results.csv",
    start_year: int = 2000
) -> pd.DataFrame:
    """
    Loads international football results.

    If the CSV file is not already saved on your computer,
    this function downloads it from GitHub.

    Then it cleans the data so the model can use it.
    """

    # Path lets Python work with file locations cleanly.
    path = Path(raw_path)

    # Make sure the folder data/raw exists before saving anything there.
    path.parent.mkdir(parents=True, exist_ok=True)

    # If results.csv is not already downloaded, download it.
    if not path.exists():
        print("Downloading international results data...")
        df = pd.read_csv(RAW_RESULTS_URL)
        df.to_csv(path, index=False)

    # If the file already exists, just load it from your computer.
    else:
        print("Loading local results data...")
        df = pd.read_csv(path)

    # Convert the date column into real datetime objects.
    # This lets us sort by date and filter by year.
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Drop rows that are missing important information.
    # The model needs teams, scores, tournament type, and neutral-site info.
    df = df.dropna(
        subset=[
            "date",
            "home_team",
            "away_team",
            "home_score",
            "away_score",
            "tournament",
            "neutral",
        ]
    )

    # Scores should be integers because they are goal counts.
    df["home_score"] = df["home_score"].astype(int)
    df["away_score"] = df["away_score"].astype(int)

    # Neutral is stored as True/False in the data.
    # We convert it to 1/0 so the model can use it numerically.
    df["neutral"] = df["neutral"].astype(int)

    # Only keep matches from start_year onward.
    # Older football data may be less useful for predicting modern teams.
    df = df[df["date"].dt.year >= start_year].copy()

    # Sort matches by date because Elo ratings must update chronologically.
    df = df.sort_values("date").reset_index(drop=True)

    return df