import argparse
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import load_results
from src.model import WorldCupPricer
from src.visuals import (
    save_champion_probability_chart,
    save_stage_probability_chart,
)


def load_groups(groups_file: str) -> pd.DataFrame:
    """
    Loads a tournament group file.

    Required columns:
    group, team

    Optional column:
    official_name
    """

    groups = pd.read_csv(groups_file)

    required_columns = {"group", "team"}

    if not required_columns.issubset(groups.columns):
        raise ValueError("Groups file must contain columns: group, team")

    group_sizes = groups.groupby("group")["team"].count()

    if not (group_sizes == 4).all():
        raise ValueError("Each group must have exactly 4 teams.")

    if len(groups) != 48:
        raise ValueError("This simulator expects exactly 48 teams.")

    groups = groups.sort_values(["group", "team"]).reset_index(drop=True)

    return groups


def warn_about_missing_team_ratings(groups: pd.DataFrame, pricer: WorldCupPricer):
    """
    Warns if a team in the tournament file was not found in the historical data.

    If a team is missing, the model will still run, but that team will use
    the default base Elo rating.
    """

    tournament_teams = set(groups["team"])
    rated_teams = set(pricer.ratings.keys())

    missing_teams = sorted(tournament_teams - rated_teams)

    if missing_teams:
        print("\nWarning: These teams were not found in the historical ratings data.")
        print("They will use the default base Elo rating.")
        for team in missing_teams:
            print(f"- {team}")


def simulate_scoreline(prediction: dict, rng: np.random.Generator) -> tuple[int, int]:
    """
    Simulates one exact scoreline using the model's score matrix.

    The score matrix stores probabilities for each possible score.
    Example:
    score_matrix[2][1] = probability of a 2-1 home win.
    """

    score_matrix = prediction["score_matrix"]
    flat_probs = score_matrix.ravel()

    index = rng.choice(len(flat_probs), p=flat_probs)

    home_goals, away_goals = divmod(index, score_matrix.shape[1])

    return int(home_goals), int(away_goals)


def get_cached_prediction(
    pricer: WorldCupPricer,
    team_a: str,
    team_b: str,
    match_cache: dict
) -> dict:
    """
    Gets a model prediction for a matchup.

    The cache prevents us from recalculating the same matchup thousands of times.
    This makes the Monte Carlo simulation much faster.
    """

    key = tuple(sorted([team_a, team_b]))

    if key not in match_cache:
        match_cache[key] = pricer.predict_match(
            home_team=key[0],
            away_team=key[1],
            tournament="FIFA World Cup",
            neutral=1
        )

    return match_cache[key]


def simulate_group_match(
    pricer: WorldCupPricer,
    team_a: str,
    team_b: str,
    rng: np.random.Generator,
    match_cache: dict
) -> dict:
    """
    Simulates a group-stage match.

    Group-stage matches can end in a draw.
    """

    prediction = get_cached_prediction(
        pricer=pricer,
        team_a=team_a,
        team_b=team_b,
        match_cache=match_cache
    )

    home_team = prediction["home_team"]
    away_team = prediction["away_team"]

    home_goals, away_goals = simulate_scoreline(prediction, rng)

    return {
        "home_team": home_team,
        "away_team": away_team,
        "home_goals": home_goals,
        "away_goals": away_goals,
    }


def empty_group_table(teams: list[str]) -> pd.DataFrame:
    """
    Creates an empty standings table for one group.
    """

    return pd.DataFrame(
        {
            "team": teams,
            "points": 0,
            "goals_for": 0,
            "goals_against": 0,
            "goal_diff": 0,
            "wins": 0,
        }
    )


def update_group_table(
    table: pd.DataFrame,
    home_team: str,
    away_team: str,
    home_goals: int,
    away_goals: int
) -> pd.DataFrame:
    """
    Updates group standings after one simulated match.
    """

    table = table.copy()

    home_index = table.index[table["team"] == home_team][0]
    away_index = table.index[table["team"] == away_team][0]

    table.loc[home_index, "goals_for"] += home_goals
    table.loc[home_index, "goals_against"] += away_goals

    table.loc[away_index, "goals_for"] += away_goals
    table.loc[away_index, "goals_against"] += home_goals

    if home_goals > away_goals:
        table.loc[home_index, "points"] += 3
        table.loc[home_index, "wins"] += 1

    elif home_goals < away_goals:
        table.loc[away_index, "points"] += 3
        table.loc[away_index, "wins"] += 1

    else:
        table.loc[home_index, "points"] += 1
        table.loc[away_index, "points"] += 1

    table["goal_diff"] = table["goals_for"] - table["goals_against"]

    return table


def rank_table(table: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Ranks teams inside a group.

    Tiebreakers used here:
    1. points
    2. goal difference
    3. goals for
    4. wins
    5. random tiebreaker

    The random tiebreaker is used because the full FIFA tiebreaking rules
    are more detailed than we need for this version.
    """

    table = table.copy()
    table["random_tiebreaker"] = rng.random(len(table))

    ranked = table.sort_values(
        by=[
            "points",
            "goal_diff",
            "goals_for",
            "wins",
            "random_tiebreaker",
        ],
        ascending=[False, False, False, False, False]
    ).reset_index(drop=True)

    ranked["group_rank"] = ranked.index + 1

    return ranked


def simulate_group_stage(
    groups: pd.DataFrame,
    pricer: WorldCupPricer,
    rng: np.random.Generator,
    match_cache: dict
) -> pd.DataFrame:
    """
    Simulates all group-stage matches.

    Each group has 4 teams.
    Each team plays the other 3 teams once.
    """

    all_ranked_tables = []

    for group_name, group_df in groups.groupby("group"):
        teams = group_df["team"].tolist()

        table = empty_group_table(teams)

        for team_a, team_b in combinations(teams, 2):
            result = simulate_group_match(
                pricer=pricer,
                team_a=team_a,
                team_b=team_b,
                rng=rng,
                match_cache=match_cache
            )

            table = update_group_table(
                table=table,
                home_team=result["home_team"],
                away_team=result["away_team"],
                home_goals=result["home_goals"],
                away_goals=result["away_goals"],
            )

        ranked = rank_table(table, rng)
        ranked["group"] = group_name

        all_ranked_tables.append(ranked)

    return pd.concat(all_ranked_tables, ignore_index=True)


def select_qualified_teams(group_results: pd.DataFrame) -> pd.DataFrame:
    """
    Selects the 32 teams that advance.

    2026-style rule:
    - top 2 from each group
    - 8 best third-place teams
    """

    top_two = group_results[group_results["group_rank"] <= 2].copy()

    third_place = group_results[group_results["group_rank"] == 3].copy()

    best_thirds = third_place.sort_values(
        by=["points", "goal_diff", "goals_for", "wins"],
        ascending=[False, False, False, False]
    ).head(8)

    qualified = pd.concat([top_two, best_thirds], ignore_index=True)

    return qualified


def seed_knockout_teams(
    qualified: pd.DataFrame,
    rng: np.random.Generator
) -> list[str]:
    """
    Creates a simplified seeded knockout bracket.

    This is not the exact official World Cup bracket.
    It ranks qualified teams by group performance and pairs high seeds
    against low seeds.
    """

    qualified = qualified.copy()
    qualified["random_tiebreaker"] = rng.random(len(qualified))

    seeded = qualified.sort_values(
        by=[
            "group_rank",
            "points",
            "goal_diff",
            "goals_for",
            "wins",
            "random_tiebreaker",
        ],
        ascending=[True, False, False, False, False, False]
    ).reset_index(drop=True)

    teams = seeded["team"].tolist()

    bracket = []

    left = 0
    right = len(teams) - 1

    while left < right:
        bracket.append(teams[left])
        bracket.append(teams[right])
        left += 1
        right -= 1

    return bracket


def simulate_knockout_match(
    pricer: WorldCupPricer,
    team_a: str,
    team_b: str,
    rng: np.random.Generator,
    match_cache: dict
) -> str:
    """
    Simulates a knockout match.

    Knockout matches need a winner.
    The model has a draw probability, so we split the draw probability
    evenly between both teams to represent extra time/penalties.
    """

    prediction = get_cached_prediction(
        pricer=pricer,
        team_a=team_a,
        team_b=team_b,
        match_cache=match_cache
    )

    home_team = prediction["home_team"]
    away_team = prediction["away_team"]

    probs = prediction["probabilities"]

    home_advances_prob = probs["home_win"] + 0.5 * probs["draw"]
    away_advances_prob = probs["away_win"] + 0.5 * probs["draw"]

    total = home_advances_prob + away_advances_prob

    home_advances_prob = home_advances_prob / total
    away_advances_prob = away_advances_prob / total

    winner = rng.choice(
        [home_team, away_team],
        p=[home_advances_prob, away_advances_prob]
    )

    return winner


def simulate_knockout_round(
    teams: list[str],
    pricer: WorldCupPricer,
    rng: np.random.Generator,
    match_cache: dict
) -> list[str]:
    """
    Simulates one knockout round.
    """

    winners = []

    for i in range(0, len(teams), 2):
        team_a = teams[i]
        team_b = teams[i + 1]

        winner = simulate_knockout_match(
            pricer=pricer,
            team_a=team_a,
            team_b=team_b,
            rng=rng,
            match_cache=match_cache
        )

        winners.append(winner)

    return winners


def initialize_counts(groups: pd.DataFrame) -> dict:
    """
    Creates stage counters for every team.
    """

    teams = groups["team"].tolist()

    counts = {}

    for team in teams:
        counts[team] = {
            "team": team,
            "group": groups.loc[groups["team"] == team, "group"].iloc[0],
            "round_32": 0,
            "round_16": 0,
            "quarterfinal": 0,
            "semifinal": 0,
            "final": 0,
            "champion": 0,
        }

    return counts


def run_tournament_simulation(
    groups_file: str = "data/processed/official_2026_groups.csv",
    n_sims: int = 1000,
    start_year: int = 2000,
    seed: int = 7,
    output_dir: str = "reports"
) -> pd.DataFrame:
    """
    Runs a Monte Carlo tournament simulation.

    Each simulation includes:
    1. group stage
    2. Round of 32
    3. Round of 16
    4. quarterfinals
    5. semifinals
    6. final
    """

    rng = np.random.default_rng(seed)

    groups = load_groups(groups_file)

    print("Loading data...")
    df = load_results(start_year=start_year)

    print("Training World Cup pricing model...")
    pricer = WorldCupPricer()
    pricer.fit(df)

    warn_about_missing_team_ratings(groups, pricer)

    counts = initialize_counts(groups)

    match_cache = {}

    print(f"Running {n_sims:,} tournament simulations...")

    for sim in range(n_sims):
        group_results = simulate_group_stage(
            groups=groups,
            pricer=pricer,
            rng=rng,
            match_cache=match_cache
        )

        qualified = select_qualified_teams(group_results)

        for team in qualified["team"]:
            counts[team]["round_32"] += 1

        knockout_teams = seed_knockout_teams(qualified, rng)

        round_16 = simulate_knockout_round(
            teams=knockout_teams,
            pricer=pricer,
            rng=rng,
            match_cache=match_cache
        )

        for team in round_16:
            counts[team]["round_16"] += 1

        quarterfinalists = simulate_knockout_round(
            teams=round_16,
            pricer=pricer,
            rng=rng,
            match_cache=match_cache
        )

        for team in quarterfinalists:
            counts[team]["quarterfinal"] += 1

        semifinalists = simulate_knockout_round(
            teams=quarterfinalists,
            pricer=pricer,
            rng=rng,
            match_cache=match_cache
        )

        for team in semifinalists:
            counts[team]["semifinal"] += 1

        finalists = simulate_knockout_round(
            teams=semifinalists,
            pricer=pricer,
            rng=rng,
            match_cache=match_cache
        )

        for team in finalists:
            counts[team]["final"] += 1

        champion = simulate_knockout_round(
            teams=finalists,
            pricer=pricer,
            rng=rng,
            match_cache=match_cache
        )[0]

        counts[champion]["champion"] += 1

        if (sim + 1) % max(1, n_sims // 10) == 0:
            print(f"Completed {sim + 1:,} / {n_sims:,} simulations")

    summary = make_simulation_summary(counts, n_sims)

    save_simulation_outputs(
        summary=summary,
        n_sims=n_sims,
        groups_file=groups_file,
        output_dir=output_dir
    )

    return summary


def make_simulation_summary(counts: dict, n_sims: int) -> pd.DataFrame:
    """
    Converts raw simulation counts into probabilities.
    """

    rows = []

    for team, values in counts.items():
        row = {
            "team": team,
            "group": values["group"],
            "round_32_prob": values["round_32"] / n_sims,
            "round_16_prob": values["round_16"] / n_sims,
            "quarterfinal_prob": values["quarterfinal"] / n_sims,
            "semifinal_prob": values["semifinal"] / n_sims,
            "final_prob": values["final"] / n_sims,
            "champion_prob": values["champion"] / n_sims,
        }

        rows.append(row)

    summary = pd.DataFrame(rows)

    summary = summary.sort_values(
        "champion_prob",
        ascending=False
    ).reset_index(drop=True)

    return summary


def save_simulation_outputs(
    summary: pd.DataFrame,
    n_sims: int,
    groups_file: str,
    output_dir: str = "reports"
):
    """
    Saves tournament simulation outputs.
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_path = Path(output_dir) / "tournament_simulation_summary.csv"
    md_path = Path(output_dir) / "tournament_simulation_summary.md"

    champion_chart_path = Path(output_dir) / "tournament_champion_probabilities.png"
    stage_chart_path = Path(output_dir) / "tournament_stage_probabilities.png"

    summary.to_csv(csv_path, index=False)

    save_champion_probability_chart(
        simulation_summary=summary,
        output_path=str(champion_chart_path)
    )

    save_stage_probability_chart(
        simulation_summary=summary,
        output_path=str(stage_chart_path)
    )

    top_10 = summary.head(10)

    with open(md_path, "w") as file:
        file.write("# Tournament Simulation Summary\n\n")

        file.write(
            f"This report summarizes {n_sims:,} Monte Carlo simulations "
            "of a 48-team World Cup-style tournament.\n\n"
        )

        file.write(f"Groups file used: `{groups_file}`\n\n")

        file.write("## Top 10 Champion Probabilities\n\n")

        for row in top_10.itertuples(index=False):
            file.write(
                f"- {row.team}: {row.champion_prob:.2%} champion probability, "
                f"{row.final_prob:.2%} final probability, "
                f"{row.semifinal_prob:.2%} semifinal probability\n"
            )

        file.write("\n## Notes\n\n")
        file.write(
            "The simulation uses the model's match probabilities to simulate group-stage "
            "matches and knockout advancement.\n\n"
        )
        file.write(
            "The group file uses the finalized 2026 World Cup group composition.\n\n"
        )
        file.write(
            "The knockout bracket is a simplified seeded bracket based on group-stage performance. "
            "It is not the exact official World Cup bracket mapping.\n"
        )

    print(f"\nSaved simulation summary to {csv_path}")
    print(f"Saved markdown summary to {md_path}")
    print(f"Saved charts to {output_dir}/")


def print_top_teams(summary: pd.DataFrame, top_n: int = 10):
    """
    Prints the top teams by champion probability.
    """

    print("\n==============================")
    print("TOURNAMENT SIMULATION RESULTS")
    print("==============================")

    columns = [
        "team",
        "round_32_prob",
        "round_16_prob",
        "quarterfinal_prob",
        "semifinal_prob",
        "final_prob",
        "champion_prob",
    ]

    print(summary[columns].head(top_n).to_string(index=False))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Monte Carlo tournament simulation for the World Cup Quant Pricing Engine"
    )

    parser.add_argument(
        "--groups-file",
        type=str,
        default="data/processed/official_2026_groups.csv",
        help="CSV file containing group and team columns"
    )

    parser.add_argument(
        "--n-sims",
        type=int,
        default=1000,
        help="Number of tournament simulations"
    )

    parser.add_argument(
        "--start-year",
        type=int,
        default=2000,
        help="First year of historical data to use"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducible simulations"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    summary = run_tournament_simulation(
        groups_file=args.groups_file,
        n_sims=args.n_sims,
        start_year=args.start_year,
        seed=args.seed
    )

    print_top_teams(summary)


if __name__ == "__main__":
    main()