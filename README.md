# World Cup Quant Pricing Engine

A Python-based football match pricing engine that estimates fair odds for World Cup matches using historical international results, Elo ratings, Poisson expected-goals modeling, market-implied probabilities, historical backtesting, calibration analysis, Monte Carlo tournament simulation, and fractional Kelly bet sizing.

## Project Overview

This project treats football match outcomes like financial contracts.

The model estimates the probability of a home win, draw, and away win, converts those probabilities into fair odds, compares them against market odds, and calculates position sizing using fractional Kelly.

The project also supports historical World Cup backtesting and full tournament simulation. The tournament simulator uses a finalized 2026 World Cup group file and estimates each team's probability of reaching different stages of the tournament.

The goal is to demonstrate quantitative finance concepts through sports betting market structure: probability modeling, pricing, edge detection, backtesting, calibration analysis, simulation, and risk-controlled position sizing.

## Example Output

Default matchup:

```text
Argentina vs France

Expected Goals:
Argentina: 1.223
France: 1.093

Model Probabilities:
home_win: 39.113%
draw: 28.175%
away_win: 32.712%

Fair Odds:
home_win: 2.56
draw: 3.55
away_win: 3.06
```

Example market comparison:

```text
outcome    model_prob    market_prob_no_vig    edge       market_odds    fair_odds    kelly_fraction    suggested_stake
home_win   0.391130      0.360418              0.030712   2.65           2.556693     0.005530          5.529556
away_win   0.327119      0.341110             -0.013991   2.80           3.056987     0.000000          0.000000
draw       0.281750      0.298471             -0.016721   3.20           3.549242     0.000000          0.000000
```

## Quant Concepts Used

- Elo ratings
- Poisson regression
- Expected goals modeling
- Fair odds pricing
- Market-implied probabilities
- No-vig odds conversion
- Expected value
- Kelly criterion
- Fractional Kelly position sizing
- Historical backtesting
- Brier score
- Log loss
- Calibration tables
- Expected calibration error
- Confidence distribution analysis
- Outcome-level performance analysis
- Monte Carlo simulation
- Tournament path simulation
- Risk-controlled betting strategy design

## Project Structure

```text
WorldCup/
├── data/
│   ├── raw/
│   └── processed/
│       └── official_2026_groups.csv
├── reports/
│   ├── example_output.txt
│   ├── project_summary.md
│   ├── probability_chart.png
│   ├── fair_odds_vs_market.png
│   ├── score_matrix_heatmap.png
│   ├── top_teams_elo.png
│   ├── backtest_2022_world_cup.csv
│   ├── backtest_2022_summary.md
│   ├── backtest_2022_calibration.csv
│   ├── backtest_2022_outcome_metrics.csv
│   ├── backtest_2022_calibration_plot.png
│   ├── backtest_2022_confidence_histogram.png
│   ├── backtest_2022_outcome_metrics.png
│   ├── tournament_simulation_summary.csv
│   ├── tournament_simulation_summary.md
│   ├── tournament_champion_probabilities.png
│   └── tournament_stage_probabilities.png
├── src/
│   ├── data.py
│   ├── elo.py
│   ├── market.py
│   ├── model.py
│   ├── visuals.py
│   ├── evaluation.py
│   ├── backtest.py
│   └── simulation.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Official 2026 Group File

The tournament simulator uses:

```text
data/processed/official_2026_groups.csv
```

The file contains the finalized 2026 World Cup groups, with both a model-compatible team name and an official display name.

Example:

```csv
group,team,official_name
A,Mexico,Mexico
A,South Korea,Korea Republic
A,Czech Republic,Czechia
D,United States,USA
D,Turkey,Türkiye
E,Ivory Coast,Côte d’Ivoire
G,Iran,IR Iran
H,Cape Verde,Cabo Verde
K,DR Congo,Congo DR
```

The model uses the `team` column because those names better match the historical international football results dataset.

## How the Model Works

### 1. Data Loading

The project downloads historical international football results and filters matches from a selected start year.

### 2. Elo Ratings

Each team receives a dynamic Elo rating that updates after each match. More important matches and larger goal differences create larger rating changes.

The Elo features are calculated chronologically so the model only uses information that would have been available before each match.

### 3. Expected Goals Modeling

The project trains two Poisson regression models:

- one model for home team goals
- one model for away team goals

The models use team names, tournament type, neutral-site status, and Elo features.

### 4. Match Probability Pricing

The model estimates expected goals for both teams, then uses Poisson distributions to calculate the probability of each possible scoreline.

Those scoreline probabilities are aggregated into:

- home win probability
- draw probability
- away win probability

### 5. Fair Odds

The model converts probabilities into fair decimal odds:

```text
fair odds = 1 / probability
```

### 6. Market Comparison

Bookmaker odds are converted into no-vig implied probabilities. The model compares its own probability to the market probability.

```text
edge = model probability - market probability
```

A positive edge suggests the model believes the market has underpriced that outcome.

### 7. Kelly Sizing

The project uses fractional Kelly sizing to estimate how much of a bankroll to allocate to a positive-edge opportunity.

### 8. Historical Backtesting

The project can backtest the model on past World Cups.

For a selected tournament year, the model trains only on matches before that World Cup begins. It then predicts each match in that tournament and evaluates the quality of the predicted probabilities.

The backtester measures:

- prediction accuracy
- Brier score
- log loss
- average model confidence
- average probability assigned to the actual outcome
- expected calibration error

### 9. Calibration Analysis

The project creates calibration tables and plots to measure whether the model's confidence matches reality.

For example, if the model is around 60% confident, it should be correct around 60% of the time. If it is much less accurate than its confidence level, the model is overconfident.

The calibration analysis includes:

- confidence bins
- average confidence per bin
- actual accuracy per bin
- calibration error per bin
- expected calibration error
- confidence distribution charts
- outcome-level performance metrics

### 10. Monte Carlo Tournament Simulation

The project can simulate a full 48-team World Cup-style tournament many times.

Each simulation includes:

- group stage
- Round of 32
- Round of 16
- quarterfinals
- semifinals
- final
- champion

The simulator uses the match pricing model to simulate group-stage scorelines and knockout advancement. After many simulations, it estimates each team's probability of reaching each stage.

The tournament file is based on the finalized 2026 groups, but the knockout bracket is still simplified.

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the default match-pricing example:

```bash
python main.py
```

Run a custom matchup:

```bash
python main.py --home Brazil --away England --home-odds 2.40 --draw-odds 3.10 --away-odds 3.00
```

Run without charts:

```bash
python main.py --no-charts
```

## Backtesting

Run a historical World Cup backtest:

```bash
python -m src.backtest --year 2022
```

Run another tournament:

```bash
python -m src.backtest --year 2018
```

The backtester trains only on matches before the selected World Cup begins, then evaluates the model on that tournament's matches.

Backtest reports are saved to the `reports/` folder:

```text
reports/backtest_2022_world_cup.csv
reports/backtest_2022_summary.md
reports/backtest_2022_calibration.csv
reports/backtest_2022_outcome_metrics.csv
reports/backtest_2022_calibration_plot.png
reports/backtest_2022_confidence_histogram.png
reports/backtest_2022_outcome_metrics.png
```

## Tournament Simulation

Run a small tournament simulation using the finalized 2026 groups:

```bash
python -m src.simulation --n-sims 100
```

Run a larger simulation:

```bash
python -m src.simulation --n-sims 1000
```

Run with an explicit groups file:

```bash
python -m src.simulation --groups-file data/processed/official_2026_groups.csv --n-sims 1000
```

Tournament simulation reports are saved to the `reports/` folder:

```text
reports/tournament_simulation_summary.csv
reports/tournament_simulation_summary.md
reports/tournament_champion_probabilities.png
reports/tournament_stage_probabilities.png
```

## Example Charts

The project saves visual outputs to the `reports/` folder:

- `probability_chart.png`
- `fair_odds_vs_market.png`
- `score_matrix_heatmap.png`
- `top_teams_elo.png`
- `backtest_2022_calibration_plot.png`
- `backtest_2022_confidence_histogram.png`
- `backtest_2022_outcome_metrics.png`
- `tournament_champion_probabilities.png`
- `tournament_stage_probabilities.png`

These charts help visualize model probabilities, fair odds, scoreline distributions, team strength, model calibration, confidence levels, outcome-level performance, champion probabilities, and stage reach probabilities.

## Current Features

- Downloads and cleans international football results
- Builds chronological Elo ratings for national teams
- Trains Poisson regression models for expected goals
- Converts expected goals into scoreline probabilities
- Converts match probabilities into fair decimal odds
- Compares model probabilities to no-vig market probabilities
- Identifies possible positive expected-value opportunities
- Applies fractional Kelly sizing for risk-controlled position sizing
- Supports command-line match pricing
- Saves visual reports
- Backtests historical World Cup tournaments using probability-quality metrics
- Builds calibration tables
- Calculates expected calibration error
- Creates confidence distribution charts
- Creates outcome-level performance reports
- Simulates the finalized 2026 World Cup group setup
- Estimates champion, finalist, semifinal, quarterfinal, Round of 16, and Round of 32 probabilities

## Current Limitations

- Market odds are manually entered.
- The model has historical probability backtesting, but does not yet backtest profit/loss against real bookmaker odds.
- The model does not currently use player injuries, squad strength, xG event data, or tactical variables.
- The model assumes home and away goals are independent Poisson variables.
- The scoreline model does not yet include Dixon-Coles low-score correction.
- Knockout-stage extra time and penalty shootout outcomes are approximated by splitting draw probability evenly between both teams.
- The tournament simulator uses the finalized group composition, but still uses a simplified seeded knockout bracket rather than the exact official World Cup bracket mapping.
- Calibration results are based on a small number of World Cup matches, so the estimates can be noisy.
- Some team names are normalized to match the historical dataset.

## Future Improvements

- Add exact official World Cup knockout bracket logic
- Add real bookmaker odds for profit/loss backtesting
- Measure ROI, drawdown, hit rate, and strategy returns
- Add a Streamlit dashboard
- Add real-time odds comparison
- Add Dixon-Coles score correction
- Add player-level features and squad strength adjustments
- Add automated model comparison between Elo-only, Poisson, and hybrid models
- Add rolling-window validation across multiple international tournaments
- Add sensitivity analysis for simulation results across different random seeds