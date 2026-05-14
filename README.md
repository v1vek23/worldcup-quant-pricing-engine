# World Cup Quant Pricing Engine

A Python-based football match pricing engine that estimates fair odds for World Cup matches using historical international results, Elo ratings, Poisson expected-goals modeling, market-implied probabilities, and fractional Kelly bet sizing.

## Project Overview

This project treats football match outcomes like financial contracts.

The model estimates the probability of a home win, draw, and away win, converts those probabilities into fair odds, compares them against market odds, and calculates position sizing using fractional Kelly.

The goal is to demonstrate quantitative finance concepts through sports betting market structure: probability modeling, pricing, edge detection, and risk-controlled position sizing.

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
- Risk-controlled betting strategy design

## Project Structure

```text
WorldCup/
├── data/
│   ├── raw/
│   └── processed/
├── reports/
│   ├── example_output.txt
│   ├── project_summary.md
│   ├── probability_chart.png
│   ├── fair_odds_vs_market.png
│   ├── score_matrix_heatmap.png
│   └── top_teams_elo.png
├── src/
│   ├── data.py
│   ├── elo.py
│   ├── market.py
│   ├── model.py
│   └── visuals.py
├── main.py
├── requirements.txt
├── README.md
└── .gitignore
```

## How the Model Works

### 1. Data Loading

The project downloads international football results and filters matches from a selected start year.

### 2. Elo Ratings

Each team receives a dynamic Elo rating that updates after each match. More important matches and larger goal differences create larger rating changes.

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

Run the default example:

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

## Example Charts

The project saves visual outputs to the `reports/` folder:

- `probability_chart.png`
- `fair_odds_vs_market.png`
- `score_matrix_heatmap.png`
- `top_teams_elo.png`

## Current Limitations

- Market odds are manually entered.
- The model has not yet been backtested against historical odds.
- The model does not currently use player injuries, squad strength, xG event data, or tactical variables.
- The model assumes home and away goals are independent Poisson variables.
- The scoreline model does not yet include Dixon-Coles low-score correction.

## Future Improvements

- Add historical bookmaker odds
- Backtest against previous World Cups
- Add model calibration tests
- Add Brier score and log loss evaluation
- Add Monte Carlo tournament simulation
- Add a Streamlit dashboard
- Add real-time odds comparison
- Add Dixon-Coles score correction
- Add player-level features and squad strength adjustments
