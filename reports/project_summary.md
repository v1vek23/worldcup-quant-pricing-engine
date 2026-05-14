# Project Summary

This project builds a football match pricing model inspired by quantitative finance.

The model estimates match probabilities using Elo ratings and Poisson regression. These probabilities are converted into fair decimal odds and compared against market odds. If the model probability exceeds the no-vig market-implied probability, the system identifies a positive expected-value opportunity.

The final layer uses fractional Kelly sizing to estimate position size while controlling risk.

## Current Example

The default example prices Argentina vs France.

The model estimates:

- Expected goals for each team
- Home win, draw, and away win probabilities
- Fair odds for each outcome
- Market-implied probabilities
- Positive or negative edge
- Fractional Kelly stake size

## Why This Is Quantitative

This project follows a similar structure to quantitative trading research:

1. Collect historical data
2. Engineer predictive features
3. Estimate probabilities
4. Convert probabilities into prices
5. Compare model prices against market prices
6. Size positions based on expected edge
7. Evaluate risk