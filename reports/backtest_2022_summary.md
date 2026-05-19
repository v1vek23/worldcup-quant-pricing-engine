# 2022 World Cup Backtest Summary

The model was trained only on matches before the tournament began, then tested on the World Cup matches from that year.

## Summary Metrics

- Matches tested: 64
- Prediction accuracy: 50.00%
- Average Brier score: 0.6165
- Average log loss: 1.0483
- Average probability assigned to actual outcome: 39.82%
- Average model confidence: 52.33%

## Interpretation

Accuracy measures how often the model's most likely outcome happened.

Brier score and log loss measure probability quality. These are more useful than accuracy because the model is trying to estimate probabilities, not just pick winners.

Average probability assigned to the actual outcome shows how much confidence the model placed on what really happened.
