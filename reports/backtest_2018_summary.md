# 2018 World Cup Backtest Summary

The model was trained only on matches before the tournament began, then tested on the World Cup matches from that year.

## Summary Metrics

- Matches tested: 64
- Prediction accuracy: 57.81%
- Average Brier score: 0.5788
- Average log loss: 0.9738
- Average probability assigned to actual outcome: 40.81%
- Average model confidence: 50.57%

## Interpretation

Accuracy measures how often the model's most likely outcome happened.

Brier score and log loss measure probability quality. These are more useful than accuracy because the model is trying to estimate probabilities, not just pick winners.

Average probability assigned to the actual outcome shows how much confidence the model placed on what really happened.
