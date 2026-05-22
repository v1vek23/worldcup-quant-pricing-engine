# Tournament Simulation Summary

This report summarizes 1,000 Monte Carlo simulations of a 48-team World Cup-style tournament.

Groups file used: `data/processed/official_2026_groups.csv`

## Top 10 Champion Probabilities

- Spain: 23.60% champion probability, 32.40% final probability, 43.30% semifinal probability
- Argentina: 13.00% champion probability, 21.10% final probability, 31.90% semifinal probability
- France: 9.40% champion probability, 16.60% final probability, 26.80% semifinal probability
- England: 6.00% champion probability, 11.50% final probability, 22.00% semifinal probability
- Brazil: 5.40% champion probability, 9.80% final probability, 18.90% semifinal probability
- Ecuador: 5.00% champion probability, 10.90% final probability, 22.00% semifinal probability
- Japan: 4.90% champion probability, 10.20% final probability, 18.80% semifinal probability
- Colombia: 4.10% champion probability, 7.80% final probability, 16.40% semifinal probability
- Morocco: 3.20% champion probability, 7.10% final probability, 15.00% semifinal probability
- Australia: 2.60% champion probability, 5.70% final probability, 13.40% semifinal probability

## Notes

The simulation uses the model's match probabilities to simulate group-stage matches and knockout advancement.

The group file uses the finalized 2026 World Cup group composition.

The knockout bracket is a simplified seeded bracket based on group-stage performance. It is not the exact official World Cup bracket mapping.
