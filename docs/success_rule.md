# Success label (version 0)

This is the rule that defines the `success` column of `data/transfers_dataset.csv`, built by `src/build_dataset.py`.
It is a first draft, to be refined. The model predicts this label, so any change to the rule changes what the score means.

## Which transfers are kept

- Permanent moves between two clubs of the five leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1). Loans and returns from loan are excluded.
- Transfer windows: see `TRANSFER_WINDOWS` in `src/build_dataset.py`. Currently summer 2023 only (season 2022 before, season 2023 after).
- Players need at least **900 minutes** in the season before and in the season after the move (about 10 full matches). Players with fewer minutes are dropped, because their stats say little.
- The position must be known from the season before the move.

## Stats used

- Minutes, goals and assists per season, summed over **club** games only. National-team games are excluded (a club entry is any competition whose team is not the player's nationality).
- Season `N` in the API means the season starting in year `N` (season 2023 = 2023-24).

## The rule

A player is labelled `success = 1` when both conditions hold:

1. **Playing time is kept:** minutes after the move ≥ 50% of minutes before the move.
2. **Production is kept:** if the player had goals or assists per 90 minutes before the move, his goals + assists per 90 minutes after the move are ≥ 75% of that value. If he had none before the move, this condition does not apply.

Otherwise `success = 0`.

## Known limits

- Injuries, loss of form, tactical role and team quality are not separated from adaptation. A long injury counts as a failure.
- Condition 2 is not meaningful for defenders and goalkeepers, who rarely score or assist.
- The thresholds (50%, 75%, 900 minutes) are conventions, not calibrated against any outcome.
- Players who moved at the end of a season or in January are not in the dataset (winter windows are excluded for now).

## Open questions for later

- Should position matter (for example, a different production target for defenders)?
- Should the comparison use the whole season or the first half-season after the move?
- Should the thresholds be chosen before looking at model results (the current choice is made before any model is trained on the real data)?
