# LigueFit Index - League Adaptability Predictor

> *"A prototype that estimates how a player might adapt when moving between football leagues"*

**LigueFit Index** is a Flask web app with a small scikit-learn / XGBoost pipeline. Given a player's season statistics and a league transfer (for example Bundesliga → Premier League), it returns an adaptability score (0–100 %), a risk level and a short recommendation.

> ⚠️ **Status: prototype, not yet evaluated.** The training data is being collected from API-Football and is still incomplete. The current model was trained on a few dozen transfers at most, so its metrics are not meaningful. Do not use the scores for real recruitment decisions.

## Features

- **Web dashboard** (`/`): enter a player's stats and a transfer, get a score and risk level
- **JSON API** (`/predict`): GET or POST, with input validation and clear error messages
- **Data collection** (`src/build_dataset.py`): builds the transfer dataset from API-Football, resumable and rate-limited
- **Training pipeline** (`train_model.py`): feature engineering, model comparison, saved artifacts and a text report
- **Models compared**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost

## Quick Start

### 1. Installation

```bash
git clone https://github.com/clement-ghn/league-adaptability-index.git
cd league-adaptability-index

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Build the dataset

The dataset is built from [API-Football](https://www.api-football.com/) and is not stored in the repository (`data/*.csv` and `data/raw/` are git-ignored). You need an API key:

```bash
# .env at the project root
API_FOOTBALL_KEY=your_key_here
```

```bash
python src/build_dataset.py
```

The free plan allows 100 requests per day and about 10 per minute. The script respects both limits and stops before the daily quota is used up. Every API response is cached in `data/raw/`, so running the script again only fetches what is missing. Run it daily until it prints `Dataset written` with the expected number of rows.

### 3. Train the model

```bash
python train_model.py
```

This writes `models/adaptability_model.joblib`, `models/feature_engineer.joblib`, `models/model_report.txt` and the plots in `static/`. It stops if the dataset is empty, and warns if it has fewer than 100 transfers.

### 4. Run the web app

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

The app loads the saved model at startup. If it is missing, or was trained with an older pipeline, `/predict` returns an error until you run `train_model.py` again.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `API_FOOTBALL_KEY` | *(unset)* | API-Football key, read from `.env`. Only for `src/build_dataset.py` |
| `FLASK_HOST` | `127.0.0.1` | Interface to bind. Set `0.0.0.0` only if you need network access |
| `FLASK_PORT` | `5000` | Port |
| `FLASK_DEBUG` | `0` | Set to `1` to enable the Flask debugger (development only) |
| `CORS_ORIGINS` | *(unset)* | Comma-separated origins allowed to call `/predict` from a browser |
| `OPTA_API_KEY` | *(unset)* | Only for `update_opta_data.py` (see [Data](#data)) |

## Data

**Source:** API-Football (v3), via `src/api_football.py` and `src/build_dataset.py`.

**What is collected:**

1. Team lists for the five leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1), season 2023.
2. The full transfer history of each team. Only permanent moves between two of these leagues are kept (no loans).
3. Each player's club stats for the season before the move and the season after it (national-team games are excluded).

**Current scope:** transfers in the summer 2023 window (season 2022 before, season 2023 after). Other windows can be added in `TRANSFER_WINDOWS` in `src/build_dataset.py`, at the cost of more API requests.

**Filters:** a player needs at least 900 minutes both before and after the move.

**Label (`success`):** defined in [`docs/success_rule.md`](docs/success_rule.md). It is a first version, to be refined before the model results are interpreted. Any change to the rule changes what the score means.

**Not available from this source:** expected goals (xG / xA) and transfer fees. The features that used them were removed.

## How It Works

1. **Data** (`src/data_processor.py`): loads the transfer dataset, cleans it, computes per-90 statistics and league-difficulty features.
2. **Feature engineering** (`src/feature_engineer.py`): position encoding, league-jump features (intensity, pace, physicality, pressing), age and performance interactions, scaling. Inputs are pre-transfer statistics, age, position and leagues.
3. **Model** (`src/ml_model.py`): trains the four models above, picks the best by F1, saves it.
4. **Prediction** (`app.py`): validates the request, applies the same feature pipeline, returns `P(success)` as the adaptability score.

### Risk levels and recommendations

These thresholds are fixed rules in `app.py`, not calibrated against outcomes:

| Score | Risk level |
|---|---|
| ≥ 70 | Low Risk |
| 50 – 69.99 | Medium Risk |
| < 50 | High Risk |

### League characteristics

Each league has five characteristics (intensity, pace, technical, physicality, pressing) on a 0–1 scale. The values are **estimates** from general football knowledge, not measured data. They are used to compute the "jump" between two leagues.

`data/opta_league_characteristics.json` overrides these when present and less than 30 days old. In the repository it is older than that, so the estimates are used. Note that its `source` field says "Opta Sports API", but its values come from the hard-coded estimates in `update_opta_data.py` (the branch used when no key is set).

## Model Results

**No valid results yet.** `models/model_report.txt` is regenerated on every training run, but while the dataset has only a few dozen transfers, the held-out set has a handful of players and one prediction changes every metric. Results will be documented here once the dataset is complete.

Metrics to look at, once the dataset is larger:

- Repeated cross-validation, not a single small split
- ROC-AUC (above 0.5 means the score ranks players better than chance)
- A baseline that always predicts the majority class

## API

### `GET /predict` or `POST /predict`

| Parameter | Type | Rule |
|---|---|---|
| `player_name` | string | optional, echoed back |
| `from_league` | string | one of the five leagues from `/api/leagues` |
| `to_league` | string | one of the five leagues from `/api/leagues` |
| `position` | string | one of `Goalkeeper`, `Defender`, `Midfielder`, `Forward` (from `/api/positions`) |
| `age` | number | 15 – 45 |
| `minutes` | number | > 0 |
| `goals`, `assists` | number | ≥ 0 |

Example:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"player_name": "Example profile", "age": 24, "from_league": "Bundesliga",
       "to_league": "Premier League", "position": "Midfielder", "minutes": 2000,
       "goals": 5, "assists": 4}'
```

Response shape (values from a model trained on a few dozen transfers, they will change after retraining):

```json
{
    "player_name": "Example profile",
    "adaptability_score": 38.89,
    "success_probability": 0.389,
    "risk_level": "High Risk",
    "league_transition": "Bundesliga → Premier League",
    "recommendation": "High risk transfer. Requires careful evaluation and strong support system.",
    "feature_importance": { "...": "top 5 features of the model" }
}
```

`feature_importance` is the same for every player: it describes the model overall, not why this player got this score.

Errors return JSON: `400` for invalid input, `503` when the model is not trained yet, `404` / `500` otherwise.

### Other endpoints

- `GET /api/leagues`: the five supported leagues
- `GET /api/positions`: the four supported positions

## Project Structure

```
league-adaptability-index/
├── app.py                        # Flask app: dashboard and /predict API
├── train_model.py                # Training pipeline (run after the dataset is built)
├── update_opta_data.py           # Refresh league characteristics from Opta (needs API key)
├── requirements.txt              # Python dependencies
├── src/
│   ├── api_football.py           # API-Football client with on-disk cache and rate limiting
│   ├── build_dataset.py          # Builds data/transfers_dataset.csv (resumable)
│   ├── data_processor.py         # Loads the dataset, cleaning, per-90 metrics
│   ├── feature_engineer.py       # Feature pipeline (fitted, saved with the model)
│   ├── ml_model.py               # Model training, comparison, saving, reporting
│   └── opta_integration.py       # Opta API client (optional)
├── utils/
│   └── real_data_criteria.py     # Example of league metrics computed from real stats (not wired in)
├── data/
│   ├── transfers_dataset.csv     # Built by src/build_dataset.py (git-ignored)
│   ├── raw/                      # API-Football response cache (git-ignored)
│   └── opta_league_characteristics.json
├── models/
│   └── model_report.txt          # Output of the last training run
├── static/                       # Plots written by train_model.py
├── templates/
│   ├── index.html                # Dashboard
│   ├── 404.html
│   └── 500.html
├── docs/
│   ├── success_rule.md           # Definition of the success label
│   └── league_criteria_sources.md
└── notebooks/
    └── liguefit_index_exploration.ipynb   # Exploration (not updated; needs: pip install jupyter)
```

## Limitations

- Small training set so far, and the success label is a rule of thumb (see [`docs/success_rule.md`](docs/success_rule.md)).
- Only the summer 2023 transfer window is used for now.
- Injuries, tactical role and team quality are not modelled, so they are mixed into the label.
- League characteristics are estimates.
- The model predicts a binary label. The "adaptability score" is its probability output and is not calibrated.
- Feature importance is global only.
- The free API-Football plan limits how fast the dataset can grow.

## Roadmap

- [x] Feature pipeline on API-Football data
- [x] Resumable, rate-limited data collection
- [x] Flask dashboard and JSON API with input validation
- [ ] Complete the summer 2023 collection and retrain
- [ ] More transfer windows (summer 2022, then winter windows)
- [ ] Documented, refined success label, reviewed before reading results
- [ ] Calibrated probabilities and proper evaluation (repeated CV, more players)
- [ ] Per-player explanations (for example SHAP values)
- [ ] Automated tests for the feature pipeline and API
- [ ] Real league characteristics from match data

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Open a Pull Request

## License

Not yet defined. The repository does not include a `LICENSE` file, so all rights are reserved by default until one is added.

## Contact

**Clement GHN** - [@clement-ghn](https://github.com/clement-ghn)
