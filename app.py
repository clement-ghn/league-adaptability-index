"""
LigueFit Index - League Adaptability Index
Main Flask application for predicting player success in new leagues
"""

import os
import sys

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import joblib

# Same import style as train_model.py: the saved FeatureEngineer pickle refers to the
# top-level `feature_engineer` module, so it must be importable under that name.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from ml_model import AdaptabilityModel
from feature_engineer import FeatureEngineer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'adaptability_model.joblib')
FEATURE_ENGINEER_PATH = os.path.join(BASE_DIR, 'models', 'feature_engineer.joblib')

# Only the leagues the model was trained on (see src/build_dataset.py)
LEAGUES = [
    'Premier League',
    'La Liga',
    'Serie A',
    'Bundesliga',
    'Ligue 1'
]

# Same positions as the training data (the API does not distinguish full-backs or wingers)
POSITIONS = [
    'Goalkeeper',
    'Defender',
    'Midfielder',
    'Forward'
]

app = Flask(__name__)

# CORS is off unless CORS_ORIGINS is set (e.g. "https://example.com,https://other.com")
cors_origins = os.getenv('CORS_ORIGINS')
if cors_origins:
    CORS(app, resources={r'/predict': {'origins': cors_origins.split(',')}})

model = None
feature_engineer = None


def load_model():
    """Load the trained model and feature engineer from disk. Returns True on success."""
    global model, feature_engineer

    if not (os.path.exists(MODEL_PATH) and os.path.exists(FEATURE_ENGINEER_PATH)):
        return False

    try:
        model_data = joblib.load(MODEL_PATH)
        model = AdaptabilityModel()
        model.best_model = model_data['model']
        model.best_model_name = model_data['model_name']
        model.feature_importance = model_data['feature_importance']
        model.metrics = model_data['metrics']
        feature_engineer = FeatureEngineer.load(FEATURE_ENGINEER_PATH)
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        model = None
        feature_engineer = None
        return False


def parse_player_input(data):
    """Validate and convert the raw request data. Raises ValueError with a readable message."""
    def number(key, default, minimum, maximum=None):
        try:
            value = float(data.get(key, default))
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a number")
        if value < minimum or (maximum is not None and value > maximum):
            upper = f" and <= {maximum}" if maximum is not None else ""
            raise ValueError(f"'{key}' must be >= {minimum}{upper}")
        return value

    features = {
        'age': number('age', 25, 15, 45),
        'minutes': number('minutes', 1000, 1),
        'goals': number('goals', 0, 0),
        'assists': number('assists', 0, 0),
        'from_league': data.get('from_league', 'Ligue 1'),
        'to_league': data.get('to_league', 'Premier League'),
        'position': data.get('position', 'Forward')
    }

    if features['from_league'] not in LEAGUES:
        raise ValueError(f"'from_league' must be one of: {', '.join(LEAGUES)}")
    if features['to_league'] not in LEAGUES:
        raise ValueError(f"'to_league' must be one of: {', '.join(LEAGUES)}")
    if features['position'] not in POSITIONS:
        raise ValueError(f"'position' must be one of: {', '.join(POSITIONS)}")

    return features


@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Predict adaptability for a player

    Parameters (JSON body for POST, query string for GET):
    - player_name: Name of the player (optional)
    - from_league: Current league
    - to_league: Target league
    - age: Player age (15-45)
    - position: Player position
    - minutes: Minutes played (> 0)
    - goals, assists: Season statistics (>= 0)
    """
    if model is None and not load_model():
        return jsonify({
            'error': 'Model not loaded. Run "python train_model.py" first.'
        }), 503

    try:
        if request.method == 'POST':
            data = request.get_json(silent=True)
            if data is None:
                return jsonify({'error': 'Request body must be JSON'}), 400
        else:
            data = request.args.to_dict()

        features = parse_player_input(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    try:
        processed_features = feature_engineer.process_single_player(features)
        prediction, probability = model.predict_adaptability(processed_features)
        adaptability_score = probability * 100

        result = {
            'player_name': data.get('player_name', 'Unknown Player'),
            'adaptability_score': float(round(adaptability_score, 2)),
            'success_probability': float(round(probability, 3)),
            'risk_level': get_risk_level(adaptability_score),
            'feature_importance': get_feature_importance(),
            'league_transition': f"{features['from_league']} → {features['to_league']}",
            'recommendation': get_recommendation(adaptability_score)
        }

        return jsonify(result)

    except Exception as e:
        print(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed'}), 500


def get_feature_importance():
    """
    Top 5 features of the trained model.
    This is global importance: the same for every player, not a per-prediction explanation.
    """
    if model is None or not model.feature_importance:
        return {}
    return dict(list(model.feature_importance.items())[:5])


def get_risk_level(score):
    """Determine risk level based on adaptability score"""
    if score >= 70:
        return "Low Risk"
    elif score >= 50:
        return "Medium Risk"
    else:
        return "High Risk"


def get_recommendation(score):
    """Get recommendation based on adaptability score"""
    if score >= 70:
        return "Highly recommended transfer. Player shows strong adaptability indicators."
    elif score >= 50:
        return "Moderate risk transfer. Consider additional factors and gradual integration."
    else:
        return "High risk transfer. Requires careful evaluation and strong support system."


@app.route('/api/leagues')
def get_leagues():
    """Get available leagues"""
    return jsonify(LEAGUES)


@app.route('/api/positions')
def get_positions():
    """Get available positions"""
    return jsonify(POSITIONS)


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith('/api/') or request.path == '/predict':
        return jsonify({'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(error):
    if request.path.startswith('/api/') or request.path == '/predict':
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500


if __name__ == '__main__':
    if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8' and hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    if not load_model():
        print("Warning: No trained model found. Run 'python train_model.py' first.")

    app.run(
        debug=os.getenv('FLASK_DEBUG', '0') == '1',
        host=os.getenv('FLASK_HOST', '127.0.0.1'),
        port=int(os.getenv('FLASK_PORT', '5000'))
    )
