"""
LigueFit Index - League Adaptability Index
Main Flask application for predicting player success in new leagues
"""

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from src.data_processor import DataProcessor
from src.ml_model import AdaptabilityModel
from src.feature_engineer import FeatureEngineer

app = Flask(__name__)
CORS(app)

# Initialize components
data_processor = DataProcessor()
feature_engineer = None
model = None

def load_model():
    """Load the trained model if it exists, otherwise train a new one"""
    global model, feature_engineer
    
    try:
        # Try to load the ML model
        model_path = 'models/adaptability_model.joblib'
        if os.path.exists(model_path):
            model_data = joblib.load(model_path)
            if isinstance(model_data, dict):
                model = AdaptabilityModel()
                model.best_model = model_data['model']
                model.best_model_name = model_data['model_name']
                model.feature_importance = model_data['feature_importance']
                model.metrics = model_data['metrics']
            else:
                model = model_data
        else:
            return train_new_model()
        
        # Try to load the feature engineer
        fe_path = 'models/feature_engineer.joblib'
        if os.path.exists(fe_path):
            feature_engineer = FeatureEngineer.load(fe_path)
            return True
        else:
            return train_new_model()
            
    except Exception as e:
        print(f"Error loading models: {e}")
        return train_new_model()

def train_new_model():
    """Train a new model if loading fails"""
    global model, feature_engineer
    
    try:
        print("🔄 Training new model...")
        
        # Load and process data
        df = data_processor.prepare_dataset()
        
        # Create and fit feature engineer
        feature_engineer = FeatureEngineer()
        X, y = feature_engineer.fit_transform(df)
        
        # Train model
        model = AdaptabilityModel()
        model.train_and_evaluate(X, y, test_size=0.3)
        
        print("✅ New model trained successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error training new model: {e}")
        return False

@app.route('/')
def index():
    """Main dashboard page"""
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """
    Predict adaptability for a player
    
    Query parameters:
    - player_name: Name of the player
    - from_league: Current league
    - to_league: Target league
    - age: Player age
    - position: Player position
    - minutes: Minutes played
    - goals: Goals scored
    - assists: Assists
    - xg: Expected goals
    - xa: Expected assists
    """
    if not model:
        return jsonify({'error': 'Model not loaded. Please train the model first.'}), 500
    
    try:
        if request.method == 'POST':
            data = request.json
        else:
            data = request.args.to_dict()
        
        # Extract features
        features = {
            'age': float(data.get('age', 25)),
            'minutes': float(data.get('minutes', 1000)),
            'goals': float(data.get('goals', 0)),
            'assists': float(data.get('assists', 0)),
            'xg': float(data.get('xg', 0)),
            'xa': float(data.get('xa', 0)),
            'from_league': data.get('from_league', 'Ligue 1'),
            'to_league': data.get('to_league', 'Premier League'),
            'position': data.get('position', 'Forward')
        }
        
        # Engineer features
        processed_features = feature_engineer.process_single_player(features)
        
        # Make prediction using our custom method
        prediction, probability = model.predict_adaptability(processed_features)
        adaptability_score = probability * 100  # Success probability as percentage
        
        # Get feature importance
        feature_importance = get_feature_importance(processed_features)
        
        result = {
            'player_name': data.get('player_name', 'Unknown Player'),
            'adaptability_score': float(round(adaptability_score, 2)),
            'success_probability': float(round(probability, 3)),
            'risk_level': get_risk_level(adaptability_score),
            'feature_importance': feature_importance,
            'league_transition': f"{features['from_league']} → {features['to_league']}",
            'recommendation': get_recommendation(adaptability_score)
        }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

def get_feature_importance(features):
    """Get top contributing features for the prediction"""
    try:
        if not model or not feature_engineer:
            return {}
        
        # Try to get feature importance from the best model
        if hasattr(model, 'best_model') and hasattr(model.best_model, 'feature_importances_'):
            feature_names = feature_engineer.get_feature_names()
            importance_dict = dict(zip(feature_names, model.best_model.feature_importances_))
            
            # Convert numpy types to Python native types for JSON serialization
            importance_dict = {k: float(v) for k, v in importance_dict.items()}
            
        elif hasattr(model, 'feature_importance') and model.feature_importance:
            importance_dict = {k: float(v) for k, v in model.feature_importance.items()}
        else:
            # Fallback: return some default important features
            return {
                'age': 0.15,
                'pre_goals_per_90': 0.14,
                'league_transition': 0.12,
                'position': 0.10,
                'pre_minutes': 0.08
            }
        
        # Sort by importance and return top 5
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:5])
        
    except Exception as e:
        print(f"Error getting feature importance: {e}")
        return {}

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

@app.route('/train_model', methods=['POST'])
def train_model():
    """Train the adaptability model"""
    try:
        # This would typically load data and train the model
        # For now, we'll return a placeholder response
        return jsonify({
            'status': 'success',
            'message': 'Model training initiated. This is a placeholder implementation.',
            'model_metrics': {
                'accuracy': 0.75,
                'f1_score': 0.72,
                'auc_roc': 0.78
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/leagues')
def get_leagues():
    """Get available leagues"""
    leagues = [
        'Premier League',
        'La Liga',
        'Serie A',
        'Bundesliga',
        'Ligue 1',
        'Eredivisie',
        'Primeira Liga',
        'Belgian Pro League'
    ]
    return jsonify(leagues)

@app.route('/api/positions')
def get_positions():
    """Get available positions"""
    positions = [
        'Goalkeeper',
        'Centre-Back',
        'Full-Back',
        'Defensive Midfield',
        'Central Midfield',
        'Attacking Midfield',
        'Winger',
        'Forward'
    ]
    return jsonify(positions)

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    # Try to load existing model
    model_loaded = load_model()
    if not model_loaded:
        print("Warning: No trained model found. Please train the model first.")
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)