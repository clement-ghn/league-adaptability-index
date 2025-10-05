"""
Machine Learning Model Module for LigueFit Index
Implements and trains adaptability prediction models
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.metrics import classification_report, confusion_matrix
import xgboost as xgb
import joblib
import os
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns


class AdaptabilityModel:
    """
    Main class for training and evaluating player adaptability models
    """
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_importance = None
        self.metrics = {}
        
    def initialize_models(self) -> Dict[str, Any]:
        """Initialize different ML models to compare"""
        models = {
            'logistic_regression': LogisticRegression(
                random_state=42, 
                max_iter=1000,
                class_weight='balanced'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                class_weight='balanced',
                max_depth=10
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1,
                max_depth=6
            ),
            'xgboost': xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                learning_rate=0.1,
                max_depth=6,
                eval_metric='logloss'
            )
        }
        
        self.models = models
        return models
    
    def train_and_evaluate(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2) -> Dict[str, Dict]:
        """
        Train and evaluate all models
        
        Args:
            X: Features DataFrame
            y: Target Series
            test_size: Proportion of data for testing
            
        Returns:
            Dictionary with metrics for each model
        """
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"Training set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        print(f"Class distribution - Success: {y_train.sum()}, Failure: {len(y_train) - y_train.sum()}")
        
        # Initialize models
        self.initialize_models()
        
        results = {}
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Train the model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, zero_division=0),
                'recall': recall_score(y_test, y_pred, zero_division=0),
                'f1_score': f1_score(y_test, y_pred, zero_division=0),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            # Cross-validation score
            cv_scores = cross_val_score(model, X_train, y_train, cv=3, scoring='f1')
            metrics['cv_f1_mean'] = cv_scores.mean()
            metrics['cv_f1_std'] = cv_scores.std()
            
            results[name] = metrics
            
            print(f"Accuracy: {metrics['accuracy']:.3f}")
            print(f"F1-Score: {metrics['f1_score']:.3f}")
            print(f"ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"CV F1: {metrics['cv_f1_mean']:.3f} (+/- {metrics['cv_f1_std']:.3f})")
        
        # Find best model based on F1 score
        best_model_name = max(results.keys(), key=lambda k: results[k]['f1_score'])
        self.best_model = self.models[best_model_name]
        self.best_model_name = best_model_name
        self.metrics = results
        
        print(f"\nBest model: {best_model_name} with F1-score: {results[best_model_name]['f1_score']:.3f}")
        
        # Get feature importance
        self.extract_feature_importance(X.columns)
        
        return results
    
    def extract_feature_importance(self, feature_names):
        """Extract and store feature importance from the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance_scores = self.best_model.feature_importances_
        elif hasattr(self.best_model, 'coef_'):
            # For logistic regression, use absolute coefficients
            importance_scores = np.abs(self.best_model.coef_[0])
        else:
            print("Cannot extract feature importance for this model type")
            return
        
        # Create feature importance dictionary
        self.feature_importance = dict(zip(feature_names, importance_scores))
        
        # Sort by importance
        self.feature_importance = dict(
            sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        )
    
    def plot_feature_importance(self, top_n: int = 15, save_path: str = None):
        """Plot feature importance"""
        if self.feature_importance is None:
            print("No feature importance available")
            return
        
        # Get top N features
        top_features = dict(list(self.feature_importance.items())[:top_n])
        
        plt.figure(figsize=(10, 8))
        features = list(top_features.keys())
        importances = list(top_features.values())
        
        plt.barh(range(len(features)), importances)
        plt.yticks(range(len(features)), features)
        plt.xlabel('Importance')
        plt.title(f'Top {top_n} Feature Importances - {self.best_model_name}')
        plt.gca().invert_yaxis()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Feature importance plot saved to {save_path}")
        
        plt.show()
    
    def plot_model_comparison(self, save_path: str = None):
        """Plot comparison of different models"""
        if not self.metrics:
            print("No metrics available for comparison")
            return
        
        metrics_df = pd.DataFrame(self.metrics).T
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
        
        for i, metric in enumerate(metrics_to_plot):
            ax = axes[i]
            values = metrics_df[metric].values
            models = metrics_df.index.values
            
            bars = ax.bar(models, values)
            ax.set_title(f'{metric.title().replace("_", " ")}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{value:.3f}', ha='center', va='bottom')
            
            # Rotate x-axis labels
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Model comparison plot saved to {save_path}")
        
        plt.show()
    
    def hyperparameter_tuning(self, X: pd.DataFrame, y: pd.Series, model_name: str = 'xgboost'):
        """
        Perform hyperparameter tuning for the specified model
        
        Args:
            X: Features DataFrame
            y: Target Series
            model_name: Name of the model to tune
        """
        if model_name not in self.models:
            print(f"Model {model_name} not found")
            return
        
        print(f"Performing hyperparameter tuning for {model_name}...")
        
        # Define parameter grids
        param_grids = {
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 1.0]
            }
        }
        
        if model_name not in param_grids:
            print(f"No parameter grid defined for {model_name}")
            return
        
        # Perform grid search
        grid_search = GridSearchCV(
            self.models[model_name],
            param_grids[model_name],
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {grid_search.best_score_:.3f}")
        
        # Update the model with best parameters
        self.models[model_name] = grid_search.best_estimator_
        
        return grid_search.best_params_
    
    def predict_adaptability(self, X: np.ndarray) -> Tuple[int, float]:
        """
        Predict adaptability for new data
        
        Args:
            X: Features array
            
        Returns:
            Tuple of (prediction, probability)
        """
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        prediction = self.best_model.predict([X])[0]
        probability = self.best_model.predict_proba([X])[0][1]
        
        return prediction, probability
    
    def save_model(self, filepath: str = 'models/adaptability_model.joblib'):
        """Save the trained model"""
        if self.best_model is None:
            raise ValueError("No model trained yet")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.best_model,
            'model_name': self.best_model_name,
            'feature_importance': self.feature_importance,
            'metrics': self.metrics
        }
        
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str = 'models/adaptability_model.joblib'):
        """Load a trained model"""
        if os.path.exists(filepath):
            model_data = joblib.load(filepath)
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.feature_importance = model_data['feature_importance']
            self.metrics = model_data['metrics']
            print(f"Model loaded from {filepath}")
        else:
            raise FileNotFoundError(f"No model found at {filepath}")
    
    def generate_report(self, X_test: pd.DataFrame, y_test: pd.Series) -> str:
        """Generate a comprehensive model report"""
        if self.best_model is None:
            return "No model available for report generation"
        
        # Make predictions
        y_pred = self.best_model.predict(X_test)
        y_pred_proba = self.best_model.predict_proba(X_test)[:, 1]
        
        # Generate report
        report = f"""
        LigueFit Index - Model Performance Report
        =======================================
        
        Best Model: {self.best_model_name}
        
        Test Set Performance:
        - Accuracy: {accuracy_score(y_test, y_pred):.3f}
        - Precision: {precision_score(y_test, y_pred, zero_division=0):.3f}
        - Recall: {recall_score(y_test, y_pred, zero_division=0):.3f}
        - F1-Score: {f1_score(y_test, y_pred, zero_division=0):.3f}
        - ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.3f}
        
        Classification Report:
        {classification_report(y_test, y_pred, target_names=['Failure', 'Success'])}
        
        Top 10 Most Important Features:
        """
        
        if self.feature_importance:
            for i, (feature, importance) in enumerate(list(self.feature_importance.items())[:10]):
                report += f"\n        {i+1:2d}. {feature}: {importance:.4f}"
        
        return report


if __name__ == "__main__":
    # Test the model training
    from data_processor import DataProcessor
    from feature_engineer import FeatureEngineer
    
    # Load and prepare data
    processor = DataProcessor()
    df = processor.prepare_dataset()
    
    # Feature engineering
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    
    # Train model
    model = AdaptabilityModel()
    results = model.train_and_evaluate(X, y)
    
    # Plot results
    model.plot_model_comparison()
    model.plot_feature_importance()
    
    # Save model
    model.save_model()
    
    print("\nModel training completed!")
    print(f"Best model: {model.best_model_name}")
    print("Model and feature engineer saved.")