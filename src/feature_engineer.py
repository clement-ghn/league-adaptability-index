"""
Feature Engineering Module for LigueFit Index
Transforms raw player and league data into ML-ready features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import os
import json
from datetime import datetime


class FeatureEngineer:
    """
    Feature engineering class for creating ML features from player and league data
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = []
        self.is_fitted = False
        
    def create_base_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create base features from raw data"""
        df = df.copy()
        
        # Age categories
        df['age_category'] = pd.cut(df['age'], 
                                   bins=[15, 21, 25, 28, 35, 45], 
                                   labels=['Young', 'Emerging', 'Prime', 'Experienced', 'Veteran'])
        
        # Playing time adaptation
        if 'post_minutes' in df.columns:
            df['minutes_ratio'] = df['post_minutes'] / (df['pre_minutes'] + 1)
        else:
            df['minutes_ratio'] = 1.0  # Default assumption
        
        # Performance adaptation ratios
        if 'post_goals_per_90' in df.columns:
            df['goals_adaptation'] = (df['post_goals_per_90'] + 0.01) / (df['pre_goals_per_90'] + 0.01)
            df['assists_adaptation'] = (df['post_assists_per_90'] + 0.01) / (df['pre_assists_per_90'] + 0.01)
        else:
            # Default values for prediction mode
            df['goals_adaptation'] = 1.0
            df['assists_adaptation'] = 1.0
        
        return df
    
    def create_positional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features"""
        df = df.copy()
        
        # Position categories (the four positions the API distinguishes, see build_dataset.py)
        position_mapping = {
            'Goalkeeper': 'GK',
            'Defender': 'DEF',
            'Midfielder': 'MID',
            'Forward': 'ATT'
        }
        
        df['position_category'] = df['position'].map(position_mapping)
        
        # Position-specific performance expectations
        for pos in ['DEF', 'MID', 'ATT', 'GK']:
            df[f'is_{pos.lower()}'] = (df['position_category'] == pos).astype(int)
        
        return df
    
    def create_league_transition_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features specific to league transitions"""
        df = df.copy()
        
        # Get league characteristics from Opta data or fallback to estimates
        league_chars = self._get_league_characteristics()
        
        # Add support for more leagues with estimates if not in Opta data
        fallback_chars = {
            'Eredivisie': {
                'intensity': 0.72, 'pace': 0.78, 'technical': 0.88, 
                'physicality': 0.65, 'pressing': 0.70
            },
            'Primeira Liga': {
                'intensity': 0.70, 'pace': 0.73, 'technical': 0.82, 
                'physicality': 0.75, 'pressing': 0.68
            }
        }
        
        # Merge Opta data with fallback estimates
        for league, chars in fallback_chars.items():
            if league not in league_chars:
                league_chars[league] = chars
        
        # Common transition types
        transition_types = {
            ('Ligue 1', 'Premier League'): 'L1_to_PL',
            ('Serie A', 'Premier League'): 'SA_to_PL',
            ('La Liga', 'Premier League'): 'LL_to_PL',
            ('Bundesliga', 'Premier League'): 'BL_to_PL',
            ('Premier League', 'Serie A'): 'PL_to_SA',
            ('Premier League', 'La Liga'): 'PL_to_LL',
        }
        
        df['transition_type'] = df.apply(
            lambda row: transition_types.get((row['from_league'], row['to_league']), 'Other'), 
            axis=1
        )
        
        # League prestige/difficulty rankings
        league_rankings = {
            'Premier League': 5, 'La Liga': 5, 'Serie A': 4, 
            'Bundesliga': 4, 'Ligue 1': 3, 'Eredivisie': 2, 'Primeira Liga': 2,
        }
        
        df['from_league_rank'] = df['from_league'].map(league_rankings).fillna(1)
        df['to_league_rank'] = df['to_league'].map(league_rankings).fillna(1)
        df['league_rank_jump'] = df['to_league_rank'] - df['from_league_rank']
        
        # Calculate league transition difficulty
        for idx, row in df.iterrows():
            from_league = row['from_league']
            to_league = row['to_league']
            
            if from_league in league_chars and to_league in league_chars:
                from_chars = league_chars[from_league]
                to_chars = league_chars[to_league]
                
                # Calculate jumps for each characteristic
                df.loc[idx, 'intensity_jump'] = to_chars['intensity'] - from_chars['intensity']
                df.loc[idx, 'pace_jump'] = to_chars['pace'] - from_chars['pace']
                df.loc[idx, 'physicality_jump'] = to_chars['physicality'] - from_chars['physicality']
                df.loc[idx, 'technical_jump'] = to_chars['technical'] - from_chars['technical']
                df.loc[idx, 'pressing_jump'] = to_chars['pressing'] - from_chars['pressing']
                
                # Overall difficulty score
                difficulty_factors = [
                    abs(df.loc[idx, 'intensity_jump']),
                    abs(df.loc[idx, 'pace_jump']),
                    abs(df.loc[idx, 'physicality_jump']),
                    abs(df.loc[idx, 'pressing_jump'])
                ]
                df.loc[idx, 'transition_difficulty'] = sum(difficulty_factors) / len(difficulty_factors)
            else:
                # Default values for unknown leagues
                df.loc[idx, 'intensity_jump'] = 0
                df.loc[idx, 'pace_jump'] = 0
                df.loc[idx, 'physicality_jump'] = 0
                df.loc[idx, 'technical_jump'] = 0
                df.loc[idx, 'pressing_jump'] = 0
                df.loc[idx, 'transition_difficulty'] = 0.1
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different aspects"""
        df = df.copy()
        
        # Age and performance interactions
        df['age_goals_interaction'] = df['age'] * df['pre_goals_per_90']
        df['age_physicality_interaction'] = df['age'] * df['physicality_jump']
        
        # League difficulty and player profile interactions
        df['transition_difficulty_age'] = df['transition_difficulty'] * df['age']
        df['intensity_jump_minutes'] = df['intensity_jump'] * df['pre_minutes'] / 1000  # Normalize

        return df
    
    def _get_league_characteristics(self) -> Dict:
        """
        Get league characteristics from Opta data file or use defaults
        
        Returns:
            Dictionary with league characteristics
        """
        opta_file = "data/opta_league_characteristics.json"
        
        # Try to load Opta data first
        if os.path.exists(opta_file):
            try:
                with open(opta_file, 'r') as f:
                    opta_data = json.load(f)
                
                # Check if data is recent (less than 30 days old)
                if 'last_updated' in opta_data:
                    last_update = datetime.fromisoformat(opta_data['last_updated'])
                    days_old = (datetime.now() - last_update).days
                    
                    if days_old < 30:
                        print(f"📊 Using Opta data (updated {days_old} days ago)")
                        return opta_data['characteristics']
                    else:
                        print(f"⚠️ Opta data is {days_old} days old, using fallback estimates")
                
            except (json.JSONDecodeError, KeyError) as e:
                print(f"⚠️ Error reading Opta data: {e}, using fallback estimates")
        
        # Fallback to estimated characteristics
        print("📊 Using estimated league characteristics (consider updating with Opta data)")
        return {
            'Premier League': {
                'intensity': 0.95, 'pace': 0.90, 'technical': 0.80, 
                'physicality': 0.95, 'pressing': 0.85
            },
            'La Liga': {
                'intensity': 0.75, 'pace': 0.70, 'technical': 0.95, 
                'physicality': 0.70, 'pressing': 0.80
            },
            'Serie A': {
                'intensity': 0.80, 'pace': 0.75, 'technical': 0.90, 
                'physicality': 0.85, 'pressing': 0.78
            },
            'Bundesliga': {
                'intensity': 0.88, 'pace': 0.92, 'technical': 0.85, 
                'physicality': 0.90, 'pressing': 0.90
            },
            'Ligue 1': {
                'intensity': 0.78, 'pace': 0.82, 'technical': 0.85, 
                'physicality': 0.80, 'pressing': 0.75
            }
        }
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Encode categorical features"""
        df = df.copy()
        
        categorical_columns = ['position', 'position_category', 'transition_type', 
                             'from_league', 'to_league', 'age_category']
        
        for col in categorical_columns:
            if col in df.columns:
                if fit or col not in self.label_encoders:
                    if col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    # Handle unseen categories during prediction
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        # Assign a default value for unseen categories
                        df[f'{col}_encoded'] = 0
        
        return df
    
    def select_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select the final features for ML model"""
        
        # Define feature columns for ML
        feature_columns = [
            # Basic info
            'age',

            # Pre-transfer performance
            'pre_goals_per_90', 'pre_assists_per_90', 'pre_minutes',
            
            # League transition features
            'intensity_jump', 'pace_jump', 'physicality_jump', 'technical_jump', 'pressing_jump',
            'transition_difficulty', 'league_rank_jump',
            'from_league_rank', 'to_league_rank',
            
            # Position features
            'is_def', 'is_mid', 'is_att', 'is_gk',
            
            # Interaction features
            'age_goals_interaction', 'age_physicality_interaction',
            'transition_difficulty_age', 'intensity_jump_minutes',
            
            # Encoded categorical features
            'position_category_encoded', 'transition_type_encoded'
        ]
        
        # Only include columns that exist in the dataframe
        available_features = [col for col in feature_columns if col in df.columns]
        self.feature_names = available_features
        
        return df[available_features]
    
    def fit_transform(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Complete feature engineering pipeline - fit and transform"""
        
        # Create all features
        df = self.create_base_features(df)
        df = self.create_positional_features(df)
        df = self.create_league_transition_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categorical_features(df, fit=True)
        
        # Select features
        X = self.select_features(df)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        
        # Target variable
        y = df['success'] if 'success' in df.columns else None
        
        self.is_fitted = True
        return X_scaled, y
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted encoders and scaler"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before transform")
        
        # Create all features
        df = self.create_base_features(df)
        df = self.create_positional_features(df)
        df = self.create_league_transition_features(df)
        df = self.create_interaction_features(df)
        df = self.encode_categorical_features(df, fit=False)
        
        # Select features
        X = self.select_features(df)
        
        # Scale features
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        
        return X_scaled
    
    def process_single_player(self, player_data: Dict) -> List[float]:
        """
        Process a single player's data for prediction
        
        Args:
            player_data: Dictionary with player information
            
        Returns:
            List of processed features ready for model prediction
        """
        # Create a single-row DataFrame
        df = pd.DataFrame([player_data])
        
        # Map input column names to expected names
        column_mapping = {
            'minutes': 'pre_minutes',
            'goals': 'pre_goals',
            'assists': 'pre_assists'
        }

        # Rename columns if needed
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]

        # Add missing columns with default values
        required_columns = ['pre_minutes', 'pre_goals', 'pre_assists']

        for col in required_columns:
            if col not in df.columns:
                if col == 'pre_minutes':
                    df[col] = 2000  # Default minutes
                else:
                    df[col] = 0  # Default for performance stats

        # Calculate basic derived metrics
        df['pre_goals_per_90'] = (df['pre_goals'] / df['pre_minutes']) * 90
        df['pre_assists_per_90'] = (df['pre_assists'] / df['pre_minutes']) * 90
        
        # Transform using the fitted pipeline
        X_processed = self.transform(df)
        
        return X_processed.iloc[0].tolist()
    
    def get_feature_names(self) -> List[str]:
        """Get the names of the final features"""
        return self.feature_names
    
    def save(self, filepath: str = 'models/feature_engineer.joblib'):
        """Save the fitted feature engineer"""
        if not self.is_fitted:
            raise ValueError("FeatureEngineer must be fitted before saving")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"FeatureEngineer saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str = 'models/feature_engineer.joblib'):
        """Load a fitted feature engineer"""
        if os.path.exists(filepath):
            return joblib.load(filepath)
        else:
            raise FileNotFoundError(f"No feature engineer found at {filepath}")


if __name__ == "__main__":
    # Test the feature engineer
    from data_processor import DataProcessor
    
    # Load sample data
    processor = DataProcessor()
    df = processor.prepare_dataset()
    
    # Test feature engineering
    fe = FeatureEngineer()
    X, y = fe.fit_transform(df)
    
    print("Feature engineering completed!")
    print(f"Features shape: {X.shape}")
    print(f"Feature names: {fe.get_feature_names()}")
    print("\nSample features:")
    print(X.head())
    
    # Save the feature engineer
    fe.save()