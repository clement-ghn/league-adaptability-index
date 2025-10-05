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
        
        # Performance consistency (xG vs actual goals)
        df['pre_finishing_ability'] = df['pre_xg_overperformance'] / (df['pre_xg'] + 0.1)
        
        # For prediction mode, we don't have post-transfer data
        if 'post_xg_overperformance' in df.columns:
            df['post_finishing_ability'] = df['post_xg_overperformance'] / (df['post_xg'] + 0.1)
        else:
            df['post_finishing_ability'] = 0  # Default for prediction
        
        # Playing time adaptation
        if 'post_minutes' in df.columns:
            df['minutes_ratio'] = df['post_minutes'] / (df['pre_minutes'] + 1)
        else:
            df['minutes_ratio'] = 1.0  # Default assumption
        
        # Performance adaptation ratios
        if 'post_goals_per_90' in df.columns:
            df['goals_adaptation'] = (df['post_goals_per_90'] + 0.01) / (df['pre_goals_per_90'] + 0.01)
            df['assists_adaptation'] = (df['post_assists_per_90'] + 0.01) / (df['pre_assists_per_90'] + 0.01)
            df['xg_adaptation'] = (df['post_xg_per_90'] + 0.01) / (df['pre_xg_per_90'] + 0.01)
        else:
            # Default values for prediction mode
            df['goals_adaptation'] = 1.0
            df['assists_adaptation'] = 1.0
            df['xg_adaptation'] = 1.0
        
        return df
    
    def create_positional_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create position-specific features"""
        df = df.copy()
        
        # Position categories for easier encoding
        position_mapping = {
            'Goalkeeper': 'GK',
            'Centre-Back': 'DEF',
            'Full-Back': 'DEF',
            'Defensive Midfield': 'MID',
            'Central Midfield': 'MID',
            'Attacking Midfield': 'MID',
            'Winger': 'ATT',
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
        
        # Get league characteristics
        league_chars = {
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
        
        # Performance consistency under pressure
        df['performance_under_pressure'] = (
            df['pre_finishing_ability'] * (1 - df['transition_difficulty'])
        )
        
        return df
    
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
            'age', 'transfer_fee',
            
            # Pre-transfer performance
            'pre_goals_per_90', 'pre_assists_per_90', 'pre_xg_per_90', 'pre_xa_per_90',
            'pre_finishing_ability', 'pre_minutes',
            
            # League transition features
            'intensity_jump', 'pace_jump', 'physicality_jump', 'technical_jump', 'pressing_jump',
            'transition_difficulty', 'league_rank_jump',
            'from_league_rank', 'to_league_rank',
            
            # Position features
            'is_def', 'is_mid', 'is_att', 'is_gk',
            
            # Interaction features
            'age_goals_interaction', 'age_physicality_interaction',
            'transition_difficulty_age', 'intensity_jump_minutes',
            'performance_under_pressure',
            
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
            'assists': 'pre_assists',
            'xg': 'pre_xg',
            'xa': 'pre_xa'
        }
        
        # Rename columns if needed
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns and new_name not in df.columns:
                df[new_name] = df[old_name]
        
        # Add missing columns with default values
        required_columns = ['transfer_fee', 'pre_minutes', 'pre_goals', 'pre_assists', 
                          'pre_xg', 'pre_xa']
        
        for col in required_columns:
            if col not in df.columns:
                if col == 'transfer_fee':
                    df[col] = 30  # Default transfer fee in millions
                elif col == 'pre_minutes':
                    df[col] = 2000  # Default minutes
                else:
                    df[col] = 0  # Default for performance stats
        
        # Calculate basic derived metrics
        df['pre_goals_per_90'] = (df['pre_goals'] / df['pre_minutes']) * 90
        df['pre_assists_per_90'] = (df['pre_assists'] / df['pre_minutes']) * 90
        df['pre_xg_per_90'] = (df['pre_xg'] / df['pre_minutes']) * 90
        df['pre_xa_per_90'] = (df['pre_xa'] / df['pre_minutes']) * 90
        df['pre_xg_overperformance'] = df['pre_goals'] - df['pre_xg']
        
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