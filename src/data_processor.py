"""
Data Processing Module for LigueFit Index
Loads the transfer dataset built from API-Football (see build_dataset.py) and prepares it for modelling
"""

import pandas as pd
import numpy as np
import requests
import time
import os
from typing import Dict, List, Optional, Tuple


class DataProcessor:
    """Main class for processing football transfer and performance data"""

    def __init__(self):
        self.transfer_data = None
        self.player_stats = None
        self.league_characteristics = None

    def load_transfer_dataset(self) -> pd.DataFrame:
        """
        Load the transfers built by src/build_dataset.py.
        Run `python src/build_dataset.py` first.
        """
        filepath = os.path.join('data', 'transfers_dataset.csv')
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"{filepath} not found. Run 'python src/build_dataset.py' first.")
        return pd.read_csv(filepath)

    def get_league_characteristics(self) -> Dict[str, Dict[str, float]]:
        """
        Define characteristics of different leagues
        These would typically be calculated from actual match data
        """
        return {
            'Premier League': {
                'intensity': 0.95,      # Physical intensity (0-1)
                'pace': 0.90,          # Game pace
                'technical': 0.80,     # Technical level
                'physicality': 0.95,   # Physical demands
                'pressing': 0.85,      # Pressing intensity
                'aerial_duels': 0.90,  # Aerial play importance
                'transition_speed': 0.88
            },
            'La Liga': {
                'intensity': 0.75,
                'pace': 0.70,
                'technical': 0.95,
                'physicality': 0.70,
                'pressing': 0.80,
                'aerial_duels': 0.65,
                'transition_speed': 0.75
            },
            'Serie A': {
                'intensity': 0.80,
                'pace': 0.75,
                'technical': 0.90,
                'physicality': 0.85,
                'pressing': 0.78,
                'aerial_duels': 0.80,
                'transition_speed': 0.70
            },
            'Bundesliga': {
                'intensity': 0.88,
                'pace': 0.92,
                'technical': 0.85,
                'physicality': 0.90,
                'pressing': 0.90,
                'aerial_duels': 0.85,
                'transition_speed': 0.85
            },
            'Ligue 1': {
                'intensity': 0.78,
                'pace': 0.82,
                'technical': 0.85,
                'physicality': 0.80,
                'pressing': 0.75,
                'aerial_duels': 0.75,
                'transition_speed': 0.80
            }
        }

    def calculate_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate derived performance metrics (per 90 minutes)"""
        df = df.copy()

        # Pre-transfer metrics (per 90 minutes)
        df['pre_goals_per_90'] = (df['pre_goals'] / df['pre_minutes']) * 90
        df['pre_assists_per_90'] = (df['pre_assists'] / df['pre_minutes']) * 90
        df['pre_goal_involvement'] = df['pre_goals'] + df['pre_assists']

        # Post-transfer metrics
        df['post_goals_per_90'] = (df['post_goals'] / df['post_minutes']) * 90
        df['post_assists_per_90'] = (df['post_assists'] / df['post_minutes']) * 90
        df['post_goal_involvement'] = df['post_goals'] + df['post_assists']

        # Performance change metrics
        df['goals_change'] = df['post_goals_per_90'] - df['pre_goals_per_90']
        df['assists_change'] = df['post_assists_per_90'] - df['pre_assists_per_90']

        return df

    def add_league_difficulty_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add features based on league characteristics and transition difficulty"""
        df = df.copy()
        league_chars = self.get_league_characteristics()

        for idx, row in df.iterrows():
            from_league = row['from_league']
            to_league = row['to_league']

            if from_league in league_chars and to_league in league_chars:
                from_chars = league_chars[from_league]
                to_chars = league_chars[to_league]

                # Calculate transition difficulty for each characteristic
                df.loc[idx, 'intensity_jump'] = to_chars['intensity'] - from_chars['intensity']
                df.loc[idx, 'pace_jump'] = to_chars['pace'] - from_chars['pace']
                df.loc[idx, 'physicality_jump'] = to_chars['physicality'] - from_chars['physicality']
                df.loc[idx, 'technical_jump'] = to_chars['technical'] - from_chars['technical']
                df.loc[idx, 'pressing_jump'] = to_chars['pressing'] - from_chars['pressing']

                # Overall difficulty score
                difficulty_factors = ['intensity_jump', 'pace_jump', 'physicality_jump', 'pressing_jump']
                df.loc[idx, 'transition_difficulty'] = df.loc[idx, difficulty_factors].abs().mean()

        return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate the dataset"""
        df = df.copy()

        # Remove rows with missing critical data
        critical_columns = ['age', 'pre_minutes', 'post_minutes', 'from_league', 'to_league', 'position', 'success']
        df = df.dropna(subset=critical_columns)

        # Remove unrealistic values
        df = df[df['age'].between(16, 40)]
        df = df[df['pre_minutes'] > 0]
        df = df[df['post_minutes'] > 0]

        # Fill missing performance stats with 0
        performance_cols = ['pre_goals', 'pre_assists', 'post_goals', 'post_assists']
        df[performance_cols] = df[performance_cols].fillna(0)

        return df

    def prepare_dataset(self) -> pd.DataFrame:
        """Main method to prepare the complete dataset"""
        print("Loading transfer dataset...")
        df = self.load_transfer_dataset()

        print("Cleaning data...")
        df = self.clean_data(df)

        print("Calculating performance metrics...")
        df = self.calculate_performance_metrics(df)

        print("Adding league difficulty features...")
        df = self.add_league_difficulty_features(df)

        print(f"Dataset prepared with {len(df)} records and {len(df.columns)} features")
        return df

    def save_data(self, df: pd.DataFrame, filename: str = 'processed_transfers.csv'):
        """Save processed data to CSV"""
        filepath = os.path.join('data', filename)
        df.to_csv(filepath, index=False)
        print(f"Data saved to {filepath}")

    def load_data(self, filename: str = 'processed_transfers.csv') -> pd.DataFrame:
        """Load processed data from CSV"""
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            return pd.read_csv(filepath)
        else:
            print(f"File {filepath} not found. Creating new dataset...")
            return self.prepare_dataset()


# Utility functions for future data scraping
class TransfermarktScraper:
    """
    Placeholder for future Transfermarkt scraping functionality
    This would require careful implementation to respect robots.txt and rate limits
    """

    def __init__(self):
        self.base_url = "https://www.transfermarkt.com"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

    def get_player_transfers(self, player_name: str) -> List[Dict]:
        """
        Placeholder for scraping player transfer history
        """
        # This would implement actual scraping logic
        print(f"Scraping transfers for {player_name} (not implemented yet)")
        return []

    def get_league_teams(self, league_name: str) -> List[str]:
        """
        Placeholder for getting teams in a specific league
        """
        print(f"Getting teams for {league_name} (not implemented yet)")
        return []


if __name__ == "__main__":
    # Test the data processor
    processor = DataProcessor()
    df = processor.prepare_dataset()
    print("\nDataset sample:")
    print(df.head())
    print("\nDataset info:")
    print(df.info())
    processor.save_data(df)
