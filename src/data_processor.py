"""
Data Processing Module for LigueFit Index
Handles data ingestion, cleaning, and preparation from various sources
"""

import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import time
import os
from typing import Dict, List, Optional, Tuple


class DataProcessor:
    """Main class for processing football transfer and performance data"""
    
    def __init__(self):
        self.transfer_data = None
        self.player_stats = None
        self.league_characteristics = None
        
    def load_sample_data(self) -> pd.DataFrame:
        """
        Create sample transfer data for MVP demonstration
        In production, this would load from actual data sources
        """
        # Sample transfer data
        transfers = [
            # Ligue 1 to Premier League
            {'player_name': 'Hugo Ekitike', 'from_league': 'Ligue 1', 'to_league': 'Premier League',
             'age': 21, 'position': 'Forward', 'transfer_fee': 35, 'season': '2023-24',
             'pre_minutes': 1200, 'pre_goals': 8, 'pre_assists': 2, 'pre_xg': 7.5, 'pre_xa': 2.1,
             'post_minutes': 800, 'post_goals': 3, 'post_assists': 1, 'post_xg': 4.2, 'post_xa': 1.2,
             'success': 0},  # Struggled to adapt
            
            {'player_name': 'Christopher Nkunku', 'from_league': 'Ligue 1', 'to_league': 'Premier League',
             'age': 25, 'position': 'Forward', 'transfer_fee': 65, 'season': '2023-24',
             'pre_minutes': 2500, 'pre_goals': 23, 'pre_assists': 17, 'pre_xg': 20.8, 'pre_xa': 15.2,
             'post_minutes': 600, 'post_goals': 4, 'post_assists': 2, 'post_xg': 3.8, 'post_xa': 2.1,
             'success': 1},  # Good adaptation despite injuries
            
            # Serie A to Premier League
            {'player_name': 'João Pedro', 'from_league': 'Serie A', 'to_league': 'Premier League',
             'age': 22, 'position': 'Forward', 'transfer_fee': 30, 'season': '2023-24',
             'pre_minutes': 2800, 'pre_goals': 9, 'pre_assists': 4, 'pre_xg': 11.2, 'pre_xa': 4.8,
             'post_minutes': 2200, 'post_goals': 11, 'post_assists': 6, 'post_xg': 9.8, 'post_xa': 5.2,
             'success': 1},  # Successful adaptation
            
            # La Liga to Premier League
            {'player_name': 'Dominik Szoboszlai', 'from_league': 'Bundesliga', 'to_league': 'Premier League',
             'age': 22, 'position': 'Central Midfield', 'transfer_fee': 70, 'season': '2023-24',
             'pre_minutes': 2600, 'pre_goals': 10, 'pre_assists': 13, 'pre_xg': 8.5, 'pre_xa': 11.8,
             'post_minutes': 2400, 'post_goals': 3, 'post_assists': 4, 'post_xg': 4.2, 'post_xa': 6.5,
             'success': 1},  # Good adaptation
            
            # More sample data...
            {'player_name': 'Nicolas Jackson', 'from_league': 'La Liga', 'to_league': 'Premier League',
             'age': 22, 'position': 'Forward', 'transfer_fee': 35, 'season': '2023-24',
             'pre_minutes': 2000, 'pre_goals': 12, 'pre_assists': 4, 'pre_xg': 13.5, 'pre_xa': 3.8,
             'post_minutes': 2500, 'post_goals': 14, 'post_assists': 5, 'post_xg': 12.8, 'post_xa': 4.2,
             'success': 1},  # Successful
             
            # Add more diverse examples
            {'player_name': 'Mason Mount', 'from_league': 'Premier League', 'to_league': 'Premier League',
             'age': 24, 'position': 'Attacking Midfield', 'transfer_fee': 60, 'season': '2023-24',
             'pre_minutes': 2200, 'pre_goals': 3, 'pre_assists': 6, 'pre_xg': 4.2, 'pre_xa': 8.1,
             'post_minutes': 1800, 'post_goals': 1, 'post_assists': 3, 'post_xg': 2.8, 'post_xa': 4.5,
             'success': 0},  # Struggled with new system
             
            # Bundesliga to Premier League examples
            {'player_name': 'Kai Havertz', 'from_league': 'Bundesliga', 'to_league': 'Premier League',
             'age': 21, 'position': 'Attacking Midfield', 'transfer_fee': 80, 'season': '2020-21',
             'pre_minutes': 2400, 'pre_goals': 12, 'pre_assists': 6, 'pre_xg': 11.5, 'pre_xa': 7.2,
             'post_minutes': 2600, 'post_goals': 8, 'post_assists': 9, 'post_xg': 9.8, 'post_xa': 8.1,
             'success': 1},  # Good adaptation
             
            {'player_name': 'Timo Werner', 'from_league': 'Bundesliga', 'to_league': 'Premier League',
             'age': 24, 'position': 'Forward', 'transfer_fee': 50, 'season': '2020-21',
             'pre_minutes': 2800, 'pre_goals': 28, 'pre_assists': 8, 'pre_xg': 24.5, 'pre_xa': 6.8,
             'post_minutes': 2200, 'post_goals': 6, 'post_assists': 8, 'post_xg': 8.9, 'post_xa': 7.2,
             'success': 0},  # Struggled to adapt
             
            # La Liga to Premier League
            {'player_name': 'Ferran Torres', 'from_league': 'La Liga', 'to_league': 'Premier League',
             'age': 20, 'position': 'Winger', 'transfer_fee': 25, 'season': '2020-21',
             'pre_minutes': 1800, 'pre_goals': 6, 'pre_assists': 6, 'pre_xg': 7.2, 'pre_xa': 5.8,
             'post_minutes': 2400, 'post_goals': 13, 'post_assists': 3, 'post_xg': 11.8, 'post_xa': 4.2,
             'success': 1},  # Successful adaptation
             
            # Serie A to Premier League  
            {'player_name': 'Romelu Lukaku', 'from_league': 'Serie A', 'to_league': 'Premier League',
             'age': 28, 'position': 'Forward', 'transfer_fee': 100, 'season': '2021-22',
             'pre_minutes': 3200, 'pre_goals': 24, 'pre_assists': 11, 'pre_xg': 22.1, 'pre_xa': 9.8,
             'post_minutes': 2800, 'post_goals': 8, 'post_assists': 2, 'post_xg': 12.5, 'post_xa': 3.1,
             'success': 0},  # Struggled to readapt
             
            # More Ligue 1 to Premier League
            {'player_name': 'Thiago Silva', 'from_league': 'Ligue 1', 'to_league': 'Premier League',
             'age': 35, 'position': 'Centre-Back', 'transfer_fee': 0, 'season': '2020-21',
             'pre_minutes': 2600, 'pre_goals': 2, 'pre_assists': 3, 'pre_xg': 1.8, 'pre_xa': 2.1,
             'post_minutes': 2800, 'post_goals': 2, 'post_assists': 1, 'post_xg': 1.9, 'post_xa': 1.2,
             'success': 1},  # Great adaptation despite age
        ]
        
        return pd.DataFrame(transfers)
    
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
        """Calculate derived performance metrics"""
        df = df.copy()
        
        # Pre-transfer metrics (per 90 minutes)
        df['pre_goals_per_90'] = (df['pre_goals'] / df['pre_minutes']) * 90
        df['pre_assists_per_90'] = (df['pre_assists'] / df['pre_minutes']) * 90
        df['pre_xg_per_90'] = (df['pre_xg'] / df['pre_minutes']) * 90
        df['pre_xa_per_90'] = (df['pre_xa'] / df['pre_minutes']) * 90
        df['pre_goal_involvement'] = df['pre_goals'] + df['pre_assists']
        df['pre_xg_overperformance'] = df['pre_goals'] - df['pre_xg']
        
        # Post-transfer metrics
        df['post_goals_per_90'] = (df['post_goals'] / df['post_minutes']) * 90
        df['post_assists_per_90'] = (df['post_assists'] / df['post_minutes']) * 90
        df['post_xg_per_90'] = (df['post_xg'] / df['post_minutes']) * 90
        df['post_xa_per_90'] = (df['post_xa'] / df['post_minutes']) * 90
        df['post_goal_involvement'] = df['post_goals'] + df['post_assists']
        df['post_xg_overperformance'] = df['post_goals'] - df['post_xg']
        
        # Performance change metrics
        df['goals_change'] = df['post_goals_per_90'] - df['pre_goals_per_90']
        df['assists_change'] = df['post_assists_per_90'] - df['pre_assists_per_90']
        df['xg_change'] = df['post_xg_per_90'] - df['pre_xg_per_90']
        
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
        critical_columns = ['age', 'pre_minutes', 'post_minutes', 'from_league', 'to_league']
        df = df.dropna(subset=critical_columns)
        
        # Remove unrealistic values
        df = df[df['age'].between(16, 40)]
        df = df[df['pre_minutes'] > 0]
        df = df[df['post_minutes'] > 0]
        
        # Fill missing performance stats with 0
        performance_cols = ['pre_goals', 'pre_assists', 'pre_xg', 'pre_xa', 
                          'post_goals', 'post_assists', 'post_xg', 'post_xa']
        df[performance_cols] = df[performance_cols].fillna(0)
        
        return df
    
    def prepare_dataset(self) -> pd.DataFrame:
        """Main method to prepare the complete dataset"""
        print("Loading sample data...")
        df = self.load_sample_data()
        
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