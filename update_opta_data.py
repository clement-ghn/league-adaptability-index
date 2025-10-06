"""
Script to update league characteristics with Opta data
Run this script to fetch fresh data from Opta API
"""

import os
import sys
import json
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from opta_integration import OptaDataIntegrator

def update_league_characteristics():
    """
    Update league characteristics with fresh Opta data
    """
    
    print("🚀 LigueFit Index - Opta Data Update")
    print("=" * 50)
    
    # Check for API key
    api_key = os.getenv('OPTA_API_KEY')
    
    if not api_key:
        print("❌ No Opta API key found!")
        print("\nTo use real Opta data:")
        print("1. Get an Opta Sports API key from https://www.optasports.com/")
        print("2. Set environment variable: set OPTA_API_KEY=your_key_here")
        print("3. Re-run this script")
        print("\n📊 Using sample/estimated data for now...")
        
        # Create sample data structure for demonstration
        sample_characteristics = {
            'Premier League': {
                'intensity': 0.942, 'pace': 0.891, 'technical': 0.798, 
                'physicality': 0.933, 'pressing': 0.847
            },
            'La Liga': {
                'intensity': 0.734, 'pace': 0.689, 'technical': 0.956, 
                'physicality': 0.671, 'pressing': 0.782
            },
            'Serie A': {
                'intensity': 0.798, 'pace': 0.743, 'technical': 0.912, 
                'physicality': 0.856, 'pressing': 0.767
            },
            'Bundesliga': {
                'intensity': 0.889, 'pace': 0.924, 'technical': 0.834, 
                'physicality': 0.901, 'pressing': 0.912
            },
            'Ligue 1': {
                'intensity': 0.767, 'pace': 0.823, 'technical': 0.845, 
                'physicality': 0.789, 'pressing': 0.743
            }
        }
        
        # Save sample data
        integrator = OptaDataIntegrator()
        integrator.save_characteristics_to_file(
            sample_characteristics, 
            "opta_league_characteristics.json"
        )
        
        return sample_characteristics
    
    # Initialize Opta integrator
    print(f"🔑 API Key found: {api_key[:8]}...")
    integrator = OptaDataIntegrator(api_key)
    
    try:
        # Fetch characteristics for current season
        print("📡 Fetching data from Opta API...")
        characteristics = integrator.get_all_league_characteristics("2024-25")
        
        # Save to file
        integrator.save_characteristics_to_file(characteristics)
        
        print("\n✅ Opta data successfully updated!")
        
        return characteristics
        
    except Exception as e:
        print(f"❌ Error fetching Opta data: {e}")
        print("📊 Falling back to estimated characteristics...")
        
        # Return estimated data as fallback
        fallback_characteristics = {
            'Premier League': {'intensity': 0.95, 'pace': 0.90, 'technical': 0.80, 'physicality': 0.95, 'pressing': 0.85},
            'La Liga': {'intensity': 0.75, 'pace': 0.70, 'technical': 0.95, 'physicality': 0.70, 'pressing': 0.80},
            'Serie A': {'intensity': 0.80, 'pace': 0.75, 'technical': 0.90, 'physicality': 0.85, 'pressing': 0.78},
            'Bundesliga': {'intensity': 0.88, 'pace': 0.92, 'technical': 0.85, 'physicality': 0.90, 'pressing': 0.90},
            'Ligue 1': {'intensity': 0.78, 'pace': 0.82, 'technical': 0.85, 'physicality': 0.80, 'pressing': 0.75}
        }
        
        return fallback_characteristics

def display_characteristics(characteristics):
    """Display characteristics in a formatted table"""
    
    print("\n📊 League Characteristics Summary")
    print("=" * 80)
    print(f"{'League':<15} {'Intensity':<10} {'Pace':<8} {'Technical':<10} {'Physical':<10} {'Pressing':<10}")
    print("-" * 80)
    
    for league, chars in characteristics.items():
        print(f"{league:<15} {chars['intensity']:<10.3f} {chars['pace']:<8.3f} "
              f"{chars['technical']:<10.3f} {chars['physicality']:<10.3f} {chars['pressing']:<10.3f}")

def compare_with_previous():
    """Compare current characteristics with previous ones"""
    
    current_file = "data/opta_league_characteristics.json"
    backup_file = "data/opta_league_characteristics_backup.json"
    
    if os.path.exists(current_file) and os.path.exists(backup_file):
        try:
            with open(current_file, 'r') as f:
                current = json.load(f)
            with open(backup_file, 'r') as f:
                previous = json.load(f)
            
            print("\n📈 Changes from previous update:")
            print("-" * 50)
            
            for league in current['characteristics']:
                if league in previous['characteristics']:
                    curr_chars = current['characteristics'][league]
                    prev_chars = previous['characteristics'][league]
                    
                    changes = []
                    for metric in curr_chars:
                        if metric in prev_chars:
                            diff = curr_chars[metric] - prev_chars[metric]
                            if abs(diff) > 0.01:  # Only show significant changes
                                changes.append(f"{metric}: {diff:+.3f}")
                    
                    if changes:
                        print(f"{league}: {', '.join(changes)}")
            
        except Exception as e:
            print(f"Error comparing with previous data: {e}")

if __name__ == "__main__":
    
    # Create backup of existing data
    current_file = "data/opta_league_characteristics.json"
    if os.path.exists(current_file):
        backup_file = "data/opta_league_characteristics_backup.json"
        os.makedirs("data", exist_ok=True)
        with open(current_file, 'r') as src, open(backup_file, 'w') as dst:
            dst.write(src.read())
    
    # Update characteristics
    characteristics = update_league_characteristics()
    
    # Display results
    display_characteristics(characteristics)
    
    # Compare with previous if available
    compare_with_previous()
    
    print(f"\n🎯 Next steps:")
    print("1. Run 'python train_model.py' to retrain with updated data")
    print("2. Test predictions with 'python app.py'")
    print("3. Set up automated updates (daily/weekly)")
    
    print(f"\n💡 Tip: For production use, get a real Opta API key")
    print("   Visit: https://www.optasports.com/services/sports-data/")