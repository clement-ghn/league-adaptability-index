"""
Training script for the LigueFit Index model
Run this script to train and save the adaptability prediction model
"""

import sys
import os
import pandas as pd

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data_processor import DataProcessor
from feature_engineer import FeatureEngineer
from ml_model import AdaptabilityModel

# Paths in this script are relative to the project root, whatever the current directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Emoji output crashes on Windows consoles/pipes using cp1252
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Below this number of transfers the metrics are not meaningful (see the README)
MIN_SAMPLES_FOR_EVALUATION = 100


def main():
    """Main training pipeline"""
    os.makedirs('static', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    print("🚀 Starting LigueFit Index Model Training")
    print("=" * 50)

    # Step 1: Load and process data
    print("\n📊 Step 1: Loading and processing data...")
    processor = DataProcessor()
    try:
        raw = processor.load_transfer_dataset()
    except (FileNotFoundError, pd.errors.EmptyDataError):
        raw = pd.DataFrame()
    if len(raw) == 0:
        print("❌ No transfers in data/transfers_dataset.csv. Run 'python src/build_dataset.py' first.")
        sys.exit(1)
    if len(raw) < MIN_SAMPLES_FOR_EVALUATION:
        print(f"⚠️  Only {len(raw)} transfers (fewer than {MIN_SAMPLES_FOR_EVALUATION}): the metrics are not meaningful yet.")

    df = processor.prepare_dataset()
    processor.save_data(df)

    print(f"✅ Data loaded: {len(df)} samples with {len(df.columns)} features")
    print(f"📈 Success rate: {df['success'].mean():.1%}")
    
    # Step 2: Feature engineering
    print("\n🔧 Step 2: Feature engineering...")
    feature_engineer = FeatureEngineer()
    X, y = feature_engineer.fit_transform(df)
    feature_engineer.save()
    
    print(f"✅ Features engineered: {X.shape[1]} final features")
    print(f"📝 Feature names: {len(feature_engineer.get_feature_names())} total")
    
    # Step 3: Model training and evaluation
    print("\n🤖 Step 3: Training machine learning models...")
    model = AdaptabilityModel()
    results = model.train_and_evaluate(X, y, test_size=0.3)
    
    # Display results
    print(f"\n🏆 Best model: {model.best_model_name}")
    best_metrics = results[model.best_model_name]
    print(f"📊 Performance metrics:")
    print(f"   - Accuracy: {best_metrics['accuracy']:.3f}")
    print(f"   - Precision: {best_metrics['precision']:.3f}")
    print(f"   - Recall: {best_metrics['recall']:.3f}")
    print(f"   - F1-Score: {best_metrics['f1_score']:.3f}")
    print(f"   - ROC-AUC: {best_metrics['roc_auc']:.3f}")
    
    # Step 4: Save model
    print("\n💾 Step 4: Saving trained model...")
    model.save_model()
    
    # Step 5: Generate visualizations
    print("\n📈 Step 5: Generating visualizations...")
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode
        
        model.plot_model_comparison(save_path='static/model_comparison.png')
        model.plot_feature_importance(save_path='static/feature_importance.png')
        print("✅ Visualizations saved to static/ directory")
    except Exception as e:
        print(f"⚠️  Warning: Could not generate plots: {e}")
        print(f"   This is normal on some systems. Graphs will be skipped.")
    
    # Step 6: Generate model report
    print("\n📋 Step 6: Generating model report...")
    # Same held-out split as the metrics above
    report = model.generate_report(model.X_test, model.y_test)
    
    # Save report
    with open('models/model_report.txt', 'w') as f:
        f.write(report)
    
    print("✅ Model report saved to models/model_report.txt")
    
    print("\n🎉 Training completed successfully!")
    print("=" * 50)
    print("🚀 You can now run the Flask app with: python app.py")
    
    return model, feature_engineer, df


def test_prediction(model, feature_engineer):
    """Test the trained model with a sample prediction"""
    print("\n🧪 Testing model with sample prediction...")
    
    try:
        # Example profile (not a real player): checks that the pipeline runs end to end
        test_player = {
            'player_name': 'Example profile',
            'age': 24,
            'from_league': 'Bundesliga',
            'to_league': 'Premier League',
            'position': 'Midfielder',
            'minutes': 2000,
            'goals': 5,
            'assists': 4
        }
        
        # Process features
        features = feature_engineer.process_single_player(test_player)
        
        # Make prediction
        prediction, probability = model.predict_adaptability(features)
        
        print(f"✅ Test prediction successful!")
        print(f"   Player: {test_player['player_name']}")
        print(f"   Predicted success: {'Yes' if prediction else 'No'}")
        print(f"   Probability: {probability:.3f}")
        print(f"   Adaptability score: {probability * 100:.1f}%")
        
    except Exception as e:
        print(f"❌ Test prediction failed: {e}")
        print(f"   Error details: {str(e)}")
        
        # Try alternative test with saved models
        try:
            print("\n🔄 Trying with saved models...")
            saved_feature_engineer = FeatureEngineer.load()
            saved_model = AdaptabilityModel()
            saved_model.load_model()
            
            features = saved_feature_engineer.process_single_player(test_player)
            prediction, probability = saved_model.predict_adaptability(features)
            
            print(f"✅ Test with saved models successful!")
            print(f"   Player: {test_player['player_name']}")
            print(f"   Adaptability score: {probability * 100:.1f}%")
            
        except Exception as e2:
            print(f"❌ Saved model test also failed: {e2}")


if __name__ == "__main__":
    # Run training
    model, feature_engineer, df = main()
    
    # Test prediction with the trained models
    test_prediction(model, feature_engineer)
    
    print("\n🎯 Ready to use LigueFit Index!")