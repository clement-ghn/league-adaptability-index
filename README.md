# 🎯 LigueFit Index - League Adaptability Predictor

> *"Predicting player success across football leagues using machine learning"*

**LigueFit Index** is an AI-powered system that evaluates the probability of a football player succeeding when transferring between different leagues. Using advanced machine learning algorithms, it analyzes player performance, league characteristics, and historical transfer data to provide adaptability scores and risk assessments.

## 🌟 Key Features

- **🤖 ML-Powered Predictions**: XGBoost, Random Forest, and Logistic Regression models
- **📊 Interactive Dashboard**: Clean web interface built with Flask and Bootstrap
- **⚽ Real Player Analysis**: Test with examples like Hugo Ekitike, João Pedro, etc.
- **🏟️ Multi-League Support**: Premier League, La Liga, Serie A, Bundesliga, Ligue 1
- **📈 Risk Assessment**: Low/Medium/High risk classification with explanations
- **🔍 Feature Importance**: Understand what factors drive predictions

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/clement-ghn/league-adaptability-index.git
cd league-adaptability-index

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model

```bash
# Train and save the ML model
python train_model.py
```

### 3. Run the Web App

```bash
# Start the Flask application
python app.py
```

Visit `http://localhost:5000` to access the dashboard!

## 📊 How It Works

### The Science Behind LigueFit Index

1. **Data Collection**: Player stats, transfer data, league characteristics
2. **Feature Engineering**: Create 20+ features including:
   - Performance metrics (goals/90, xG, xA)
   - League transition difficulty
   - Age and position factors
   - Physical and tactical adaptability

3. **Machine Learning**: Train multiple models and select the best performer
4. **Prediction**: Generate adaptability scores (0-100%) with risk levels

### Example Prediction

```python
# Hugo Ekitike (Ligue 1 → Premier League)
{
    "adaptability_score": 45.2,
    "risk_level": "Medium Risk", 
    "recommendation": "Moderate risk transfer. Consider gradual integration.",
    "key_factors": ["age", "league_intensity_jump", "previous_performance"]
}
```

## 🏗️ Project Structure

```
league-adaptability-index/
├── src/                          # Core modules
│   ├── data_processor.py         # Data loading and processing
│   ├── feature_engineer.py       # Feature engineering pipeline
│   └── ml_model.py              # Machine learning models
├── templates/                    # Flask HTML templates
│   └── index.html               # Main dashboard
├── static/                      # CSS, JS, images
├── models/                      # Trained ML models
├── data/                        # Dataset files
├── notebooks/                   # Jupyter analysis
│   └── liguefit_index_exploration.ipynb
├── app.py                       # Flask web application
├── train_model.py              # Model training script
└── requirements.txt            # Dependencies
```

## 🔬 Technical Details

### Machine Learning Pipeline

- **Models Tested**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- **Best Performer**: XGBoost with F1-score of ~0.72
- **Features**: 20+ engineered features including league difficulty metrics
- **Validation**: Cross-validation with stratified sampling

### Key Features

| Feature Category | Examples |
|-----------------|----------|
| **Player Profile** | Age, position, playing time |
| **Performance** | Goals/90, assists/90, xG, xA |
| **League Characteristics** | Intensity, pace, physicality |
| **Transition Factors** | League difficulty jump, adaptation history |

### League Characteristics Matrix

| League | Intensity | Pace | Technical | Physical |
|--------|-----------|------|-----------|----------|
| **Premier League** | 0.95 | 0.90 | 0.80 | 0.95 |
| **La Liga** | 0.75 | 0.70 | 0.95 | 0.70 |
| **Serie A** | 0.80 | 0.75 | 0.90 | 0.85 |
| **Bundesliga** | 0.88 | 0.92 | 0.85 | 0.90 |
| **Ligue 1** | 0.78 | 0.82 | 0.85 | 0.80 |

## 📈 Example Results

### Success Stories
- **João Pedro** (Serie A → PL): 78% adaptability score ✅
- **Szoboszlai** (Bundesliga → PL): 71% adaptability score ✅

### Risk Cases  
- **Hugo Ekitike** (Ligue 1 → PL): 45% adaptability score ⚠️

## 🛠️ API Usage

### REST Endpoints

```bash
# Predict player adaptability
POST /predict
{
    "player_name": "Hugo Ekitike",
    "age": 21,
    "from_league": "Ligue 1",
    "to_league": "Premier League",
    "position": "Forward",
    "minutes": 1200,
    "goals": 8,
    "assists": 2,
    "xg": 7.5,
    "xa": 2.1
}
```

### Response Format

```json
{
    "adaptability_score": 45.2,
    "success_probability": 0.452,
    "risk_level": "Medium Risk",
    "league_transition": "Ligue 1 → Premier League",
    "recommendation": "Moderate risk transfer...",
    "feature_importance": {
        "age": 0.23,
        "intensity_jump": 0.19,
        "pre_goals_per_90": 0.15
    }
}
```

## 📚 Data Sources

- **Transfer Data**: Sample dataset with historical transfers
- **Player Stats**: Goals, assists, minutes, xG, xA metrics  
- **League Info**: Intensity, pace, technical level characteristics
- **Future Integration**: Transfermarkt, FBref, ESPN APIs

## 🎯 Roadmap

### Sprint 1 ✅ - MVP Complete
- [x] Data processing pipeline
- [x] Feature engineering
- [x] ML model training
- [x] Flask web application
- [x] Interactive dashboard

### Sprint 2 🚧 - Enhanced Features
- [ ] Real-time data integration
- [ ] More sophisticated league metrics
- [ ] Player comparison tools
- [ ] Historical success tracking

### Sprint 3 📋 - Production Ready
- [ ] Cloud deployment (Heroku/AWS)
- [ ] Database integration
- [ ] User authentication
- [ ] API rate limiting

### Sprint 4 🎨 - Advanced Analytics
- [ ] Seasonal adaptation tracking
- [ ] Team-specific factors
- [ ] Injury risk correlation
- [ ] Market value predictions

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🔧 Development Setup

```bash
# Development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Code formatting
black src/
flake8 src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Football data community for inspiration
- scikit-learn and XGBoost teams for excellent ML libraries
- Flask team for the lightweight web framework

## 📞 Contact

**Clement GHN** - [@clement-ghn](https://github.com/clement-ghn)

Project Link: [https://github.com/clement-ghn/league-adaptability-index](https://github.com/clement-ghn/league-adaptability-index)

---

*Made with ⚽ and 🤖 for football analytics enthusiasts*
