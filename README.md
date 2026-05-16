# AI-Stockbot-Pro
Advanced AI-Powered Stock Analysis & Prediction Platform using Machine Learning.

![AI StockBot Pro](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)

## 🌟 Features

- **Multi-Model AI Predictions**: GRU Neural Networks, Random Forest, and Ensemble methods
- **Real-Time News Sentiment Analysis**: FinBERT transformer models for market sentiment
- **Technical Analysis**: RSI, SMA, Bollinger Bands, Volume Analysis
- **Model Backtesting**: Test prediction accuracy on historical data
- **Interactive Charts**: Professional Plotly visualizations
- **5 Analysis Tools**: Prediction, News, Technical Analysis, Model Comparison, Backtesting

## 🛠️ Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/AI-StockBot-Pro.git
cd AI-StockBot-Pro
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Configuration
Create a `.streamlit/config.toml` file:
```toml
[theme]
base = "light"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
```

## 🚀 Usage

Run the Streamlit app:
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📊 How to Use

1. **Enter Stock Symbol**: Type any valid ticker (e.g., AAPL, TSLA, NVDA)
2. **Select Forecast Period**: Choose 3, 5, 7, 10, or 14 days
3. **Choose AI Model**: GRU Neural Network, Random Forest, or Ensemble
4. **Click Analysis Button**: Generate Prediction, News Analysis, Technical Analysis, Compare Models, or Backtest

## 🧠 AI Models

- **GRU Neural Network**: Deep learning model for sequential data
- **Random Forest**: Ensemble learning for robust predictions
- **Ensemble**: Combines multiple models for higher accuracy

## 📰 News Sentiment

Powered by NewsAPI and FinBERT sentiment analysis for real-time market sentiment scoring.

## 📈 Technical Indicators

- RSI (Relative Strength Index)
- SMA (Simple Moving Averages)
- Bollinger Bands
- Volume Analysis
- Support/Resistance Levels

## 🧪 Model Backtesting

Test prediction accuracy on recent historical data with:
- Mean Absolute Error (MAE)
- Mean Absolute Percentage Error (MAPE)
- Direction Accuracy
- Prediction Bias Analysis

## 🔑 API Keys

To use the News Analysis feature, you need a NewsAPI key:

1. Get your free API key from [NewsAPI](https://newsapi.org/)
2. Add it to `scripts/news_fetcher.py`:
```python
NEWSAPI_KEY = "your_api_key_here"
```

## 📁 Project Structure
