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

AI-StockBot-Pro/
├── app.py                      # Main Streamlit application
├── scripts/
│   ├── news_fetcher.py        # News and sentiment fetching
│   ├── gru_model.py           # GRU neural network model
│   ├── finbert_sentiment.py   # FinBERT sentiment analysis
│   └── predictor.py           # Random Forest predictor
├── .streamlit/
│   └── config.toml            # Streamlit configuration
├── requirements.txt           # Python dependencies
├── .gitignore                 # Git ignore rules
└── README.md        



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## ⚠️ Disclaimer

This tool is for educational and research purposes only. Do not use it for actual trading decisions. The predictions are not financial advice. Always do your own research and consult with financial professionals before making investment decisions.

## 👤 Author

**Khadija Sajid**
- FAST-NUCES Lahore
- BS Fintech Student

## 🙏 Acknowledgments

- Built with Streamlit
- Market data from Yahoo Finance (yfinance)
- News sentiment from NewsAPI
- AI models using TensorFlow and PyTorch

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ Star this repo if you find it helpful!
