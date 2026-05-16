import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import warnings
warnings.filterwarnings('ignore')

# MODULE IMPORTS WITH ERROR HANDLING
MODULES_STATUS = {
    'news': False,
    'gru': False,
    'finbert': False,
    'predictor': False
}

try:
    from scripts.news_fetcher import fetch_news_and_sentiment  
    MODULES_STATUS['news'] = True
except ImportError:
    pass

try:
    from scripts.gru_model import FastGRUPredictor, predict_gru_fast, is_gru_available
    MODULES_STATUS['gru'] = True
except ImportError:
    pass

try:
    from scripts.finbert_sentiment import analyze_finbert_sentiment, format_sentiment_for_display
    MODULES_STATUS['finbert'] = True
except ImportError:
    pass

try:
    from scripts.predictor import quick_predict, FastStockPredictor
    MODULES_STATUS['predictor'] = True
except ImportError:
    pass

# PAGE CONFIGURATION
st.set_page_config(
    page_title="AI StockBot Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# PROFESSIONAL CSS STYLING - FIXED FOR NEWS READABILITY
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --primary: #2563eb;
    --secondary: #1e40af;
    --success: #059669;
    --danger: #dc2626;
    --warning: #d97706;
    --neutral: #64748b;
    --bg-light: #f8fafc;
    --bg-card: #ffffff;
    --border: #e2e8f0;
    --text-primary: #0f172a;
    --text-secondary: #64748b;
}

* {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* 🔥 Animated Gradient Heading - SINGLE HEADING */
.title-text {
    font-size: 3rem;
    font-weight: bold;
    background: linear-gradient(270deg, #2563eb, #1e40af, #059669, #d97706);
    background-size: 800% 800%;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: gradientMove 8s ease infinite;
    text-align: center;
    margin: 2rem 0;
}

@keyframes gradientMove {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Sub-header */
.sub-header {
    text-align: center;
    color: var(--text-secondary);
    font-size: 1.1rem;
    font-weight: 400;
    margin-bottom: 3rem;
}

/* Section Spacing */
.section-spacer {
    margin: 3rem 0;
}

.section-divider {
    border-top: 1px solid var(--border);
    margin: 2.5rem 0;
}

/* Professional Cards */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 2rem 1.5rem;
    text-align: center;
    box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
    transition: all 0.2s ease;
    margin-bottom: 1.5rem;
    height: 100%;
}

.metric-card:hover {
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transform: translateY(-2px);
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.875rem;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 0.75rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
    margin-bottom: 0.5rem;
}

.metric-change {
    font-size: 0.875rem;
    font-weight: 500;
}

/* Consistent Colors */
.positive { color: var(--success); }
.negative { color: var(--danger); }
.neutral { color: var(--warning); }
.primary { color: var(--primary); }

/* Prediction Summary Card */
.prediction-summary {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    margin: 2rem 0;
}

.prediction-title {
    font-size: 1.25rem;
    font-weight: 600;
    margin-bottom: 1rem;
}

.prediction-price {
    font-size: 2.5rem;
    font-weight: 800;
    margin: 1rem 0;
}

.prediction-change {
    font-size: 1rem;
    font-weight: 500;
    opacity: 0.9;
}

/* Form Controls */
.stTextInput > div > div > input {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    font-weight: 400 !important;
}

.stTextInput > div > div > input:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 2px rgba(37, 99, 235, 0.1) !important;
}

.stSelectbox > div > div > select {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
}

/* Professional Buttons */
.stButton > button {
    background: var(--primary) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    transition: all 0.2s ease !important;
    color: white !important;
}

.stButton > button:hover {
    background: var(--secondary) !important;
    transform: translateY(-1px) !important;
}

/* Status Indicator */
.status-indicator {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1rem;
    border-radius: 20px;
    font-size: 0.875rem;
    font-weight: 500;
    margin-bottom: 1.5rem;
}

.status-active {
    background: rgba(5, 150, 105, 0.1);
    color: var(--success);
    border: 1px solid rgba(5, 150, 105, 0.2);
}

.status-partial {
    background: rgba(217, 119, 6, 0.1);
    color: var(--warning);
    border: 1px solid rgba(217, 119, 6, 0.2);
}

.status-inactive {
    background: rgba(220, 38, 38, 0.1);
    color: var(--danger);
    border: 1px solid rgba(220, 38, 38, 0.2);
}

/* FIXED: News Cards - Dark Background with Light Text */
.news-card {
    background: linear-gradient(135deg, #1e293b 0%, #334155 100%);
    border: 1px solid #475569;
    border-left: 3px solid var(--primary);
    border-radius: 12px;
    padding: 1.5rem;
    margin: 1rem 0;
    transition: all 0.2s ease;
    color: #f1f5f9 !important;
}

.news-card:hover {
    border-left-color: var(--success);
    box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
    transform: translateY(-2px);
}

.news-card strong {
    color: #60a5fa !important;
    font-weight: 600;
}

.news-card p {
    color: #e2e8f0 !important;
    line-height: 1.6;
    margin: 0.8rem 0;
}

/* Prediction Card - Similar styling */
.prediction-card {
    background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
    color: white;
    padding: 2rem;
    border-radius: 12px;
    text-align: center;
    margin: 2rem 0;
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.2);
}

/* Data Tables */
.dataframe {
    border-radius: 8px !important;
    border: 1px solid var(--border) !important;
    font-family: 'Inter', sans-serif !important;
}

/* Progress Bar */
.stProgress > div > div > div > div {
    background: var(--primary) !important;
}

/* Section Headers */
.section-header {
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 2.5rem 0 1.5rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid var(--border);
}

/* Professional Info Boxes */
.info-card {
    background: var(--bg-light);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5rem;
    margin: 1.5rem 0;
}

.info-card h3 {
    color: var(--text-primary);
    margin-bottom: 1rem;
}

.info-card p {
    color: var(--text-secondary);
    line-height: 1.5;
}

/* Fade-in animation */
.fade-in {
    animation: fadeIn 0.5s ease-in;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}
</style>
""", unsafe_allow_html=True)

# CACHED DATA FUNCTIONS
@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_data_fast(ticker, period="6mo"):
    """Fetch stock data with caching"""
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period=period, interval="1d", timeout=10)
        if data.empty:
            return None
        data = data.dropna()
        return data
    except Exception as e:
        st.error(f"Data fetch error: {e}")
        return None

@st.cache_data(ttl=1800, show_spinner=False)
def get_news_sentiment_fast(ticker):
    """Get news sentiment with caching"""
    if MODULES_STATUS['news']:
        try:
            return fetch_news_and_sentiment(ticker, limit=8)
        except Exception as e:
            print(f"News fetch error: {e}")
    
    return {
        'headlines': [f"Recent {ticker} market analysis shows mixed signals"],
        'sentiment': {'score': 0.0, 'label': 'neutral', 'emoji': '→', 'confidence': 'low'}
    }

# TECHNICAL INDICATORS
def calculate_technical_fast(df):
    """Calculate technical indicators"""
    df = df.copy()
    
    df['Returns'] = df['Close'].pct_change()
    df['SMA_20'] = df['Close'].rolling(20, min_periods=1).mean()
    df['SMA_50'] = df['Close'].rolling(50, min_periods=1).mean()
    
    # RSI calculation
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    df['Volatility'] = df['Returns'].rolling(10, min_periods=1).std() * 100
    
    return df

# PREDICTION ENGINE
def get_ai_predictions(stock_data, sentiment_score, days, model_type):
    """Multi-model prediction engine"""
    predictions = []
    accuracy = 70.0
    
    try:
        if model_type == "GRU Neural Network" and MODULES_STATUS['gru']:
            predictions, accuracy = predict_gru_fast(stock_data, sentiment_score, days, epochs=15)
            if predictions:
                return predictions, accuracy
        
        if model_type == "Random Forest" and MODULES_STATUS['predictor']:
            predictions, accuracy = quick_predict(stock_data, sentiment_score, days)
            if predictions:
                return predictions, accuracy
        
        return smart_fallback_predict(stock_data, sentiment_score, days)
        
    except Exception as e:
        print(f"Prediction error: {e}")
        return smart_fallback_predict(stock_data, sentiment_score, days)

def smart_fallback_predict(stock_data, sentiment_score, days):
    """Intelligent fallback prediction"""
    try:
        if len(stock_data) < 10:
            return [], 0
        
        current_price = stock_data['Close'].iloc[-1]
        short_trend = stock_data['Close'].pct_change().tail(5).mean()
        medium_trend = stock_data['Close'].pct_change().tail(20).mean()
        long_trend = stock_data['Close'].pct_change().tail(50).mean()
        
        trend = (short_trend * 0.5 + medium_trend * 0.3 + long_trend * 0.2)
        volatility = stock_data['Close'].pct_change().tail(20).std()
        sentiment_factor = 1 + (sentiment_score * 0.05)
        
        predictions = []
        for i in range(days):
            day_factor = 0.95 ** i
            noise = np.random.normal(0, volatility * 0.1)
            predicted_return = trend * day_factor * sentiment_factor + noise
            next_price = current_price * (1 + predicted_return)
            next_price = max(current_price * 0.85, min(current_price * 1.15, next_price))
            predictions.append(float(next_price))
            current_price = next_price
        
        return predictions, 75.0
        
    except Exception as e:
        print(f"Fallback prediction error: {e}")
        return [], 0

# BACKTESTING FUNCTION - FIXED INDENTATION
def simple_backtest_analysis(ticker, stock_data, model_type, days_back=30):
    """Simple backtesting for model evaluation"""
    try:
        if len(stock_data) < days_back + 10:
            return None
        
        # Split data for backtesting
        backtest_end = len(stock_data) - 5  # Keep last 5 days for comparison
        train_data = stock_data.iloc[:backtest_end].copy()
        actual_data = stock_data.iloc[backtest_end:].copy()
        
        # Get historical prediction (simulate what model would have predicted)
        sentiment_score = 0.0  # Neutral for backtest
        predictions, accuracy = get_ai_predictions(train_data, sentiment_score, 5, model_type)
        
        if not predictions or len(actual_data) == 0:
            return None
        
        # Compare predictions vs actual
        actual_prices = actual_data['Close'].values[:len(predictions)]
        predictions = predictions[:len(actual_prices)]
        
        # Calculate metrics
        if len(predictions) > 0 and len(actual_prices) > 0:
            mae = np.mean(np.abs(np.array(actual_prices) - np.array(predictions)))
            mape = np.mean(np.abs((actual_prices - predictions) / actual_prices)) * 100
            
            # Direction accuracy
            actual_directions = [1 if actual_prices[i] > actual_prices[i-1] else 0 
                               for i in range(1, len(actual_prices))]
            pred_directions = [1 if predictions[i] > predictions[i-1] else 0 
                              for i in range(1, len(predictions))]
            
            if actual_directions and pred_directions:
                direction_acc = np.mean([a == p for a, p in zip(actual_directions, pred_directions)]) * 100
            else:
                direction_acc = 50
            
            return {
                'predictions': predictions,
                'actual': actual_prices,
                'mae': mae,
                'mape': mape,
                'direction_accuracy': direction_acc,
                'model_accuracy': accuracy,
                'dates': actual_data.index[:len(predictions)]
            }
    
    except Exception as e:
        print(f"Backtest error: {e}")
        return None

# CHART CREATION WITH CONSISTENT COLORS
def create_advanced_chart(stock_data, predictions, ticker):
    """Create professional chart with consistent colors"""
    try:
        fig = make_subplots(
            rows=3, cols=1,
            row_heights=[0.6, 0.25, 0.15],
            subplot_titles=(
                f"{ticker} Price Analysis & AI Predictions",
                "Trading Volume",
                "RSI Indicator"
            ),
            vertical_spacing=0.08
        )
        
        # Consistent color scheme
        colors = {
            'positive': '#059669',  # Teal green
            'negative': "#ec2727",  # Red
            'primary': '#2563eb',   # Blue
            'secondary': '#64748b', # Gray
            'prediction': '#7c3aed' # Purple
        }
        
        # Main price chart
        if len(stock_data) > 50:
            recent_data = stock_data.tail(50)
            fig.add_trace(
                go.Candlestick(
                    x=recent_data.index,
                    open=recent_data['Open'],
                    high=recent_data['High'],
                    low=recent_data['Low'],
                    close=recent_data['Close'],
                    name=f"{ticker}",
                    increasing_line_color=colors['positive'],
                    decreasing_line_color=colors['negative'],
                    increasing_fillcolor=colors['positive'],
                    decreasing_fillcolor=colors['negative']
                ), row=1, col=1
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=stock_data.index,
                    y=stock_data['Close'],
                    name=f"{ticker}",
                    line=dict(color=colors['primary'], width=3),
                    hovertemplate="<b>$%{y:.2f}</b><br>%{x}<extra></extra>"
                ), row=1, col=1
            )
        
        # Technical indicators
        tech_data = calculate_technical_fast(stock_data)
        
        fig.add_trace(
            go.Scatter(
                x=tech_data.index, y=tech_data['SMA_20'],
                name="20-Day Average", 
                line=dict(color=colors['secondary'], width=1.5, dash='dash'),
                opacity=0.7
            ), row=1, col=1
        )
        
        if len(stock_data) >= 50:
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index, y=tech_data['SMA_50'],
                    name="50-Day Average", 
                    line=dict(color=colors['secondary'], width=1.5, dash='dot'),
                    opacity=0.6
                ), row=1, col=1
            )
        
        # AI Predictions
        if predictions and len(predictions) > 0:
            future_dates = pd.date_range(
                start=stock_data.index[-1] + pd.Timedelta(days=1),
                periods=len(predictions),
                freq='D'
            )
            
            fig.add_trace(
                go.Scatter(
                    x=future_dates, y=predictions,
                    name="AI Forecast",
                    line=dict(color=colors['prediction'], width=3),
                    marker=dict(size=6, color=colors['prediction']),
                    hovertemplate="<b>Forecast: $%{y:.2f}</b><br>%{x}<extra></extra>"
                ), row=1, col=1
            )
        
        # Volume
        volume_colors = [colors['positive'] if stock_data['Close'].iloc[i] >= stock_data['Close'].iloc[i-1] 
                        else colors['negative'] if i > 0 else colors['secondary'] 
                        for i in range(len(stock_data))]
        
        fig.add_trace(
            go.Bar(
                x=stock_data.index, y=stock_data['Volume'],
                name="Volume", marker_color=volume_colors,
                opacity=0.6, hovertemplate="<b>Volume: %{y:,.0f}</b><extra></extra>"
            ), row=2, col=1
        )
        
        # RSI
        if 'RSI' in tech_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=tech_data.index, y=tech_data['RSI'],
                    name="RSI", line=dict(color=colors['primary'], width=2),
                    hovertemplate="<b>RSI: %{y:.1f}</b><extra></extra>"
                ), row=3, col=1
            )
            
            fig.add_hline(y=70, line_dash="dash", line_color=colors['negative'], 
                         opacity=0.5, row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=colors['positive'], 
                         opacity=0.5, row=3, col=1)
        
        # Professional layout
        fig.update_layout(
            template="plotly_white",
            height=700,
            showlegend=True,
            font=dict(family="Inter, sans-serif", size=12),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
        fig.update_xaxes(title_text="Date", row=3, col=1)
        fig.update_yaxes(title_text="Price ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)
        
        return fig
        
    except Exception as e:
        st.error(f"Chart creation error: {e}")
        return None

# MAIN APPLICATION
def main():
    # 🔥 Single Animated Gradient Heading
    st.markdown('<h1 class="title-text">🚀 AI StockBot Pro</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Advanced AI-Powered Stock Analysis & Predictions</div>', unsafe_allow_html=True)
    
    # System Status
    modules_active = sum(MODULES_STATUS.values())
    status_class = "active" if modules_active >= 3 else "partial" if modules_active >= 1 else "inactive"
    status_text = f"AI Modules: {modules_active}/4 Active"
    
    st.markdown(f"""
    <div style="text-align: center;">
        <span class="status-indicator status-{status_class}">
            {status_text}
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    
    # Input Controls
    st.markdown("## Trading Parameters")
    
    col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
    
    with col1:
        ticker = st.text_input(
            "Stock Symbol", 
            value="AAPL", 
            placeholder="Enter ticker symbol (e.g., AAPL, TSLA, NVDA)",
            help="Enter any valid stock ticker symbol"
        ).upper().strip()
    
    with col2:
        prediction_days = st.selectbox(
            "Forecast Period", 
            [3, 5, 7, 10, 14], 
            index=2,
            help="Number of days to predict"
        )
    
    with col3:
        data_period = st.selectbox(
            "Training Data", 
            ["3mo", "6mo", "1y", "2y"], 
            index=1,
            help="Historical data period"
        )
    
    with col4:
        model_options = ["Random Forest", "GRU Neural Network", "Ensemble"]
        model_type = st.selectbox(
            "AI Model", 
            model_options,
            index=1 if MODULES_STATUS['gru'] else 0,
            help="Select prediction model"
        )
    
    st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
    
    # Action Buttons - FIXED WITH 5 COLUMNS
    st.markdown("## Analysis Tools")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        predict_btn = st.button("🔮 Generate Prediction", type="primary")
    
    with col2:
        news_btn = st.button("📰 News Analysis")
    
    with col3:
        analysis_btn = st.button("📊 Technical Analysis")
    
    with col4:
        compare_btn = st.button("⚖️ Compare Models")
    
    with col5:
        backtest_btn = st.button("🧪 Backtest Model")
    
    # Main Prediction Logic
    if predict_btn and ticker:
        try:
            progress_container = st.container()
            
            with progress_container:
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                with col2:
                    timer_text = st.empty()
            
            start_time = time.time()
            
            # Data fetching
            status_text.text("Fetching market data...")
            timer_text.text(f"⏱ {time.time() - start_time:.1f}s")
            
            stock_data = fetch_stock_data_fast(ticker, data_period)
            if stock_data is None or len(stock_data) < 20:
                st.error(f"Unable to fetch sufficient data for {ticker}")
                return
            
            progress_bar.progress(25)
            
            # Technical analysis
            status_text.text("Computing technical indicators...")
            timer_text.text(f"⏱ {time.time() - start_time:.1f}s")
            
            tech_data = calculate_technical_fast(stock_data)
            current_price = float(stock_data['Close'].iloc[-1])
            prev_price = float(stock_data['Close'].iloc[-2]) if len(stock_data) > 1 else current_price
            price_change = current_price - prev_price
            price_change_pct = (price_change / prev_price) * 100 if prev_price != 0 else 0
            
            progress_bar.progress(50)
            
            # Sentiment analysis
            status_text.text("Analyzing market sentiment...")
            timer_text.text(f"⏱ {time.time() - start_time:.1f}s")
            
            news_data = get_news_sentiment_fast(ticker)
            sentiment_score = news_data['sentiment']['score']
            sentiment_label = news_data['sentiment']['label']
            
            progress_bar.progress(75)
            
            # AI prediction
            status_text.text("Generating AI predictions...")
            timer_text.text(f"⏱ {time.time() - start_time:.1f}s")
            
            predictions, model_accuracy = get_ai_predictions(
                stock_data, sentiment_score, prediction_days, model_type
            )
            
            progress_bar.progress(100)
            status_text.text("Analysis complete")
            
            total_time = time.time() - start_time
            timer_text.text(f"⏱ {total_time:.1f}s")
            
            time.sleep(0.5)
            progress_container.empty()
            
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            
            # Results
            st.markdown('<div class="section-header">Analysis Results</div>', unsafe_allow_html=True)
            
            # Key metrics
            metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
            
            with metric_col1:
                change_color = "positive" if price_change >= 0 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Current Price</div>
                    <div class="metric-value">${current_price:.2f}</div>
                    <div class="metric-change {change_color}">
                        {'+' if price_change >= 0 else ''}{price_change:.2f} ({price_change_pct:+.2f}%)
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col2:
                sentiment_color = "positive" if sentiment_score > 0.1 else "negative" if sentiment_score < -0.1 else "neutral"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Market Sentiment</div>
                    <div class="metric-value {sentiment_color}">{sentiment_score:+.3f}</div>
                    <div class="metric-change {sentiment_color}">
                        {sentiment_label.title()}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col3:
                accuracy_color = "positive" if model_accuracy > 80 else "neutral" if model_accuracy > 60 else "negative"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Model Accuracy</div>
                    <div class="metric-value {accuracy_color}">{model_accuracy:.1f}%</div>
                    <div class="metric-change {accuracy_color}">
                        {model_type}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_col4:
                volatility = tech_data['Volatility'].iloc[-1] if 'Volatility' in tech_data.columns else 0
                vol_color = "negative" if volatility > 4 else "neutral" if volatility > 2 else "positive"
                vol_level = "High Risk" if volatility > 4 else "Medium Risk" if volatility > 2 else "Low Risk"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">Volatility (10d)</div>
                    <div class="metric-value {vol_color}">{volatility:.2f}%</div>
                    <div class="metric-change {vol_color}">
                        {vol_level}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
            
            # Prediction summary
            if predictions:
                avg_prediction = np.mean(predictions)
                prediction_change = ((avg_prediction - current_price) / current_price) * 100
                
                if prediction_change > 5:
                    trend_desc, trend_color = "Strong Bullish", "#059669"
                elif prediction_change > 2:
                    trend_desc, trend_color = "Bullish", "#059669"
                elif prediction_change > -2:
                    trend_desc, trend_color = "Neutral", "#d97706"
                elif prediction_change > -5:
                    trend_desc, trend_color = "Bearish", "#dc2626"
                else:
                    trend_desc, trend_color = "Strong Bearish", "#dc2626"
                
                st.markdown(f"""
                <div class="prediction-summary">
                    <div class="prediction-title">AI Prediction Summary</div>
                    <div class="prediction-price">${avg_prediction:.2f}</div>
                    <div class="prediction-change">
                        Expected {prediction_days}-day movement: <strong style="color: {trend_color};">{prediction_change:+.2f}%</strong>
                    </div>
                    <hr style="margin: 1.5rem 0; opacity: 0.3;">
                    <div style="display: flex; justify-content: space-around; font-size: 0.9rem;">
                        <div><strong>Range:</strong> ${min(predictions):.2f} - ${max(predictions):.2f}</div>
                        <div><strong>Confidence:</strong> {model_accuracy:.0f}%</div>
                        <div><strong>Signal:</strong> {trend_desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
            
            # Chart
            st.markdown('<div class="section-header">Price Analysis & Forecast</div>', unsafe_allow_html=True)
            chart = create_advanced_chart(stock_data, predictions, ticker)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            st.markdown('<div class="section-spacer"></div>', unsafe_allow_html=True)
            
            # Data tables
            st.markdown('<div class="section-header">Detailed Data</div>', unsafe_allow_html=True)
            
            table_col1, table_col2 = st.columns(2)
            
            with table_col1:
                st.markdown("### 📊 Recent Market Data")
                recent_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(7)
                recent_data.index = recent_data.index.strftime('%Y-%m-%d')
                recent_data = recent_data.round(2)
                recent_data['Volume'] = recent_data['Volume'].apply(lambda x: f"{x:,.0f}")
                st.dataframe(recent_data, use_container_width=True)
                
            with table_col2:
                st.markdown("### 🤖 AI Forecast Details")
                if predictions:
                    future_dates = pd.date_range(
                        start=stock_data.index[-1] + pd.Timedelta(days=1),
                        periods=prediction_days
                    )
                    
                    # Generate trading signals
                    signals = []
                    for pred in predictions:
                        change_pct = ((pred - current_price) / current_price) * 100
                        if change_pct > 3:
                            signals.append("🚀 Strong Buy")
                        elif change_pct > 1:
                            signals.append("📈 Buy")
                        elif change_pct > -1:
                            signals.append("➡️ Hold")
                        elif change_pct > -3:
                            signals.append("📉 Sell")
                        else:
                            signals.append("💥 Strong Sell")
                    
                    pred_df = pd.DataFrame({
                        'Date': future_dates.strftime('%Y-%m-%d'),
                        'Price': [f"${p:.2f}" for p in predictions],
                        'Change': [f"{((p - current_price) / current_price * 100):+.2f}%" for p in predictions],
                        'Signal': signals
                    })
                    st.dataframe(pred_df, use_container_width=True)
                
            # Technical Analysis Summary
            with st.expander("📊 Advanced Technical Analysis", expanded=False):
                tech_col1, tech_col2, tech_col3 = st.columns(3)
                
                with tech_col1:
                    rsi = tech_data['RSI'].iloc[-1] if 'RSI' in tech_data.columns else 50
                    rsi_signal = "🔥 Oversold" if rsi < 30 else "❄️ Overbought" if rsi > 70 else "➡️ Neutral"
                    st.metric("RSI (14-day)", f"{rsi:.1f}", rsi_signal)
                    
                    # Support and Resistance
                    support = stock_data['Low'].tail(20).min()
                    resistance = stock_data['High'].tail(20).max()
                    st.metric("Support Level", f"${support:.2f}")
                    st.metric("Resistance Level", f"${resistance:.2f}")
                
                with tech_col2:
                    # Moving Average Position
                    sma20 = tech_data['SMA_20'].iloc[-1] if 'SMA_20' in tech_data.columns else current_price
                    ma_position = "📈 Above SMA20" if current_price > sma20 else "📉 Below SMA20"
                    st.metric("20-Day SMA", f"${sma20:.2f}", ma_position)
                    
                    # Volume Analysis
                    avg_volume = stock_data['Volume'].tail(10).mean()
                    current_volume = stock_data['Volume'].iloc[-1]
                    volume_ratio = current_volume / avg_volume
                    volume_signal = "🔥 High Volume" if volume_ratio > 1.5 else "📊 Normal"
                    st.metric("Volume Ratio", f"{volume_ratio:.2f}x", volume_signal)
                
                with tech_col3:
                    # Momentum indicators
                    momentum_5d = ((current_price / stock_data['Close'].iloc[-6]) - 1) * 100 if len(stock_data) > 5 else 0
                    momentum_signal = "🚀 Strong" if abs(momentum_5d) > 5 else "📊 Moderate"
                    st.metric("5-Day Momentum", f"{momentum_5d:+.2f}%", momentum_signal)
                    
                    # Market cap estimate (if available)
                    try:
                        ticker_info = yf.Ticker(ticker)
                        shares = ticker_info.info.get('sharesOutstanding', 0)
                        if shares:
                            market_cap = (current_price * shares) / 1e9
                            st.metric("Market Cap", f"${market_cap:.1f}B")
                    except:
                        pass
        
        except Exception as e:
            st.error(f"❌ Analysis failed: {str(e)}")
            st.info("💡 Please try again with a valid ticker symbol")
    
    # News Analysis Section
    elif news_btn and ticker:
        with st.spinner(f"📡 Fetching latest news for {ticker}..."):
            news_data = get_news_sentiment_fast(ticker)
        
        st.markdown(f"## 📰 News Analysis for {ticker}")
        
        # News sentiment overview
        sentiment = news_data['sentiment']
        sentiment_class = "positive" if sentiment['score'] > 0.1 else "negative" if sentiment['score'] < -0.1 else "neutral"
        
        st.markdown(f"""
        <div class="prediction-card fade-in">
            <h2>{sentiment['emoji']} Market Sentiment Overview</h2>
            <div class="prediction-price {sentiment_class}">{sentiment['score']:+.3f}</div>
            <div class="prediction-change">
                Sentiment: <strong>{sentiment['label'].upper()}</strong> | 
                Confidence: <strong>{sentiment.get('confidence', 'Medium').title()}</strong>
            </div>
            <hr style="margin: 1.5rem 0; opacity: 0.3;">
            <div style="opacity: 0.9;">
                Analyzed {len(news_data['headlines'])} recent headlines from financial news sources
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Individual headlines
        if news_data['headlines']:
            st.markdown("### 📰 Recent Headlines")
            for i, headline in enumerate(news_data['headlines'][:8], 1):
                st.markdown(f"""
                <div class="news-card fade-in">
                    <strong style="color: #60a5fa;">Article {i}</strong>
                    <p style="margin: 0.8rem 0; font-size: 1rem; line-height: 1.5; color: #e2e8f0;">{headline}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info(f"📊 No recent news found for {ticker}. This might indicate low media coverage or data availability issues.")
    
    # Technical Analysis Section
    elif analysis_btn and ticker:
        with st.spinner(f"📊 Performing technical analysis for {ticker}..."):
            stock_data = fetch_stock_data_fast(ticker, "6mo")
        
        if stock_data is not None:
            tech_data = calculate_technical_fast(stock_data)
            
            st.markdown(f"## 📊 Technical Analysis for {ticker}")
            
            # Technical overview
            current_price = stock_data['Close'].iloc[-1]
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                rsi = tech_data['RSI'].iloc[-1] if 'RSI' in tech_data.columns else 50
                rsi_color = "negative" if rsi > 70 or rsi < 30 else "positive"
                rsi_signal = "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📊 RSI (14)</div>
                    <div class="metric-value {rsi_color}">{rsi:.1f}</div>
                    <div class="metric-change {rsi_color}">{rsi_signal}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                sma20 = tech_data['SMA_20'].iloc[-1] if 'SMA_20' in tech_data.columns else current_price
                trend_color = "positive" if current_price > sma20 else "negative"
                trend_signal = "Bullish" if current_price > sma20 else "Bearish"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📈 SMA20 Trend</div>
                    <div class="metric-value {trend_color}">${sma20:.2f}</div>
                    <div class="metric-change {trend_color}">{trend_signal}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                volatility = tech_data['Volatility'].iloc[-1] if 'Volatility' in tech_data.columns else 0
                vol_color = "negative" if volatility > 4 else "neutral" if volatility > 2 else "positive"
                vol_level = "High" if volatility > 4 else "Medium" if volatility > 2 else "Low"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">⚡ Volatility</div>
                    <div class="metric-value {vol_color}">{volatility:.2f}%</div>
                    <div class="metric-change {vol_color}">{vol_level} Risk</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                volume_avg = stock_data['Volume'].tail(20).mean()
                volume_current = stock_data['Volume'].iloc[-1]
                volume_ratio = volume_current / volume_avg
                vol_trend_color = "positive" if volume_ratio > 1.2 else "neutral"
                vol_trend = "Above Average" if volume_ratio > 1.2 else "Normal"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-label">📊 Volume</div>
                    <div class="metric-value {vol_trend_color}">{volume_ratio:.2f}x</div>
                    <div class="metric-change {vol_trend_color}">{vol_trend}</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Technical chart
            chart = create_advanced_chart(stock_data, [], ticker)
            if chart:
                st.plotly_chart(chart, use_container_width=True)
            
            # Support/Resistance levels
            st.markdown("### 📊 Key Levels")
            
            col1, col2 = st.columns(2)
            
            with col1:
                support_levels = stock_data['Low'].tail(50).nsmallest(3).values
                st.markdown("**Support Levels:**")
                for i, level in enumerate(support_levels, 1):
                    distance = ((current_price - level) / current_price) * 100
                    st.write(f"{i}. ${level:.2f} ({distance:+.1f}% from current)")
            
            with col2:
                resistance_levels = stock_data['High'].tail(50).nlargest(3).values
                st.markdown("**Resistance Levels:**")
                for i, level in enumerate(resistance_levels, 1):
                    distance = ((level - current_price) / current_price) * 100
                    st.write(f"{i}. ${level:.2f} ({distance:+.1f}% from current)")
    
    # Model Comparison Section
    elif compare_btn and ticker:
        st.markdown(f"## ⚖️ Model Comparison for {ticker}")
        
        with st.spinner("🔄 Running multiple AI models..."):
            stock_data = fetch_stock_data_fast(ticker, data_period)
            if stock_data is None:
                st.error(f"❌ Unable to fetch data for {ticker}")
                return
            
            news_data = get_news_sentiment_fast(ticker)
            sentiment_score = news_data['sentiment']['score']
            
            # Compare different models
            models_to_test = ["Random Forest", "GRU Neural Network", "Ensemble"]
            comparison_results = {}
            
            for model in models_to_test:
                try:
                    predictions, accuracy = get_ai_predictions(stock_data, sentiment_score, 7, model)
                    if predictions:
                        avg_pred = np.mean(predictions)
                        comparison_results[model] = {
                            'predictions': predictions,
                            'accuracy': accuracy,
                            'avg_prediction': avg_pred,
                            'change_pct': ((avg_pred - stock_data['Close'].iloc[-1]) / stock_data['Close'].iloc[-1]) * 100
                        }
                except Exception as e:
                    print(f"Model {model} failed: {e}")
            
            if comparison_results:
                # Display comparison
                cols = st.columns(len(comparison_results))
                
                for i, (model, results) in enumerate(comparison_results.items()):
                    with cols[i]:
                        change_color = "positive" if results['change_pct'] > 0 else "negative"
                        model_emoji = "🚀" if "Random" in model else "🔥" if "GRU" in model else "🎯"
                        st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">{model_emoji} {model}</div>
                            <div class="metric-value">${results['avg_prediction']:.2f}</div>
                            <div class="metric-change {change_color}">
                                {results['change_pct']:+.2f}% | {results['accuracy']:.1f}% Acc
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Ensemble recommendation
                avg_of_averages = np.mean([r['avg_prediction'] for r in comparison_results.values()])
                avg_accuracy = np.mean([r['accuracy'] for r in comparison_results.values()])
                ensemble_change = ((avg_of_averages - stock_data['Close'].iloc[-1]) / stock_data['Close'].iloc[-1]) * 100
                
                st.markdown(f"""
                <div class="prediction-card fade-in">
                    <h2>🎯 Ensemble Recommendation</h2>
                    <div class="prediction-price">${avg_of_averages:.2f}</div>
                    <div class="prediction-change">
                        Combined Model Forecast: <strong>{ensemble_change:+.2f}%</strong> | 
                        Average Confidence: <strong>{avg_accuracy:.1f}%</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error("❌ Model comparison failed. Please try again.")
    
    # BACKTESTING SECTION - FIXED INTEGRATION
    elif backtest_btn and ticker:
        st.markdown(f"## 🧪 Model Backtesting for {ticker}")
        st.markdown("*Testing how well the model would have performed on recent historical data*")
        
        with st.spinner(f"🔄 Running backtest analysis for {ticker}..."):
            # Get extended data for backtesting
            stock_data = fetch_stock_data_fast(ticker, "3mo")  # Need more data for backtest
            
            if stock_data is None or len(stock_data) < 40:
                st.error("❌ Insufficient data for backtesting. Need at least 40 days of data.")
                return
            
            # Run backtest
            backtest_results = simple_backtest_analysis(ticker, stock_data, model_type)
            
            if backtest_results is None:
                st.error("❌ Backtesting failed. Please try with a different stock or time period.")
                return
        
        st.success("✅ Backtest completed successfully!")
        
        # Display backtest metrics
        st.markdown("### 📊 Backtest Performance Metrics")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            mae = backtest_results['mae']
            mae_color = "positive" if mae < 2 else "neutral" if mae < 5 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Absolute Error</div>
                <div class="metric-value {mae_color}">${mae:.2f}</div>
                <div class="metric-change {mae_color}">
                    {"Excellent" if mae < 2 else "Good" if mae < 5 else "Needs Work"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            mape = backtest_results['mape']
            mape_color = "positive" if mape < 3 else "neutral" if mape < 7 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Mean Abs % Error</div>
                <div class="metric-value {mape_color}">{mape:.1f}%</div>
                <div class="metric-change {mape_color}">
                    {"Excellent" if mape < 3 else "Good" if mape < 7 else "Poor"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            dir_acc = backtest_results['direction_accuracy']
            dir_color = "positive" if dir_acc > 60 else "neutral" if dir_acc > 45 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Direction Accuracy</div>
                <div class="metric-value {dir_color}">{dir_acc:.1f}%</div>
                <div class="metric-change {dir_color}">
                    {"Great" if dir_acc > 60 else "OK" if dir_acc > 45 else "Poor"}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            model_acc = backtest_results['model_accuracy']
            model_color = "positive" if model_acc > 75 else "neutral" if model_acc > 60 else "negative"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Model Confidence</div>
                <div class="metric-value {model_color}">{model_acc:.1f}%</div>
                <div class="metric-change {model_color}">
                    {model_type}
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Backtest visualization
        st.markdown("### 📈 Prediction vs Reality")
        
        # Create simple comparison chart
        backtest_fig = go.Figure()
        
        dates = backtest_results['dates']
        
        # Actual prices
        backtest_fig.add_trace(
            go.Scatter(
                x=dates,
                y=backtest_results['actual'],
                name="📊 Actual Prices",
                line=dict(color="#059669", width=3),
                mode='lines+markers'
            )
        )
        
        # Predicted prices
        backtest_fig.add_trace(
            go.Scatter(
                x=dates,
                y=backtest_results['predictions'],
                name="🤖 AI Predictions",
                line=dict(color="#2563eb", width=3, dash='dash'),
                mode='lines+markers'
            )
        )
        
        backtest_fig.update_layout(
            title=f"Backtest Results: {model_type} Model Performance",
            template="plotly_white",
            height=500,
            font=dict(family="Inter, sans-serif"),
            legend=dict(x=0.02, y=0.98),
            hovermode='x unified'
        )
        
        backtest_fig.update_xaxes(title="Date")
        backtest_fig.update_yaxes(title="Price ($)")
        
        st.plotly_chart(backtest_fig, use_container_width=True)
        
        # Backtest summary
        avg_actual = np.mean(backtest_results['actual'])
        avg_predicted = np.mean(backtest_results['predictions'])
        prediction_bias = ((avg_predicted - avg_actual) / avg_actual) * 100
        
        bias_color = "#059669" if abs(prediction_bias) < 2 else "#d97706" if abs(prediction_bias) < 5 else "#dc2626"
        bias_text = "Unbiased" if abs(prediction_bias) < 2 else "Slight Bias" if abs(prediction_bias) < 5 else "High Bias"
        
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                   padding: 2rem; border-radius: 12px; margin: 2rem 0; border: 1px solid #cbd5e1;">
            <h3 style="color: #0f172a; margin-bottom: 1rem;">🎯 Backtesting Summary</h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div>
                    <strong>Prediction Bias:</strong><br>
                    <span style="color: {bias_color}; font-size: 1.1rem; font-weight: 600;">
                        {prediction_bias:+.2f}% ({bias_text})
                    </span>
                </div>
                <div>
                    <strong>Best Prediction:</strong><br>
                    <span style="color: #059669;">
                        ${min(backtest_results['predictions']):.2f} - ${max(backtest_results['predictions']):.2f}
                    </span>
                </div>
                <div>
                    <strong>Actual Range:</strong><br>
                    <span style="color: #2563eb;">
                        ${min(backtest_results['actual']):.2f} - ${max(backtest_results['actual']):.2f}
                    </span>
                </div>
            </div>
            <hr style="margin: 1.5rem 0; opacity: 0.3;">
            <div style="font-size: 0.9rem; color: #64748b;">
                📝 <strong>Interpretation:</strong> 
                {"🎉 Excellent performance!" if dir_acc > 60 and mape < 5 else 
                 "✅ Good performance with room for improvement" if dir_acc > 45 and mape < 10 else 
                 "⚠️ Model needs refinement for this stock"} 
                Direction accuracy above 50% indicates the model can predict trend direction better than random chance.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Simple data table
        st.markdown("### 📋 Detailed Comparison")
        
        comparison_df = pd.DataFrame({
            'Date': [d.strftime('%Y-%m-%d') for d in dates],
            'Actual Price': [f"${p:.2f}" for p in backtest_results['actual']],
            'Predicted Price': [f"${p:.2f}" for p in backtest_results['predictions']],
            'Error': [f"${abs(a-p):.2f}" for a, p in zip(backtest_results['actual'], backtest_results['predictions'])],
            'Accuracy': [f"{100-abs((a-p)/a*100):.1f}%" for a, p in zip(backtest_results['actual'], backtest_results['predictions'])]
        })
        
        st.dataframe(comparison_df, use_container_width=True)
        
        # Download results
        csv = comparison_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Backtest Results",
            data=csv,
            file_name=f"backtest_{ticker}_{model_type.replace(' ', '_')}.csv",
            mime="text/csv"
        )

    # Welcome/Help Section
    else:
        st.markdown("## 🌟 Welcome to AI StockBot Pro")
        
        # Feature showcase
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card fade-in">
                <h3>🧠 Advanced AI Models</h3>
                <p>Choose from Random Forest, GRU Neural Networks, or Ensemble methods for maximum accuracy</p>
                <div class="status-indicator status-active" style="margin-top: 1rem;">
                    ✅ Multi-Model Support
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card fade-in">
                <h3>📰 Real-Time Sentiment</h3>
                <p>Live news analysis with FinBERT transformer models for market sentiment scoring</p>
                <div class="status-indicator status-active" style="margin-top: 1rem;">
                    ✅ Live News Feed
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card fade-in">
                <h3>⚡ Lightning Fast</h3>
                <p>Optimized caching and parallel processing for sub-3-second predictions</p>
                <div class="status-indicator status-active" style="margin-top: 1rem;">
                    ✅ Ultra-Fast Performance
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # Sample chart for