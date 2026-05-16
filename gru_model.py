import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Check for required libraries
try:
    import torch
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    PYTORCH_AVAILABLE = True
    TENSORFLOW_AVAILABLE = True
    print("✅ PyTorch available for GRU")
    print("✅ TensorFlow available for GRU")
except ImportError as e:
    print(f"❌ Deep learning libraries not available: {e}")
    PYTORCH_AVAILABLE = False
    TENSORFLOW_AVAILABLE = False

def is_gru_available():
    """Check if GRU dependencies are available"""
    return PYTORCH_AVAILABLE or TENSORFLOW_AVAILABLE

class FastGRUPredictor:
    def __init__(self, sequence_length=60):
        self.sequence_length = sequence_length
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.feature_columns = None
        
    def create_features(self, data):
        """Create technical indicators and features"""
        df = data.copy()
        
        # Ensure we have the basic OHLCV columns
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: {col} column missing, using Close price as fallback")
                df[col] = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
        
        # Price-based features
        df['Price_Change'] = df['Close'].pct_change()
        df['Price_MA5'] = df['Close'].rolling(window=5).mean()
        df['Price_MA20'] = df['Close'].rolling(window=20).mean()
        df['Price_MA50'] = df['Close'].rolling(window=50).mean()
        
        # Technical indicators
        df['RSI'] = self.calculate_rsi(df['Close'])
        df['MACD'] = self.calculate_macd(df['Close'])
        df['BB_upper'], df['BB_lower'] = self.calculate_bollinger_bands(df['Close'])
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        
        # Ratios and normalized features
        df['High_Low_Ratio'] = df['High'] / df['Low']
        df['Close_Open_Ratio'] = df['Close'] / df['Open']
        
        # Fix the problematic ratio calculations
        df['Price_MA5_Ratio'] = np.where(df['Price_MA5'] != 0, df['Close'] / df['Price_MA5'], 1.0)
        df['Price_MA20_Ratio'] = np.where(df['Price_MA20'] != 0, df['Close'] / df['Price_MA20'], 1.0)
        df['Volume_Ratio'] = np.where(df['Volume_MA'] != 0, df['Volume'] / df['Volume_MA'], 1.0)
        
        # Handle any remaining NaN values
        df = df.fillna(method='forward').fillna(method='backward').fillna(0)
        
        # Select feature columns for training
        feature_cols = [
            'Close', 'Volume', 'Price_Change', 'RSI', 'MACD',
            'High_Low_Ratio', 'Close_Open_Ratio', 'Price_MA5_Ratio', 
            'Price_MA20_Ratio', 'Volume_Ratio', 'BB_upper', 'BB_lower'
        ]
        
        # Ensure all feature columns exist
        available_features = [col for col in feature_cols if col in df.columns]
        self.feature_columns = available_features
        
        return df[available_features]
    
    def calculate_rsi(self, prices, window=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill NaN with neutral RSI
    
    def calculate_macd(self, prices, fast=12, slow=26):
        """Calculate MACD indicator"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        return macd.fillna(0)
    
    def calculate_bollinger_bands(self, prices, window=20, num_std=2):
        """Calculate Bollinger Bands"""
        ma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        upper = ma + (std * num_std)
        lower = ma - (std * num_std)
        return upper.fillna(prices), lower.fillna(prices)
    
    def prepare_data(self, features, target_col='Close'):
        """Prepare data for GRU training"""
        # Scale the features
        scaled_data = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        target_idx = features.columns.get_loc(target_col)
        
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            y.append(scaled_data[i, target_idx])
        
        return np.array(X), np.array(y)
    
    def create_tensorflow_model(self, input_shape):
        """Create TensorFlow GRU model - REDUCED OVERFITTING"""
        model = Sequential([
            GRU(32, return_sequences=True, input_shape=input_shape),  # Reduced from 50 to 32
            Dropout(0.4),  # Increased from 0.2 to 0.4
            GRU(16, return_sequences=False),  # Reduced from 50 to 16
            Dropout(0.4),  # Increased from 0.2 to 0.4
            Dense(8),  # Reduced from 25 to 8
            Dropout(0.3),  # Added dropout to Dense layer
            Dense(1)
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mae']
        )
        
        return model
    
    def train(self, stock_data, epochs=20, batch_size=32, validation_split=0.2):
        """Train the GRU model"""
        if not TENSORFLOW_AVAILABLE:
            raise Exception("TensorFlow not available for GRU training")
        
        print("🧠 Training prediction model...")
        
        try:
            # Create features
            features = self.create_features(stock_data)
            
            if len(features) < self.sequence_length + 10:
                raise Exception(f"Insufficient data. Need at least {self.sequence_length + 10} days")
            
            # Prepare training data
            X, y = self.prepare_data(features)
            
            if len(X) == 0:
                raise Exception("No training sequences created")
            
            # Create model
            self.model = self.create_tensorflow_model((X.shape[1], X.shape[2]))
            
            # Train model with reduced verbosity
            history = self.model.fit(
                X, y,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=0,  # Reduce output
                shuffle=True
            )
            
            # Calculate accuracy metrics
            predictions = self.model.predict(X, verbose=0)
            
            # Inverse transform for accuracy calculation
            dummy_features = np.zeros((len(predictions), len(self.feature_columns)))
            dummy_features[:, 0] = predictions.flatten()  # Close price is first column
            predictions_unscaled = self.scaler.inverse_transform(dummy_features)[:, 0]
            
            dummy_features[:, 0] = y
            actual_unscaled = self.scaler.inverse_transform(dummy_features)[:, 0]
            
            # Calculate accuracy (1 - normalized RMSE)
            rmse = np.sqrt(mean_squared_error(actual_unscaled, predictions_unscaled))
            mean_price = np.mean(actual_unscaled)
            accuracy = max(0, (1 - (rmse / mean_price)) * 100)
            
            print(f"✅ Model trained - Accuracy: {accuracy:.1f}%")
            return accuracy
            
        except Exception as e:
            print(f"❌ Training failed: {e}")
            raise e
    
    def predict(self, stock_data, days=7, sentiment_score=0.0):
        """Make predictions"""
        if self.model is None:
            raise Exception("Model not trained yet")
        
        try:
            # Create features for the entire dataset
            features = self.create_features(stock_data)
            
            # Get the last sequence for prediction
            scaled_data = self.scaler.transform(features)
            last_sequence = scaled_data[-self.sequence_length:].reshape(1, self.sequence_length, -1)
            
            predictions = []
            current_sequence = last_sequence.copy()
            
            for _ in range(days):
                # Predict next value
                next_pred = self.model.predict(current_sequence, verbose=0)
                
                # Convert prediction back to original scale
                dummy_features = np.zeros((1, len(self.feature_columns)))
                dummy_features[0, 0] = next_pred[0, 0]  # Close price
                pred_unscaled = self.scaler.inverse_transform(dummy_features)[0, 0]
                
                # Apply sentiment adjustment
                sentiment_factor = 1 + (sentiment_score * 0.02)  # Small sentiment influence
                pred_unscaled *= sentiment_factor
                
                predictions.append(float(pred_unscaled))
                
                # Update sequence for next prediction
                # Create new features based on prediction
                new_features = np.zeros((1, len(self.feature_columns)))
                new_features[0, 0] = next_pred[0, 0]  # Close price
                
                # Shift sequence and add new prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = new_features[0, :]
            
            return predictions
            
        except Exception as e:
            print(f"❌ Prediction failed: {e}")
            return []

# Quick prediction function
def predict_gru_fast(stock_data, sentiment_score=0.0, days=7, epochs=20):
    """Fast GRU prediction with error handling"""
    try:
        if not is_gru_available():
            raise Exception("GRU dependencies not available")
        
        predictor = FastGRUPredictor()
        
        # Train the model
        accuracy = predictor.train(stock_data, epochs=epochs)
        
        # Make predictions
        predictions = predictor.predict(stock_data, days, sentiment_score)
        
        if not predictions:
            raise Exception("No predictions generated")
        
        return predictions, accuracy
        
    except Exception as e:
        print(f"❌ GRU prediction error: {e}")
        # Return fallback prediction
        return fallback_predict(stock_data, sentiment_score, days), 0.0

def fallback_predict(stock_data, sentiment_score, days):
    """Fallback prediction using trend analysis"""
    if len(stock_data) < 10:
        return []
    
    # Use simple trend and volatility
    returns = stock_data['Close'].pct_change().tail(20).mean()
    volatility = stock_data['Close'].pct_change().tail(20).std()
    current_price = stock_data['Close'].iloc[-1]
    
    predictions = []
    for i in range(days):
        # Add trend with sentiment influence and some randomness
        trend = returns * (1 + sentiment_score * 0.1)
        random_factor = np.random.normal(0, volatility * 0.5)
        next_price = current_price * (1 + trend + random_factor)
        predictions.append(max(next_price, current_price * 0.8))  # Prevent negative prices
        current_price = next_price
    
    return predictions