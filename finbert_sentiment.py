import warnings
warnings.filterwarnings('ignore')

# Check if transformers is available
FINBERT_AVAILABLE = False
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    FINBERT_AVAILABLE = True
    print("✅ FinBERT transformers available")
except ImportError:
    print("⚠️ FinBERT transformers not available, using simple sentiment")

# Simple word-based sentiment analysis as fallback
POSITIVE_WORDS = [
    'profit', 'growth', 'increase', 'gain', 'positive', 'strong', 'bullish',
    'upgrade', 'beat', 'exceed', 'outperform', 'surge', 'rally', 'boom',
    'success', 'improvement', 'rising', 'higher', 'optimistic', 'confident'
]

NEGATIVE_WORDS = [
    'loss', 'decline', 'decrease', 'fall', 'negative', 'weak', 'bearish',
    'downgrade', 'miss', 'underperform', 'crash', 'drop', 'plunge',
    'failure', 'concern', 'risk', 'warning', 'lower', 'pessimistic', 'worry'
]

class SimpleFinBERT:
    """Simple FinBERT-like sentiment analyzer"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.available = FINBERT_AVAILABLE
        
        if self.available:
            try:
                self._load_model()
            except Exception as e:
                print(f"Failed to load FinBERT: {e}")
                self.available = False
    
    def _load_model(self):
        """Load FinBERT model"""
        model_name = "ProsusAI/finbert"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
    
    def analyze_sentiment(self, text):
        """Analyze sentiment of text"""
        if not text or not text.strip():
            return 0.0, 'neutral'
        
        if self.available and self.model is not None:
            return self._finbert_sentiment(text)
        else:
            return self._simple_sentiment(text)
    
    def _finbert_sentiment(self, text):
        """Use FinBERT for sentiment analysis"""
        try:
            # Tokenize
            inputs = self.tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=128
            )
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                probs = probabilities.cpu().numpy()[0]
            
            # FinBERT classes: [negative, neutral, positive]
            negative_prob = float(probs[0])
            neutral_prob = float(probs[1])
            positive_prob = float(probs[2])
            
            # Calculate sentiment score
            sentiment_score = positive_prob - negative_prob
            
            # Determine label
            if sentiment_score > 0.1:
                label = 'positive'
            elif sentiment_score < -0.1:
                label = 'negative'
            else:
                label = 'neutral'
            
            return sentiment_score, label
            
        except Exception as e:
            print(f"FinBERT error: {e}")
            return self._simple_sentiment(text)
    
    def _simple_sentiment(self, text):
        """Simple word-based sentiment analysis"""
        text_lower = text.lower()
        
        positive_count = sum(1 for word in POSITIVE_WORDS if word in text_lower)
        negative_count = sum(1 for word in NEGATIVE_WORDS if word in text_lower)
        
        total_words = len(text.split())
        
        if total_words == 0:
            return 0.0, 'neutral'
        
        # Calculate sentiment score
        sentiment_score = (positive_count - negative_count) / max(total_words, 1)
        
        # Normalize score to reasonable range
        sentiment_score = max(-1, min(1, sentiment_score * 10))
        
        # Determine label
        if sentiment_score > 0.1:
            label = 'positive'
        elif sentiment_score < -0.1:
            label = 'negative'
        else:
            label = 'neutral'
        
        return sentiment_score, label

# Global instance
_finbert_analyzer = None

def get_finbert_analyzer():
    """Get FinBERT analyzer instance"""
    global _finbert_analyzer
    if _finbert_analyzer is None:
        _finbert_analyzer = SimpleFinBERT()
    return _finbert_analyzer

def analyze_finbert_sentiment(headlines):
    """
    Analyze sentiment of headlines using FinBERT or simple sentiment
    Returns (average_score, sentiment_label, detailed_analysis)
    """
    if not headlines:
        return 0.0, 'neutral', {}
    
    analyzer = get_finbert_analyzer()
    
    # Handle different input types
    if isinstance(headlines, str):
        headlines = [headlines]
    elif not isinstance(headlines, list):
        headlines = list(headlines)
    
    scores = []
    individual_analysis = []
    
    for headline in headlines:
        if not headline or not str(headline).strip():
            continue
        
        score, label = analyzer.analyze_sentiment(str(headline))
        scores.append(score)
        
        individual_analysis.append({
            'headline': str(headline)[:100] + "..." if len(str(headline)) > 100 else str(headline),
            'score': round(score, 3),
            'label': label,
            'emoji': '📈' if score > 0.1 else '📉' if score < -0.1 else '➡️'
        })
    
    if not scores:
        return 0.0, 'neutral', {}
    
    # Calculate overall sentiment
    avg_score = sum(scores) / len(scores)
    
    # Determine overall label
    if avg_score > 0.05:
        overall_label = 'positive'
    elif avg_score < -0.05:
        overall_label = 'negative'
    else:
        overall_label = 'neutral'
    
    # Create detailed analysis
    detailed_analysis = {
        'total_headlines': len(headlines),
        'valid_scores': len(scores),
        'average_score': avg_score,
        'individual_analysis': individual_analysis,
        'confidence': 'high' if len(scores) >= 5 else 'medium' if len(scores) >= 2 else 'low'
    }
    
    return avg_score, overall_label, detailed_analysis

def is_finbert_available():
    """Check if FinBERT is available"""
    return FINBERT_AVAILABLE

# Simplified functions for compatibility
def format_sentiment_for_display(sentiment_score, sentiment_label, detailed_analysis=None):
    """Format sentiment for display"""
    if sentiment_score > 0.15:
        emoji = "🚀"
        color = "#00ff88"
    elif sentiment_score > 0.05:
        emoji = "📈"
        color = "#00ff88"
    elif sentiment_score > -0.05:
        emoji = "➡️"
        color = "#ffa502"
    elif sentiment_score > -0.15:
        emoji = "📉"
        color = "#ff4757"
    else:
        emoji = "💥"
        color = "#ff4757"
    
    return {
        'emoji': emoji,
        'score': sentiment_score,
        'label': sentiment_label,
        'color': color,
        'confidence': detailed_analysis.get('confidence', 'medium') if detailed_analysis else 'medium'
    }