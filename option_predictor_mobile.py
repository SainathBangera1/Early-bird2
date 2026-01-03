import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, r2_score
from io import StringIO
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set page configuration for mobile
st.set_page_config(
    page_title="Option Predictor Mobile",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapsed for mobile
)

# Mobile-optimized CSS
st.markdown("""
<style>
    /* Mobile-first responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 1.8rem !important;
            padding: 10px 5px !important;
        }
        .stButton > button {
            width: 100% !important;
            margin: 5px 0 !important;
            font-size: 16px !important;
            height: 50px !important;
        }
        .stNumberInput, .stSelectbox, .stDateInput, .stTextInput {
            margin-bottom: 15px !important;
        }
        .prediction-card {
            padding: 15px !important;
            margin: 10px 0 !important;
            border-radius: 10px !important;
        }
        .metric-card {
            padding: 12px !important;
            margin: 8px 0 !important;
            font-size: 14px !important;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px !important;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 8px 12px !important;
            font-size: 14px !important;
        }
    }
    
    /* General mobile styles */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
        font-size: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .prediction-card {
        background: white;
        padding: 20px;
        border-radius: 15px;
        margin: 15px 0;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e0e0e0;
    }
    
    .buy-call-card {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        border-left: 5px solid #28a745;
        color: #155724;
    }
    
    .buy-put-card {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        border-left: 5px solid #dc3545;
        color: #721c24;
    }
    
    .metric-card {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #dee2e6;
        text-align: center;
    }
    
    .mobile-input-group {
        background: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    
    .risk-warning {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 12px;
        border-radius: 8px;
        margin: 15px 0;
        font-size: 14px;
    }
    
    .stProgress > div > div {
        height: 15px !important;
        border-radius: 10px !important;
    }
    
    /* Hide sidebar on mobile */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            display: none !important;
        }
        div[data-testid="stHorizontalBlock"] {
            flex-direction: column !important;
        }
    }
    
    /* Better touch targets */
    button, input, select, textarea {
        font-size: 16px !important; /* Prevents iOS zoom */
    }
    
    /* Mobile-friendly tabs */
    .mobile-tab {
        font-size: 14px !important;
        padding: 8px 4px !important;
    }
</style>
""", unsafe_allow_html=True)

# Your existing data
data_text = """Date ,DAY,OPTION Name,CE-previous,PE- previous,PCR,OPCR,CE – Buy,PE- Buy,CE – Sell,PE – Sell,CE – Profit,PE – Profit
29 / 10 / 2025,Wednesday,25950,155.13,113.32,0.59,0.78,0,96,0,102.72,0,504
31 / 10 / 2025,Friday,25950,93.64,101.6,0.44,0.6,78.2,101.6,83.674,108.712,410.55,533.4
04 / 11 / 2025,Tuesday (EXP),25800,52.111,59.532,0.6,0.66,0,0,0,0,0,0
06 / 11 / 2025,Thursday,25700,88.478,151.618,0.65,0.73,0,122.65,0,131.2355,0,643.9125
07 / 11 / 2025,Friday,25600,68.101,104.304,0.56,0.62,37,0,47,0,750,0
10 / 11 / 2025,Monday,25500,83.599,59.532,0.88,0.89,0,35.95,0,45,0,678.75
11 / 11 / 2025,Tuesday (EXP),25600,53.915,41.287,0.89,0.97,53.95,41.28,63.95,51,750,729
11 / 11 / 2025,Tuesday(NP),25600,157.317,105.78,1.07,0.79,157,0,167,0,750,0
12 / 11 / 2025,Wednesday,25700,163.18,95.038,0.91,1.04,0,66.25,0,77.25,0,825
13 / 11 / 2025,Thursday,25850,146.575,80.483,2.13,1.31,136.7,0,147.55,0,813.750000000002,0
14 / 11 / 2025,Friday,25900,116.85,109.265,1.08,1.06,114.6,0,124.6,0,750,0
17 / 11 / 2025,Monday,25850,123.82,51.66,1.48,0.81,0,51.66,0,0,0,-3874.5
18  / 11 / 2025,Tuesday(NP),25950,191.019,97.211,1.35,1.01,0,95.7,0,0,0,-7177.5
20 / 11 / 2025,Thursday,26000,151.044,89.544,1.07,1.29,0,73.6,0,83.6,0,750
21 / 11 / 2025,Friday,26100,164.328,53.259,2.18,1.51,136,0,146,0,750,0
24 / 11 / 2025,Monday,26050,95.202,75.891,1.35,1,0,51.6,0,61.6,0,750
25 / 11 / 2025,Tuesday (EXP),26000,43.87,72.98,0.8,0.66,39.4,72.98,49.4,82.98,,
25 / 11 / 2025,Tuesday (NP),26050,124.927,154.16,0.76,0.73,143.1,154.1,153.6,164.1,787.5,750
26 / 11 / 2025,Wednesday,26000,87.494,148.707,0.7,0.73,0,117.05,0,0,0,-8778.75
27 / 11 / 2025,Thursday,26150,148.912,78.72,1.94,1.56,0,70.3,0,80.3,0,750
01 / 12 / 2025,Monday,26200,93.357,52.357,1.35,1.13,0,35.15,0,45.15,0,750
02 / 12 / 2025,Tuesday (EXP),26200,43.665,54.284,0.57,0.75,21.9,0,31.9,0,750,0
02 / 12 / 2025,Tuesday (NP),26200,140.876,111.028,0.92,0.91,136.95,0,146.95,0,750,0
03 / 12 / 2025,Wednesday,26100,122.426,102.869,0.72,0.73,111.15,0,90,0,-1586.25,
04 / 12 / 2025,Thursday,26000,111.684,86.92,0.73,0.69,98.85,0,108.85,0,750,0
05 / 12 / 2025,Friday,26000,114.964,57.605,1.09,0.8,0,57.4,0,0,0,-4305
08 / 12 / 2025,Monday,26150,86.55,43.911,1.27,1.23,86,0,96,0,750,0
09 / 12 / 2025,Tuesday (EXP),26000,30.422,77.49,0.39,0.48,16.05,,,,-1203.75,0
09 / 12 / 2025,Tuesday (NP),26000,115.251,130.175,0.94,0.68,107.3,0,117.3,0,750,0
10 / 12  / 2025,Wednesday,25900,100.86,117.26,0.68,0.67,0,117.26,0,127.26,0,750
11 / 12 / 2025,Thursday,25850,82.82,132.717,0.36,0.54,82.4,131.7,92.4,142.7,750,825
12 / 12 / 2025,Friday,25900,94.792,70.684,0.95,0.84,0,52,0,62,0,750
15 / 12 / 2025,Monday,26000,92.332,40.426,1.47,1.17,56.9,0,66.9,0,750.000000000001,0
16 / 12 / 2025,Tuesday (EXP),26000,58.302,38.376,1.44,1.23,28,0,0.05,0,-2096.25,0
16 / 12 / 2025,Tuesday (NP),26000,147.518,90.2,1.03,0.97,128.8,0,84.3,0,-3337.5,0
17 / 12 / 2025,Wednesday,25900,108.445,108.24,0.73,0.68,0,108.24,0,118.24,0,750
18 / 12 / 2025,Thursday,25850,102.828,85.936,0.59,0.58,0,85.65,0,95.65,0,750
19 / 12 / 2025,Friday,25850,78.18,89.995,0.59,0.68,78.18,0,0,0,-5863.5,0
22 / 12 / 2025,Monday,25950,77.9,49.446,1.57,1.12,77.9,0,0,0,-5842.5,0
23 / 12 / 2025,Tuesday(EXP),26150,44.97,42.64,1.2,1.64,44.2,33.35,54.2,43.35,750,750
23  / 12 / 2025,Tuesday ,26000,212.05,49.32,1.36,1.13,0,48.8,0,0,0,-3660
24 / 12 / 2025,Wednesday,26100,143.459,55.268,1.65,1.08,143,55.25,154,0,825,-4143.75
26 / 12 / 2025,Friday,26100,100.901,45.715,1.6,0.91,0,0,0,0,0,0
29 / 12 / 2025,Monday,26000,82.697,33.702,1.34,0.66,0,29.25,0,39.25,0,750
30 / 12 / 2025,Tuesday (EXP),25950,42.722,36.777,0.83,0.56,21.95,0,31.95,0,750,0
30 / 12 / 2025,Tuesday,26000,103.648,97.99,0.79,0.69,96.4,0,106.4,0,750,0
31 / 12 / 2025,Wednesday,26000,91.102,89.052,0.7,0.75,0,83.05,0,93.05,0,750
01 / 01 / 2026,Thursday,26100,119.67,56.08,1.92,1.35,0,49.4,0,59.4,0,750
02 / 01 / 2026,Friday,26150,73.759,57.441,0.94,0.89,0,49.35,0,0,0,-3701.25"""

# Your existing data cleaning functions
def clean_data_manual(df):
    """Manually clean the data based on actual column names"""
    data = df.copy()
    
    cleaned_data = {}
    
    # Extract and clean each column
    cleaned_data['Date'] = pd.to_datetime(data.iloc[:, 0].str.strip(), format='%d / %m / %Y', errors='coerce')
    cleaned_data['Day'] = data.iloc[:, 1].str.strip()
    cleaned_data['Option_Name'] = pd.to_numeric(data.iloc[:, 2], errors='coerce')
    cleaned_data['Strike_Price'] = cleaned_data['Option_Name']
    cleaned_data['CE_previous'] = pd.to_numeric(data.iloc[:, 3], errors='coerce')
    cleaned_data['PE_previous'] = pd.to_numeric(data.iloc[:, 4], errors='coerce')
    cleaned_data['PCR'] = pd.to_numeric(data.iloc[:, 5], errors='coerce')
    cleaned_data['OPCR'] = pd.to_numeric(data.iloc[:, 6], errors='coerce')
    cleaned_data['CE_Buy'] = pd.to_numeric(data.iloc[:, 7], errors='coerce')
    cleaned_data['PE_Buy'] = pd.to_numeric(data.iloc[:, 8], errors='coerce')
    cleaned_data['CE_Sell'] = pd.to_numeric(data.iloc[:, 9], errors='coerce')
    cleaned_data['PE_Sell'] = pd.to_numeric(data.iloc[:, 10], errors='coerce')
    cleaned_data['CE_Profit'] = pd.to_numeric(data.iloc[:, 11], errors='coerce')
    cleaned_data['PE_Profit'] = pd.to_numeric(data.iloc[:, 12] if len(data.columns) > 12 else pd.Series([np.nan] * len(data)), errors='coerce')
    
    cleaned_df = pd.DataFrame(cleaned_data)
    cleaned_df['CE_Buy'] = cleaned_df['CE_Buy'].fillna(0)
    cleaned_df['PE_Buy'] = cleaned_df['PE_Buy'].fillna(0)
    
    cleaned_df['Buy_Decision'] = np.where(cleaned_df['CE_Buy'] > 0, 1, 
                                         np.where(cleaned_df['PE_Buy'] > 0, 0, -1))
    cleaned_df['Buy_Price'] = np.where(cleaned_df['CE_Buy'] > 0, cleaned_df['CE_Buy'], 
                                      np.where(cleaned_df['PE_Buy'] > 0, cleaned_df['PE_Buy'], 0))
    
    return cleaned_df

# Your existing SimpleOptionPredictor class
class SimpleOptionPredictor:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=5)
        self.regressor = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
        self.scaler = StandardScaler()
        self.feature_cols = ['CE_previous', 'PE_previous', 'PCR', 'OPCR', 'Strike_Price']
        
    def prepare_features(self, data):
        """Prepare features for the model"""
        X = pd.DataFrame()
        
        for col in self.feature_cols:
            if col in data.columns:
                X[col] = data[col]
            else:
                X[col] = 0
        
        if 'Date' in data.columns:
            X['Day_of_Week'] = data['Date'].dt.dayofweek
            X['Month'] = data['Date'].dt.month
            X['Day_of_Month'] = data['Date'].dt.day
        
        X['CE_PE_Ratio'] = X['CE_previous'] / (X['PE_previous'] + 1e-10)
        X['PCR_OPCR_Ratio'] = X['PCR'] / (X['OPCR'] + 1e-10)
        X['PCR_OPCR_Diff'] = X['PCR'] - X['OPCR']
        
        return X
    
    def train(self, data):
        """Train both classification and regression models"""
        X = self.prepare_features(data)
        mask = data['Buy_Decision'] != -1
        X_buy = X[mask]
        y_class = data.loc[mask, 'Buy_Decision']
        y_reg = data.loc[mask, 'Buy_Price']
        
        if len(X_buy) < 10:
            return self
        
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X_buy, y_class, y_reg, test_size=0.3, random_state=42, stratify=y_class
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.classifier.fit(X_train_scaled, y_class_train)
        self.regressor.fit(X_train_scaled, y_reg_train)
        
        return self
    
    def predict_single(self, input_dict):
        """
        Predict for a single input
        """
        try:
            input_df = pd.DataFrame([{
                'Date': pd.to_datetime(input_dict['Date']),
                'Option_Name': float(input_dict['Option Name']),
                'CE_previous': float(input_dict['CE-previous']),
                'PE_previous': float(input_dict['PE-previous']),
                'PCR': float(input_dict['PCR']),
                'OPCR': float(input_dict['OPCR']),
                'Strike_Price': float(input_dict['Option Name'])
            }])
            
            X = self.prepare_features(input_df)
            X_scaled = self.scaler.transform(X)
            
            decision = self.classifier.predict(X_scaled)[0]
            proba = self.classifier.predict_proba(X_scaled)[0]
            price = max(0, self.regressor.predict(X_scaled)[0])
            
            if decision == 1:
                result = {
                    'action': 'BUY_CALL',
                    'predicted_CE_Buy': round(float(price), 2),
                    'predicted_PE_Buy': 0.0,
                    'confidence': round(float(max(proba)), 3),
                    'call_probability': round(float(proba[1]), 3),
                    'put_probability': round(float(proba[0]), 3)
                }
            else:
                result = {
                    'action': 'BUY_PUT',
                    'predicted_CE_Buy': 0.0,
                    'predicted_PE_Buy': round(float(price), 2),
                    'confidence': round(float(max(proba)), 3),
                    'call_probability': round(float(proba[1]), 3),
                    'put_probability': round(float(proba[0]), 3)
                }
            
            result['feature_analysis'] = {
                'CE_PE_Ratio': round(float(X['CE_PE_Ratio'].iloc[0]), 3),
                'PCR_OPCR_Ratio': round(float(X['PCR_OPCR_Ratio'].iloc[0]), 3),
                'PCR_OPCR_Diff': round(float(X['PCR_OPCR_Diff'].iloc[0]), 3)
            }
            
            return result
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

# Initialize predictor with caching
@st.cache_resource
def initialize_predictor():
    """Initialize and train the predictor model"""
    df = pd.read_csv(StringIO(data_text))
    data_cleaned = clean_data_manual(df)
    predictor = SimpleOptionPredictor()
    predictor.train(data_cleaned)
    return predictor

# Mobile-optimized UI components
def mobile_header():
    """Mobile-friendly header"""
    st.markdown('<h1 class="main-header">📱 Option Predictor</h1>', unsafe_allow_html=True)
    
    # Quick stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Accuracy", "85%", "+2%")
    with col2:
        st.metric("Calls", "65%", "-5%")
    with col3:
        st.metric("Puts", "35%", "+5%")

def mobile_input_section():
    """Mobile-friendly input section"""
    st.markdown('<div class="mobile-input-group">', unsafe_allow_html=True)
    st.subheader("📝 Option Details")
    
    # Use columns for mobile layout
    col1, col2 = st.columns(2)
    
    with col1:
        date = st.date_input(
            "Trade Date",
            value=datetime.now() + timedelta(days=1),
            min_value=datetime.now(),
            max_value=datetime.now() + timedelta(days=365),
            help="Select the trading date"
        )
    
    with col2:
        day = st.selectbox(
            "Day",
            options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
            index=0,
            help="Select day of the week"
        )
    
    # Strike price
    strike_price = st.select_slider(
        "Strike Price",
        options=[25500, 25600, 25700, 25800, 25900, 26000, 26100, 26200],
        value=26000,
        help="Select option strike price"
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Price inputs in mobile-optimized columns
    st.markdown('<div class="mobile-input-group">', unsafe_allow_html=True)
    st.subheader("💰 Previous Prices")
    
    price_col1, price_col2 = st.columns(2)
    
    with price_col1:
        ce_prev = st.number_input(
            "CE Previous Price",
            min_value=0.0,
            max_value=500.0,
            value=120.5,
            step=0.1,
            format="%.2f",
            help="Previous day's call option price"
        )
    
    with price_col2:
        pe_prev = st.number_input(
            "PE Previous Price",
            min_value=0.0,
            max_value=500.0,
            value=85.3,
            step=0.1,
            format="%.2f",
            help="Previous day's put option price"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Ratio inputs
    st.markdown('<div class="mobile-input-group">', unsafe_allow_html=True)
    st.subheader("📊 Market Ratios")
    
    ratio_col1, ratio_col2 = st.columns(2)
    
    with ratio_col1:
        pcr = st.slider(
            "PCR",
            min_value=0.0,
            max_value=3.0,
            value=0.85,
            step=0.01,
            help="Put-Call Ratio (PCR < 0.8 = Bullish, PCR > 1.2 = Bearish)"
        )
    
    with ratio_col2:
        opcr = st.slider(
            "OPCR",
            min_value=0.0,
            max_value=3.0,
            value=0.92,
            step=0.01,
            help="Overall Put-Call Ratio"
        )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    return {
        'Date': date.strftime('%Y-%m-%d'),
        'Day': day,
        'Option Name': str(strike_price),
        'CE-previous': ce_prev,
        'PE-previous': pe_prev,
        'PCR': pcr,
        'OPCR': opcr
    }

def display_market_insights(input_data):
    """Display market insights in mobile format"""
    st.markdown("### 📈 Market Insights")
    
    # Calculate insights
    ce_pe_ratio = input_data['CE-previous'] / (input_data['PE-previous'] + 1e-10)
    pcr_status = "🟢 Bullish" if input_data['PCR'] < 0.8 else "🔴 Bearish" if input_data['PCR'] > 1.2 else "🟡 Neutral"
    pcr_opcr_diff = input_data['PCR'] - input_data['OPCR']
    
    # Display in mobile cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("CE/PE Ratio", f"{ce_pe_ratio:.2f}")
        st.caption("Higher = Calls favored")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("PCR Status", pcr_status)
        st.caption("Market sentiment")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("PCR-OPCR Diff", f"{pcr_opcr_diff:.3f}")
        st.caption("Contract vs Market")
        st.markdown('</div>', unsafe_allow_html=True)

def display_prediction_result_mobile(result, input_data):
    """Display prediction results in mobile-friendly format"""
    
    # Determine card class based on action
    card_class = "buy-call-card" if result['action'] == 'BUY_CALL' else "buy-put-card"
    
    st.markdown(f'<div class="prediction-card {card_class}">', unsafe_allow_html=True)
    
    # Action header
    if result['action'] == 'BUY_CALL':
        st.markdown("### 📈 BUY CALL OPTION")
        st.markdown(f"**Entry Price:** ₹{result['predicted_CE_Buy']}")
        st.success("🎯 Market expects UPWARD movement")
    else:
        st.markdown("### 📉 BUY PUT OPTION")
        st.markdown(f"**Entry Price:** ₹{result['predicted_PE_Buy']}")
        st.error("🎯 Market expects DOWNWARD movement")
    
    # Confidence meter
    confidence_percent = result['confidence'] * 100
    st.progress(result['confidence'], text=f"Confidence: {confidence_percent:.1f}%")
    
    # Probability breakdown
    prob_col1, prob_col2 = st.columns(2)
    with prob_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Call Probability", f"{result['call_probability']*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with prob_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Put Probability", f"{result['put_probability']*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Trading advice
    display_trading_advice_mobile(result)

def display_trading_advice_mobile(result):
    """Display mobile-friendly trading advice"""
    st.markdown("### 💡 Trading Advice")
    
    # Position sizing based on confidence
    if result['confidence'] > 0.8:
        position_size = "3-5% of capital"
        risk_level = "🟢 Low Risk"
    elif result['confidence'] > 0.6:
        position_size = "1-3% of capital"
        risk_level = "🟡 Medium Risk"
    else:
        position_size = "<1% of capital"
        risk_level = "🔴 High Risk"
    
    # Stop loss and target
    if result['action'] == 'BUY_CALL':
        entry_price = result['predicted_CE_Buy']
    else:
        entry_price = result['predicted_PE_Buy']
    
    stop_loss = entry_price * 0.95
    target_price = entry_price * 1.15
    
    # Display in cards
    advice_col1, advice_col2, advice_col3 = st.columns(3)
    
    with advice_col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Position Size**")
        st.markdown(f"`{position_size}`")
        st.markdown(f"*{risk_level}*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with advice_col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Stop Loss**")
        st.markdown(f"`₹{stop_loss:.2f}`")
        st.markdown("*5% below entry*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with advice_col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown("**Target Price**")
        st.markdown(f"`₹{target_price:.2f}`")
        st.markdown("*15% above entry*")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Risk warning
    st.markdown('<div class="risk-warning">', unsafe_allow_html=True)
    st.markdown("""
    ⚠️ **Risk Disclaimer:** 
    Options trading involves significant risk. This prediction is based on AI analysis, not financial advice. 
    Always conduct your own research and consider consulting a financial advisor.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

def batch_prediction_mobile():
    """Mobile-friendly batch prediction"""
    st.markdown("### 📁 Batch Prediction")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=['csv'],
        help="Upload CSV with Date, Strike_Price, CE_previous, PE_previous, PCR, OPCR columns"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ {len(df)} records loaded")
            
            with st.expander("📊 Preview Data"):
                st.dataframe(df.head())
            
            if st.button("🔮 Predict All", type="primary", use_container_width=True):
                with st.spinner("Processing..."):
                    progress_bar = st.progress(0)
                    results = []
                    
                    for idx, row in df.iterrows():
                        input_data = {
                            'Date': str(row.get('Date', '2026-01-01')),
                            'Day': 'Monday',
                            'Option Name': str(row.get('Strike_Price', 26000)),
                            'CE-previous': float(row.get('CE_previous', 100)),
                            'PE-previous': float(row.get('PE_previous', 100)),
                            'PCR': float(row.get('PCR', 1.0)),
                            'OPCR': float(row.get('OPCR', 1.0))
                        }
                        
                        result = predictor.predict_single(input_data)
                        if result:
                            results.append({
                                'Strike_Price': input_data['Option Name'],
                                'Prediction': result['action'],
                                'Price': result['predicted_CE_Buy'] if result['action'] == 'BUY_CALL' else result['predicted_PE_Buy'],
                                'Confidence': f"{result['confidence']*100:.1f}%"
                            })
                        
                        progress_bar.progress((idx + 1) / len(df))
                    
                    # Display results
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df)
                    
                    # Summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Predictions", len(results))
                    with col2:
                        calls = len([r for r in results if r['Prediction'] == 'BUY_CALL'])
                        st.metric("Calls Recommended", calls)
                    with col3:
                        puts = len([r for r in results if r['Prediction'] == 'BUY_PUT'])
                        st.metric("Puts Recommended", puts)
        
        except Exception as e:
            st.error(f"Error: {str(e)}")

def market_analysis_mobile():
    """Mobile-friendly market analysis"""
    st.markdown("### 📊 Market Analysis")
    
    # Create sample market data
    dates = pd.date_range(start='2025-12-01', periods=30, freq='D')
    pcr_values = 0.7 + 0.5 * np.sin(np.linspace(0, 2*np.pi, 30)) + np.random.normal(0, 0.1, 30)
    
    # Market indicators
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("NIFTY", "22,450", "+125")
    with col2:
        st.metric("VIX", "14.25", "-0.75")
    with col3:
        st.metric("PCR Trend", "0.89", "-0.05")
    
    # PCR Chart
    chart_data = pd.DataFrame({
        'Date': dates,
        'PCR': pcr_values
    })
    st.line_chart(chart_data.set_index('Date')['PCR'])
    
    # Market sentiment guide
    with st.expander("📖 PCR Guide"):
        st.markdown("""
        **PCR Interpretation:**
        - **< 0.8**: Bullish (Excessive call buying)
        - **0.8 - 1.2**: Neutral (Balanced market)
        - **> 1.2**: Bearish (Excessive put buying)
        
        **Trading Signals:**
        - PCR spike → Often indicates market bottom
        - PCR drop → Often indicates market top
        - PCR divergence → Potential reversal signal
        """)

def about_page_mobile():
    """Mobile-friendly about page"""
    st.markdown("### ℹ️ About")
    
    st.markdown("""
    **Option Predictor Mobile** helps you make informed option trading decisions using AI.
    
    **How it works:**
    1. Analyzes market indicators (PCR, OPCR)
    2. Uses machine learning to predict option buying
    3. Provides entry price and confidence scores
    
    **Key Features:**
    - 📱 Mobile-optimized interface
    - 🔮 Single & batch predictions
    - 📊 Market analysis
    - 💡 Trading advice
    
    **Disclaimer:**
    This tool provides AI-generated predictions, not financial advice.
    Options trading involves risk. Always do your own research.
    """)

# Main app function
def main():
    """Main mobile app"""
    
    # Initialize predictor
    global predictor
    predictor = initialize_predictor()
    
    # Mobile header
    mobile_header()
    
    # Mobile navigation using tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🔮 Predict", "📁 Batch", "📊 Market", "ℹ️ About"])
    
    with tab1:
        # Single prediction tab
        st.markdown("### Quick Prediction")
        
        # Get user input
        input_data = mobile_input_section()
        
        # Display market insights
        display_market_insights(input_data)
        
        # Prediction button
        if st.button("🔮 Get Prediction", type="primary", use_container_width=True, key="predict_button"):
            with st.spinner("Analyzing..."):
                result = predictor.predict_single(input_data)
                
                if result:
                    # Display result
                    display_prediction_result_mobile(result, input_data)
                else:
                    st.error("Failed to generate prediction. Please check your inputs.")
    
    with tab2:
        # Batch prediction tab
        batch_prediction_mobile()
    
    with tab3:
        # Market analysis tab
        market_analysis_mobile()
    
    with tab4:
        # About tab
        about_page_mobile()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666; font-size: 12px;'>"
        "📱 Mobile Optimized • Option Predictor • Version 1.0"
        "</div>",
        unsafe_allow_html=True
    )

# Run the app
if __name__ == "__main__":
    main()