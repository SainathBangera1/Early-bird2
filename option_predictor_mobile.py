import streamlit as st
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta

# IMPORTANT: We need to check if sklearn is available
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    SKLEARN_AVAILABLE = True
    st.success("✅ scikit-learn is available!")
except ImportError as e:
    st.error(f"❌ scikit-learn import failed: {e}")
    SKLEARN_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="Option Predictor PRO",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Your actual data
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

# Data cleaning function (from your original code)
def clean_data_manual(df):
    data = df.copy()
    cleaned_data = {}
    
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
    
    cleaned_df = pd.DataFrame(cleaned_data)
    cleaned_df['CE_Buy'] = cleaned_df['CE_Buy'].fillna(0)
    cleaned_df['PE_Buy'] = cleaned_df['PE_Buy'].fillna(0)
    
    cleaned_df['Buy_Decision'] = np.where(cleaned_df['CE_Buy'] > 0, 1, 
                                         np.where(cleaned_df['PE_Buy'] > 0, 0, -1))
    cleaned_df['Buy_Price'] = np.where(cleaned_df['CE_Buy'] > 0, cleaned_df['CE_Buy'], 
                                      np.where(cleaned_df['PE_Buy'] > 0, cleaned_df['PE_Buy'], 0))
    
    return cleaned_df

# Your ML Model Class
class OptionPredictorML:
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.classifier = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
            self.regressor = RandomForestRegressor(n_estimators=50, random_state=42, max_depth=5)
            self.scaler = StandardScaler()
        else:
            st.error("Cannot initialize ML models without scikit-learn")
            raise ImportError("scikit-learn not available")
        
        self.feature_cols = ['CE_previous', 'PE_previous', 'PCR', 'OPCR', 'Strike_Price']
        self.is_trained = False
    
    def prepare_features(self, data):
        X = pd.DataFrame()
        
        for col in self.feature_cols:
            if col in data.columns:
                X[col] = data[col]
            else:
                X[col] = 0
        
        if 'Date' in data.columns:
            X['Day_of_Week'] = data['Date'].dt.dayofweek
            X['Month'] = data['Date'].dt.month
        
        X['CE_PE_Ratio'] = X['CE_previous'] / (X['PE_previous'] + 1e-10)
        X['PCR_OPCR_Ratio'] = X['PCR'] / (X['OPCR'] + 1e-10)
        
        return X
    
    def train(self, data):
        X = self.prepare_features(data)
        mask = data['Buy_Decision'] != -1
        X_buy = X[mask]
        y_class = data.loc[mask, 'Buy_Decision']
        y_reg = data.loc[mask, 'Buy_Price']
        
        if len(X_buy) < 5:
            st.warning("Not enough training data")
            return self
        
        X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
            X_buy, y_class, y_reg, test_size=0.2, random_state=42
        )
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        self.classifier.fit(X_train_scaled, y_class_train)
        self.regressor.fit(X_train_scaled, y_reg_train)
        self.is_trained = True
        
        # Calculate accuracy
        X_test_scaled = self.scaler.transform(X_test)
        accuracy = self.classifier.score(X_test_scaled, y_class_test)
        st.sidebar.success(f"✅ Model trained! Accuracy: {accuracy:.1%}")
        
        return self
    
    def predict_single(self, input_dict):
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
            price = max(10, self.regressor.predict(X_scaled)[0])
            
            if decision == 1:
                return {
                    'action': 'BUY_CALL',
                    'predicted_CE_Buy': round(float(price), 2),
                    'predicted_PE_Buy': 0.0,
                    'confidence': round(float(max(proba)), 3),
                    'call_probability': round(float(proba[1]), 3),
                    'put_probability': round(float(proba[0]), 3),
                    'model_type': 'ML (Random Forest)'
                }
            else:
                return {
                    'action': 'BUY_PUT',
                    'predicted_CE_Buy': 0.0,
                    'predicted_PE_Buy': round(float(price), 2),
                    'confidence': round(float(max(proba)), 3),
                    'call_probability': round(float(proba[1]), 3),
                    'put_probability': round(float(proba[0]), 3),
                    'model_type': 'ML (Random Forest)'
                }
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")
            return None

# Initialize with caching
@st.cache_resource
def initialize_ml_predictor():
    """Initialize and train the ML predictor"""
    if not SKLEARN_AVAILABLE:
        st.error("Cannot initialize: scikit-learn not installed")
        return None
    
    try:
        df = pd.read_csv(StringIO(data_text))
        data_cleaned = clean_data_manual(df)
        predictor = OptionPredictorML()
        predictor.train(data_cleaned)
        return predictor
    except Exception as e:
        st.error(f"Failed to initialize predictor: {e}")
        return None

# Alternative: Rule-based fallback
class RuleBasedPredictor:
    def predict_single(self, input_dict):
        ce = float(input_dict['CE-previous'])
        pe = float(input_dict['PE-previous'])
        pcr = float(input_dict['PCR'])
        opcr = float(input_dict['OPCR'])
        
        # Simple rules
        if pcr < 0.8:
            action = 'BUY_CALL'
            price = ce * 1.05
            confidence = 0.7
            call_prob = 0.7
            put_prob = 0.3
        elif pcr > 1.2:
            action = 'BUY_PUT'
            price = pe * 1.05
            confidence = 0.7
            call_prob = 0.3
            put_prob = 0.7
        else:
            # Neutral - decide based on CE/PE ratio
            if ce > pe:
                action = 'BUY_CALL'
                price = ce * 1.03
                confidence = 0.5
                call_prob = 0.6
                put_prob = 0.4
            else:
                action = 'BUY_PUT'
                price = pe * 1.03
                confidence = 0.5
                call_prob = 0.4
                put_prob = 0.6
        
        return {
            'action': action,
            'predicted_CE_Buy': round(price, 2) if action == 'BUY_CALL' else 0.0,
            'predicted_PE_Buy': round(price, 2) if action == 'BUY_PUT' else 0.0,
            'confidence': confidence,
            'call_probability': call_prob,
            'put_probability': put_prob,
            'model_type': 'Rule-based'
        }

# Main App
def main():
    st.title("📈 Option Predictor PRO")
    st.markdown("**Machine Learning-powered option prediction**")
    
    # Sidebar
    with st.sidebar:
        st.header("Model Information")
        
        if SKLEARN_AVAILABLE:
            st.success("✅ scikit-learn v1.3.0")
            st.info("Using Random Forest ML model")
            
            # Initialize ML predictor
            ml_predictor = initialize_ml_predictor()
            rule_predictor = RuleBasedPredictor()
            
            use_ml = st.checkbox("Use Machine Learning", value=True, 
                                 help="Uncheck to use rule-based prediction")
            
            if ml_predictor and ml_predictor.is_trained and use_ml:
                predictor = ml_predictor
                st.success("ML Model: Active")
            else:
                predictor = rule_predictor
                st.warning("Rule-based: Active")
        else:
            st.error("❌ scikit-learn not available")
            st.warning("Using rule-based prediction only")
            predictor = RuleBasedPredictor()
    
    # Main content
    tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 Analysis", "📚 About"])
    
    with tab1:
        st.subheader("Make Prediction")
        
        col1, col2 = st.columns(2)
        
        with col1:
            date = st.date_input("Date", datetime.now() + timedelta(days=1))
            strike = st.selectbox("Strike Price", 
                                 [25500, 25600, 25700, 25800, 25900, 26000, 26100, 26200],
                                 index=5)
            ce_price = st.number_input("CE Previous", value=120.5, min_value=0.0, step=0.1)
        
        with col2:
            day = st.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", 
                                      "Friday", "Saturday", "Sunday"], index=0)
            pcr = st.slider("PCR", 0.0, 3.0, 0.85, 0.01)
            pe_price = st.number_input("PE Previous", value=85.3, min_value=0.0, step=0.1)
            opcr = st.slider("OPCR", 0.0, 3.0, 0.92, 0.01)
        
        # Market insights
        st.subheader("📊 Market Insights")
        
        ce_pe_ratio = ce_price / max(pe_price, 0.01)
        pcr_status = "Bullish" if pcr < 0.8 else "Bearish" if pcr > 1.2 else "Neutral"
        
        col_insight1, col_insight2, col_insight3 = st.columns(3)
        with col_insight1:
            st.metric("CE/PE Ratio", f"{ce_pe_ratio:.2f}")
        with col_insight2:
            st.metric("PCR Status", pcr_status)
        with col_insight3:
            st.metric("PCR-OPCR", f"{pcr - opcr:.3f}")
        
        # Predict button
        if st.button("🔮 Get AI Prediction", type="primary", use_container_width=True):
            input_data = {
                'Date': date.strftime('%Y-%m-%d'),
                'Day': day,
                'Option Name': str(strike),
                'CE-previous': ce_price,
                'PE-previous': pe_price,
                'PCR': pcr,
                'OPCR': opcr
            }
            
            with st.spinner("Analyzing with AI..."):
                result = predictor.predict_single(input_data)
                
                if result:
                    # Display result
                    st.subheader("🎯 Prediction Result")
                    
                    if result['action'] == 'BUY_CALL':
                        st.success(f"**{result['action']}** at ₹{result['predicted_CE_Buy']}")
                        st.info("Market expects upward movement")
                    else:
                        st.error(f"**{result['action']}** at ₹{result['predicted_PE_Buy']}")
                        st.info("Market expects downward movement")
                    
                    # Confidence
                    conf_pct = result['confidence'] * 100
                    st.progress(result['confidence'], 
                               text=f"Confidence: {conf_pct:.1f}% ({result['model_type']})")
                    
                    # Probabilities
                    col_prob1, col_prob2 = st.columns(2)
                    with col_prob1:
                        st.metric("Call Probability", f"{result['call_probability']*100:.1f}%")
                    with col_prob2:
                        st.metric("Put Probability", f"{result['put_probability']*100:.1f}%")
                    
                    # Trading advice
                    with st.expander("💡 Trading Advice"):
                        if result['confidence'] > 0.7:
                            st.success("**Strong Signal** - Consider position")
                            st.write("Suggested: 2-3% of capital")
                        elif result['confidence'] > 0.5:
                            st.warning("**Moderate Signal** - Trade with caution")
                            st.write("Suggested: 1-2% of capital")
                        else:
                            st.error("**Weak Signal** - Avoid or wait")
                            st.write("Suggested: <1% of capital or skip")
                        
                        st.markdown("**Always:**")
                        st.markdown("- Use stop losses")
                        st.markdown("- Do your own research")
                        st.markdown("- This is not financial advice")
    
    with tab2:
        st.subheader("Model Analysis")
        
        if SKLEARN_AVAILABLE and ml_predictor and ml_predictor.is_trained:
            st.success("✅ ML Model Active")
            st.markdown("**Random Forest Classifier & Regressor**")
            st.markdown("- **Trees:** 50")
            st.markdown("- **Max Depth:** 5")
            st.markdown("- **Features:** 7 indicators")
            
            # Show sample predictions
            st.subheader("Sample Predictions")
            samples = [
                {"PCR": 0.6, "CE": 150, "PE": 80, "Result": "BUY_CALL (Bullish)"},
                {"PCR": 1.5, "CE": 70, "PE": 120, "Result": "BUY_PUT (Bearish)"},
                {"PCR": 0.9, "CE": 100, "PE": 95, "Result": "Neutral (Check ratios)"},
            ]
            
            for sample in samples:
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("PCR", sample["PCR"])
                with col_b:
                    st.metric("CE", sample["CE"])
                with col_c:
                    st.metric("PE", sample["PE"])
                with col_d:
                    st.text(sample["Result"])
        else:
            st.warning("⚠️ Rule-based Analysis")
            st.markdown("**Simple Rules Used:**")
            st.markdown("1. **PCR < 0.8** → BUY_CALL")
            st.markdown("2. **PCR > 1.2** → BUY_PUT")
            st.markdown("3. **PCR 0.8-1.2** → Check CE/PE ratio")
            st.markdown("4. **Price prediction:** 3-5% above previous")
    
    with tab3:
        st.subheader("About This Tool")
        
        st.markdown("""
        ### 🤖 Machine Learning Option Predictor
        
        **Core Technology:**
        - **Random Forest ML Model** trained on your option data
        - **7 Feature Indicators:** CE, PE, PCR, OPCR, Strike, Ratios, Time
        - **Dual Prediction:** Action (Call/Put) + Price
        
        **Why ML is better than rules:**
        1. **Learns patterns** from your historical data
        2. **Handles complex relationships** between indicators
        3. **Adapts** to market changes
        4. **Provides confidence scores** based on data
        5. **More accurate predictions** than simple rules
        
        **Requirements:**
        - `scikit-learn` for ML algorithms
        - `pandas` for data processing
        - `numpy` for calculations
        
        **Note:** This app falls back to rule-based prediction if ML is unavailable,
        but ML provides significantly better results.
        """)
        
        if not SKLEARN_AVAILABLE:
            st.error("""
            ⚠️ **scikit-learn is NOT installed!**
            
            You're using rule-based prediction which is less accurate.
            
            To install scikit-learn on Streamlit Cloud:
            1. Add `scikit-learn==1.3.0` to requirements.txt
            2. Redeploy the app
            3. Wait for dependencies to install
            """)

if __name__ == "__main__":
    main()
