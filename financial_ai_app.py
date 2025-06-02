import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

# Import xgboost to ensure it's available if needed by the loaded model
try:
    import xgboost
except ImportError:
    st.error("XGBoost library not found. Please install it (`pip install xgboost`) for AI features.")
    # You might want to handle this more gracefully, perhaps disabling AI features

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FIXED: ComprehensiveRatioPredictor class with robust feature handling
# ============================================================================

class ComprehensiveRatioPredictor:
    """Predicts all financial ratios using trained models - FIXED VERSION"""

    def __init__(self, models_dict, company_encoder=None):
        self.models = models_dict
        self.le_company = company_encoder
        self.available_ratios = list(models_dict.keys()) if models_dict else []

    def predict_all_ratios(self, input_data, prediction_method='iterative'):
        """Predict all financial ratios from partial input - FIXED LOGIC"""
        results = {}
        input_copy = input_data.copy()

        # Handle company encoding if needed
        if self.le_company and 'Company' in input_copy:
            try:
                # Check if the company is known by the encoder
                if hasattr(self.le_company, 'classes_') and input_copy['Company'] in self.le_company.classes_:
                    input_copy['Company_Encoded'] = self.le_company.transform([input_copy['Company']])[0]
                else:
                    # Handle unknown companies - assign a default encoding (e.g., -1 or 0)
                    # Or handle based on how the model was trained for unseen categories
                    input_copy['Company_Encoded'] = 0 # Assuming 0 is a safe default or represents 'unknown'
                    st.warning(f"Company '{input_copy['Company']}' not recognized by the model encoder. Using default encoding.")
            except Exception as e:
                st.error(f"Error encoding company: {e}")
                input_copy['Company_Encoded'] = 0 # Fallback encoding

        # Convert percentage inputs (assuming they come in as 0-100) to decimals (0-1)
        # This is crucial for consistency with model training data
        percentage_columns = ['Gross_Margin', 'Net_Profit_Margin', 'ROA', 'ROE', 'Debt_to_Assets']
        ratio_columns = ['Current_Ratio', 'Debt_to_Equity']

        for key, value in input_copy.items():
            try:
                # Ensure value is numeric before processing
                numeric_value = pd.to_numeric(value, errors='coerce')
                if pd.isna(numeric_value):
                    input_copy[key] = np.nan # Keep as NaN if conversion fails
                    continue

                # Convert percentage columns (expected range 0-100) to decimal (0-1)
                if key in percentage_columns:
                     # Check if the value looks like a percentage (e.g., > 1 or specific input context)
                     # Assuming custom inputs are 0-100, historical might already be decimal
                     # Let's refine this based on where input_data comes from (CSV vs Custom Input)
                     # For now, assume values > 1 are percentages needing division by 100
                     if numeric_value > 1:
                         input_copy[key] = numeric_value / 100.0
                     else:
                         input_copy[key] = numeric_value # Assume already decimal
                # Keep ratio columns as they are (already numeric)
                elif key in ratio_columns:
                    input_copy[key] = numeric_value
                # Handle other potential numeric columns like Year, Quarter
                elif key in ['Year', 'Quarter', 'Company_Encoded']:
                     input_copy[key] = numeric_value
                # else: handle other non-numeric keys if necessary

            except Exception as e:
                 st.warning(f"Could not process input for {key}: {value}. Error: {e}. Setting to NaN.")
                 input_copy[key] = np.nan


        # Iterative prediction approach
        if prediction_method == 'iterative' and self.models:
            max_iterations = 3
            for iteration in range(max_iterations):
                predicted_in_iteration = False
                for ratio in self.available_ratios:
                    # Predict only if the ratio is missing or NaN
                    if ratio not in input_copy or pd.isna(input_copy.get(ratio)):
                        if ratio in self.models:
                            model_info = self.models[ratio]
                            # Check if model_info is a dict with 'model' and 'features'
                            if isinstance(model_info, dict) and 'model' in model_info:
                                model = model_info['model']
                                features = model_info.get('features', [])

                                # Check if enough features are available
                                available_features_values = {f: input_copy.get(f) for f in features if f in input_copy and pd.notna(input_copy.get(f))}

                                # Define minimum required features (e.g., 50%)
                                min_required_features = max(1, int(len(features) * 0.5))

                                if len(available_features_values) >= min_required_features:
                                    try:
                                        # Prepare feature vector, imputing missing ones (e.g., with 0 or mean)
                                        # Using 0 for simplicity, but mean/median might be better depending on training
                                        feature_vector = [available_features_values.get(f, 0) for f in features]

                                        if hasattr(model, 'predict'):
                                            prediction = model.predict([feature_vector])[0]

                                            # Basic sanity check on prediction (optional)
                                            # e.g., cap unreasonable values
                                            # if ratio in percentage_columns:
                                            #     prediction = max(0, min(1, prediction)) # Cap between 0 and 1

                                            input_copy[ratio] = prediction
                                            predicted_in_iteration = True

                                            # Calculate confidence based on available features
                                            feature_availability_ratio = len(available_features_values) / len(features) if features else 1
                                            if feature_availability_ratio >= 0.8:
                                                confidence = 'High'
                                            elif feature_availability_ratio >= 0.5:
                                                confidence = 'Medium'
                                            else:
                                                confidence = 'Low'

                                            results[ratio] = {
                                                'predicted_value': prediction,
                                                'confidence': confidence,
                                                'iteration': iteration + 1,
                                                'features_used': list(available_features_values.keys()),
                                                'features_missing': [f for f in features if f not in available_features_values]
                                            }
                                    except Exception as e:
                                        st.warning(f"Could not predict {ratio} in iteration {iteration+1}. Error: {e}")
                                        # Ensure the ratio remains NaN or missing if prediction fails
                                        if ratio not in input_copy:
                                             input_copy[ratio] = np.nan
                                else:
                                     # Not enough features to predict this ratio in this iteration
                                     pass
                            else:
                                 st.warning(f"Model information for ratio '{ratio}' is not in the expected format.")
                # If no new predictions were made in this iteration, stop early
                if not predicted_in_iteration:
                    break

        return results, input_copy

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; /* Adjusted size */
        color: #004488; /* Darker blue */
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #333333;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid #004488;
        padding-bottom: 0.3rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }
    .stAlert > div {
         border-radius: 0.5rem; /* Rounded corners for alerts */
    }
    .stButton>button {
        border-radius: 0.5rem;
        background-color: #004488;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #002244;
        color: white;
    }
    /* Make sliders look cleaner */
    .stSlider [data-baseweb="slider"] {
        background-color: #dee2e6;
    }
    .stSlider [data-baseweb="slider"] > div:nth-child(2) {
        background-color: #004488;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 Financial AI Assistant</h1>', unsafe_allow_html=True)
st.markdown("**Saudi Food Sector Investment Analysis System**")
st.markdown("*Analyze Almarai, Savola, and NADEC with AI-powered insights*")

# ============================================================================
# Enhanced AI System Loading with Clearer Status & Fallback Info
# ============================================================================

@st.cache_resource
def load_comprehensive_ai_system():
    """Load the comprehensive AI system with enhanced error handling and status reporting"""
    predictor_path = 'comprehensive_ratio_predictor.pkl'
    encoder_path = 'company_encoder.pkl'
    ai_status = {'status': 'UNKNOWN', 'message': '', 'predictor': None, 'encoder': None, 'models': [], 'model_count': 0}

    try:
        # Attempt to load the main predictor
        comprehensive_predictor = joblib.load(predictor_path)
        ai_status['predictor'] = comprehensive_predictor

        # Attempt to load the company encoder
        try:
            company_encoder = joblib.load(encoder_path)
            ai_status['encoder'] = company_encoder
        except FileNotFoundError:
            ai_status['message'] += f"Optional encoder file '{encoder_path}' not found. Company encoding might be limited. "
            company_encoder = None # Continue without encoder if optional
        except Exception as e:
            ai_status['message'] += f"Error loading company encoder '{encoder_path}': {str(e)[:100]}. Proceeding without it. "
            company_encoder = None

        # Verify the loaded predictor has models
        if hasattr(comprehensive_predictor, 'models') and comprehensive_predictor.models:
            available_models = list(comprehensive_predictor.models.keys())
            ai_status.update({
                'status': 'AI_MODELS_ACTIVE',
                'message': f"✅ Successfully loaded comprehensive AI system with {len(available_models)} models.",
                'models': available_models,
                'model_count': len(available_models)
            })
        else:
            # Predictor loaded but seems empty or invalid
            ai_status.update({
                'status': 'FALLBACK_MODE',
                'message': f"⚠️ Loaded predictor from '{predictor_path}' but found no valid models inside. Using fallback calculations."
            })

    except FileNotFoundError:
        ai_status.update({
            'status': 'FALLBACK_MODE',
            'message': f"❌ Main AI model file '{predictor_path}' not found. Using fallback calculations."
        })
    except ModuleNotFoundError as e:
         # Specific handling for missing dependencies like xgboost
        ai_status.update({
            'status': 'FALLBACK_MODE',
            'message': f"❌ Model loading failed due to missing library: '{e.name}'. Please install required libraries (check requirements.txt). Using fallback calculations."
        })
    except Exception as e:
        # Catch other potential loading errors
        ai_status.update({
            'status': 'FALLBACK_MODE',
            'message': f"❌ Error loading AI model from '{predictor_path}': {str(e)[:150]}. Using fallback calculations."
        })

    # Display status message in the sidebar
    if ai_status['status'] == 'AI_MODELS_ACTIVE':
        st.sidebar.success(ai_status['message'])
        st.sidebar.info(f"Available AI Models: {', '.join(ai_status['models'])}")
    else:
        st.sidebar.error("AI System Status: Using Mathematical Fallbacks")
        st.sidebar.warning(ai_status['message'])

    return ai_status

# ============================================================================
# FIXED: Data loading and cleaning functions with robust error handling
# ============================================================================

@st.cache_data
def load_financial_data():
    """Load and prepare financial data with FIXED cleaning and error handling"""
    possible_filenames = [
        'SavolaAlmaraiNADECFinancialRatiosCSV.csv', # Check local first
        '/home/ubuntu/upload/SavolaAlmaraiNADECFinancialRatiosCSV.csv', # Check upload dir
        'Savola Almarai NADEC Financial Ratios CSV.csv', # With spaces
        'Savola_Almarai_NADEC_Financial_Ratios_CSV.csv', # With underscores
        'financial_data.csv',
        'data.csv'
    ]
    df = None
    loaded_filename = None

    for filename in possible_filenames:
        try:
            df = pd.read_csv(filename)
            loaded_filename = filename
            st.success(f"✅ Data loaded successfully from: {loaded_filename}")
            break
        except FileNotFoundError:
            continue # Try next filename
        except Exception as e:
            st.warning(f"⚠️ Could not read file '{filename}'. Error: {e}")
            continue

    if df is None:
        st.error("❌ Critical Error: Could not find or load the financial data CSV file. App functionality will be limited.")
        # Optionally return sample data or empty dataframe
        # return create_sample_data()
        return pd.DataFrame() # Return empty dataframe to avoid downstream errors

    try:
        # Clean the data with FIXED logic
        df_cleaned = clean_financial_data(df.copy()) # Work on a copy
        st.info(f"Data cleaned: {df_cleaned.shape[0]} records found.")
        return df_cleaned
    except Exception as e:
        st.error(f"❌ Error during data cleaning: {e}. Returning raw data.")
        return df # Return raw data if cleaning fails

def clean_financial_data(df):
    """Clean financial data - Robust percentage/decimal handling"""
    # Standardize column names (lowercase, replace spaces/special chars with underscore)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^\w_]', '', regex=True)
    # Rename specific columns if needed after standardization
    # e.g., df = df.rename(columns={'period_type': 'period_type'})

    # Ensure essential columns exist
    required_cols = ['company', 'year', 'period_type']
    if not all(col in df.columns for col in required_cols):
        st.warning(f"Missing required columns in CSV. Expected: {required_cols}. Found: {list(df.columns)}")
        # Attempt to derive if possible (example below)
        if 'period' in df.columns:
             if 'year' not in df.columns:
                 df['year'] = df['period'].astype(str).str.extract(r'(\d{4})').astype(float).fillna(method='ffill').astype(int)
             if 'period_type' not in df.columns:
                 df['period_type'] = df['period'].astype(str).apply(lambda x: 'Annual' if 'Annual' in x else ('Quarterly' if 'Q' in x else 'Unknown'))
             if 'quarter' not in df.columns:
                 df['quarter'] = df['period'].astype(str).str.extract(r'Q(\d)').fillna(0).astype(int)
        else:
            st.error("Cannot proceed without 'company', 'year', 'period_type' columns or derivable 'period' column.")
            return pd.DataFrame() # Return empty if critical columns missing

    # Define numeric columns (standardized names)
    numeric_columns = [
        'gross_margin', 'net_profit_margin', 'roa', 'roe', 'debt_to_assets',
        'current_ratio', 'debt_to_equity'
    ]
    percentage_columns = ['gross_margin', 'net_profit_margin', 'roa', 'roe', 'debt_to_assets']

    for col in numeric_columns:
        if col in df.columns:
            # 1. Convert to string first to handle mixed types and clean
            df[col] = df[col].astype(str)
            # 2. Remove common non-numeric characters (%, ,, spaces)
            df[col] = df[col].str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
            # 3. Convert to numeric, coercing errors to NaN
            df[col] = pd.to_numeric(df[col], errors='coerce')

            # 4. Handle percentage conversion (if applicable)
            if col in percentage_columns:
                # Apply conversion only where the value is likely a percentage (e.g., > 1)
                # This assumes decimals are stored as < 1 (e.g., 0.15 for 15%)
                # and percentages as > 1 (e.g., 15 for 15%)
                df[col] = df[col].apply(lambda x: x / 100.0 if pd.notna(x) and abs(x) > 1 else x)
        else:
            st.warning(f"Numeric column '{col}' not found in the data.")
            df[col] = np.nan # Add missing column as NaN

    # Ensure 'year' is integer
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

    # Ensure 'company' is string and stripped
    if 'company' in df.columns:
        df['company'] = df['company'].astype(str).str.strip()

    # Drop rows where essential identifiers are missing
    df.dropna(subset=['company', 'year', 'period_type'], inplace=True)

    return df

# Sample data function (optional, if CSV loading fails)
def create_sample_data():
    # ... (keep existing sample data generation logic if needed) ...
    st.warning("Using generated sample data. Results may not be accurate.")
    # Ensure sample data matches the cleaned format (decimals for percentages)
    # ...
    pass # Placeholder

# ============================================================================
# Enhanced Financial AI Class with Fallback Logic
# ============================================================================

class EnhancedFinancialAI:
    def __init__(self, ai_system_status, financial_data):
        self.ai_status = ai_system_status
        self.df = financial_data
        self.predictor = ai_system_status.get('predictor')
        self.encoder = ai_system_status.get('encoder')

    def _get_historical_data(self, company, year, period_type):
        """Retrieve historical data for a specific company, year, and period."""
        if self.df.empty:
            return pd.Series(dtype='float64') # Return empty series if no data
        
        # Filter based on standardized column names
        data = self.df[
            (self.df['company'].str.lower() == company.lower()) &
            (self.df['year'] == year) &
            (self.df['period_type'].str.lower() == period_type.lower())
        ]
        if not data.empty:
            # Return the first match as a Series
            return data.iloc[0]
        else:
            # st.warning(f"No historical data found for {company}, {year}, {period_type}")
            return pd.Series(dtype='float64') # Return empty series if no match

    def _mathematical_fallback_prediction(self, company, year, period_type):
        """Basic fallback using historical averages or simple rules."""
        # Example: Return last year's annual data or overall average
        last_year_data = self._get_historical_data(company, year - 1, 'Annual')
        if not last_year_data.empty:
            return last_year_data
        # Fallback further: company average
        company_avg = self.df[self.df['company'].str.lower() == company.lower()].select_dtypes(include=np.number).mean()
        if not company_avg.empty:
            return company_avg
        # Ultimate fallback: empty series
        return pd.Series(dtype='float64')

    def predict_ratios(self, company, year, period_type='Annual'):
        """Predict ratios using AI if available, otherwise use fallback."""
        if self.ai_status['status'] == 'AI_MODELS_ACTIVE' and self.predictor:
            # Prepare input data for AI model
            # Needs historical data as features, potentially
            # This part depends heavily on how the 'comprehensive_ratio_predictor.pkl' expects input
            # Assuming it needs current knowns + potentially lagged features
            historical_data = self._get_historical_data(company, year, period_type)
            if historical_data.empty:
                 historical_data = self._get_historical_data(company, year -1, 'Annual') # Try last year
            
            input_dict = historical_data.to_dict() if not historical_data.empty else {}
            input_dict['Company'] = company # Ensure Company name is passed
            input_dict['Year'] = year
            # Add other necessary features if not present (e.g., Quarter)
            input_dict.setdefault('Quarter', 0 if period_type == 'Annual' else 1) # Example default

            try:
                # Use the predictor's method
                predictions, filled_data = self.predictor.predict_all_ratios(input_dict)
                # Return the dictionary containing all available/predicted ratios
                # st.write("AI Prediction Input:", input_dict)
                # st.write("AI Prediction Output (Filled Data):", filled_data)
                # st.write("AI Prediction Details:", predictions)
                return filled_data # Return the dict with original + predicted values
            except Exception as e:
                st.error(f"Error during AI prediction: {e}. Falling back to mathematical methods.")
                # Fallback on prediction error
                fallback_data = self._mathematical_fallback_prediction(company, year, period_type)
                return fallback_data.to_dict() if not fallback_data.empty else {}
        else:
            # Use mathematical fallback if AI is not active
            fallback_data = self._mathematical_fallback_prediction(company, year, period_type)
            return fallback_data.to_dict() if not fallback_data.empty else {}

    def perform_health_check(self, company, year, period_type='Annual'):
        """Perform financial health check based on key ratios."""
        data_dict = self.predict_ratios(company, year, period_type)
        if not data_dict:
             return "Could not retrieve or predict data for health check.", {}

        # Define thresholds (example)
        thresholds = {
            'current_ratio': (1.5, 2.5), # Ideal range
            'debt_to_equity': (0.5, 1.5),
            'roa': (0.05, float('inf')), # Minimum acceptable
            'net_profit_margin': (0.1, float('inf'))
        }
        issues = []
        # Use .get(key, default_value) for safety
        cr = data_dict.get('current_ratio', 0)
        de = data_dict.get('debt_to_equity', float('inf'))
        roa = data_dict.get('roa', 0)
        npm = data_dict.get('net_profit_margin', 0)

        if not (thresholds['current_ratio'][0] <= cr <= thresholds['current_ratio'][1]):
            issues.append(f"Current Ratio ({cr:.2f}) outside ideal range {thresholds['current_ratio']}")
        if not (thresholds['debt_to_equity'][0] <= de <= thresholds['debt_to_equity'][1]):
            issues.append(f"Debt-to-Equity ({de:.2f}) outside ideal range {thresholds['debt_to_equity']}")
        if roa < thresholds['roa'][0]:
            issues.append(f"ROA ({roa:.2%}) below minimum {thresholds['roa'][0]:.1%}")
        if npm < thresholds['net_profit_margin'][0]:
            issues.append(f"Net Profit Margin ({npm:.2%}) below minimum {thresholds['net_profit_margin'][0]:.1%}")

        if not issues:
            summary = f"✅ Strong Financial Health for {company} ({year} {period_type}). All key ratios within healthy ranges."
        else:
            summary = f"⚠️ Potential Financial Health Concerns for {company} ({year} {period_type}):\n- " + "\n- ".join(issues)

        return summary, data_dict

    def generate_recommendation(self, company, year, period_type='Annual'):
        """Generate investment recommendation based on AI/fallback data."""
        data_dict = self.predict_ratios(company, year, period_type)
        if not data_dict:
            return "Insufficient data for recommendation.", "N/A", {} 

        # Simple rule-based recommendation (example)
        score = 0
        roe = data_dict.get('roe', 0)
        roa = data_dict.get('roa', 0)
        npm = data_dict.get('net_profit_margin', 0)
        de = data_dict.get('debt_to_equity', 1.0) # Default to 1 if missing

        if roe > 0.15: score += 2
        elif roe > 0.10: score += 1
        if roa > 0.07: score += 2
        elif roa > 0.04: score += 1
        if npm > 0.12: score += 1
        if de < 1.0: score += 1
        elif de > 2.0: score -=1

        if score >= 5:
            rec = "Strong Buy"
            rec_style = "recommendation-buy"
        elif score >= 3:
            rec = "Buy"
            rec_style = "recommendation-buy"
        elif score >= 1:
            rec = "Hold"
            rec_style = "recommendation-hold"
        else:
            rec = "Sell"
            rec_style = "recommendation-sell"
        
        # Add AI confidence if available
        if self.ai_status['status'] == 'AI_MODELS_ACTIVE':
             rec += " (AI Assisted)"
        else:
             rec += " (Fallback Based)"

        return rec, rec_style, data_dict

# ============================================================================
# Load Data and AI System
# ============================================================================

ai_system_info = load_comprehensive_ai_system()
financial_data_df = load_financial_data()

# Initialize the AI/Fallback Handler
if not financial_data_df.empty:
    financial_ai = EnhancedFinancialAI(ai_system_info, financial_data_df)
    companies = sorted(financial_data_df['company'].unique())
    years = sorted(financial_data_df['year'].unique(), reverse=True)
    periods = ['Annual', 'Quarterly'] # Assuming these are the main types
else:
    # Handle case where data loading failed completely
    st.error("Application cannot function without financial data.")
    companies = []
    years = []
    periods = []
    financial_ai = None # Ensure financial_ai is None if data is missing

# ============================================================================
# Sidebar Navigation
# ============================================================================
st.sidebar.title("🧭 Navigation")
analysis_type = st.sidebar.selectbox(
    "Choose Analysis Type:",
    [
        "Dashboard",
        "Individual Company Analysis",
        "Company Comparison",
        "Financial Health Check",
        "Ratio Prediction",
        "Custom Financial Analysis"
    ]
)

# ============================================================================
# Main Content Area - Based on Selection
# ============================================================================

if financial_ai is None:
    st.error("Financial data could not be loaded. Please check the CSV file and configuration.")

# --- Dashboard --- (Example Structure)
elif analysis_type == "Dashboard":
    st.markdown("<h2 class='sub-header'>📊 Financial AI Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Overview of Saudi Food Sector Companies (Almarai, Savola, NADEC)")

    if not financial_data_df.empty:
        latest_year = financial_data_df['year'].max()
        st.info(f"Displaying latest available annual data ({latest_year})")
        latest_annual_data = financial_data_df[
            (financial_data_df['year'] == latest_year) &
            (financial_data_df['period_type'].str.lower() == 'annual')
        ]

        if not latest_annual_data.empty:
            cols = st.columns(len(companies))
            for i, company in enumerate(companies):
                with cols[i]:
                    st.subheader(company)
                    company_data = latest_annual_data[latest_annual_data['company'] == company]
                    if not company_data.empty:
                        roe = company_data['roe'].iloc[0]
                        roa = company_data['roa'].iloc[0]
                        npm = company_data['net_profit_margin'].iloc[0]
                        st.metric("ROE", f"{roe:.2%}")
                        st.metric("ROA", f"{roa:.2%}")
                        st.metric("Net Profit Margin", f"{npm:.2%}")
                        
                        # Get recommendation
                        rec, rec_style, _ = financial_ai.generate_recommendation(company, latest_year, 'Annual')
                        st.markdown(f'<div class="{rec_style}">{rec}</div>', unsafe_allow_html=True)
                    else:
                        st.warning("No data for latest year.")
            
            # Sector Trends Plot
            st.markdown("<h3 class='sub-header' style='margin-top: 2rem;'>📈 Sector ROE Trends (Annual)</h3>", unsafe_allow_html=True)
            annual_data = financial_data_df[financial_data_df['period_type'].str.lower() == 'annual']
            fig_trends = px.line(annual_data, x='year', y='roe', color='company',
                                 title="Return on Equity (ROE) Over Time",
                                 labels={'year': 'Year', 'roe': 'ROE (%)', 'company': 'Company'},
                                 markers=True)
            fig_trends.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig_trends, use_container_width=True)
        else:
            st.warning(f"No annual data found for the latest year ({latest_year}).")
    else:
        st.warning("No financial data available for dashboard.")

# --- Individual Company Analysis --- (FIXED N/A)
elif analysis_type == "Individual Company Analysis":
    st.markdown("<h2 class='sub-header'>🏢 Individual Company Analysis</h2>", unsafe_allow_html=True)
    st.markdown("Deep dive into specific company performance.")

    if companies and years and periods:
        sel_company = st.selectbox("Select Company:", companies)
        sel_year = st.selectbox("Select Year:", years)
        sel_period = st.selectbox("Select Period:", periods)

        if st.button("📊 Generate Analysis"):
            st.subheader(f"Analysis for {sel_company} - {sel_year} {sel_period}")
            
            # Use the unified function to get data (historical or predicted)
            data_dict = financial_ai.predict_ratios(sel_company, sel_year, sel_period)

            if data_dict:
                # Define ratios to display
                profitability_ratios = ['gross_margin', 'net_profit_margin', 'roa', 'roe']
                health_ratios = ['current_ratio', 'debt_to_equity', 'debt_to_assets']

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Profitability Ratios**")
                    for ratio in profitability_ratios:
                        value = data_dict.get(ratio)
                        # FIXED: Display N/A properly, format percentages
                        display_value = f"{value:.2%}" if pd.notna(value) and ratio != 'current_ratio' and ratio != 'debt_to_equity' else (f"{value:.2f}" if pd.notna(value) else "N/A")
                        st.metric(label=ratio.replace('_', ' ').title(), value=display_value)
                
                with col2:
                    st.markdown("**Financial Health Ratios**")
                    for ratio in health_ratios:
                        value = data_dict.get(ratio)
                        # FIXED: Display N/A properly, format percentages/ratios
                        display_value = f"{value:.2%}" if pd.notna(value) and ratio == 'debt_to_assets' else (f"{value:.2f}" if pd.notna(value) else "N/A")
                        st.metric(label=ratio.replace('_', ' ').title(), value=display_value)
                
                # Display AI prediction details if available
                # This part needs refinement based on the actual structure of 'predictions' from ComprehensiveRatioPredictor
                # if ai_system_info['status'] == 'AI_MODELS_ACTIVE' and 'predictions' in data_dict: # Assuming predictions are stored
                #     st.markdown("**AI Prediction Details**")
                #     st.json(data_dict['predictions']) # Display raw prediction info

            else:
                st.warning(f"Could not retrieve or predict data for {sel_company}, {sel_year}, {sel_period}.")
    else:
        st.warning("No data loaded. Cannot perform individual analysis.")

# --- Company Comparison --- (Example Structure)
elif analysis_type == "Company Comparison":
    st.markdown("<h2 class='sub-header'>🆚 Company Comparison</h2>", unsafe_allow_html=True)
    st.markdown("Side-by-side financial performance comparison.")

    if companies and years and periods:
        comp_year = st.selectbox("Comparison Year:", years, key="comp_year")
        comp_period = st.selectbox("Comparison Period:", periods, key="comp_period")
        comp_ratio = st.selectbox("Ratio to Compare:", 
                                  financial_data_df.select_dtypes(include=np.number).columns.drop(['year', 'quarter'], errors='ignore'), 
                                  key="comp_ratio", index=5) # Default to ROE index

        if st.button("🔍 Compare Companies"):
            st.subheader(f"Comparison for {comp_year} {comp_period} - Ratio: {comp_ratio.replace('_', ' ').title()}")
            comparison_data = []
            for company in companies:
                # Use predict_ratios to get consistent data (historical or predicted)
                data_dict = financial_ai.predict_ratios(company, comp_year, comp_period)
                ratio_value = data_dict.get(comp_ratio, np.nan) # Get the specific ratio
                comparison_data.append({
                    'Company': company,
                    comp_ratio: ratio_value
                })
            
            comparison_df = pd.DataFrame(comparison_data).set_index('Company')
            comparison_df.dropna(inplace=True)

            if not comparison_df.empty:
                st.dataframe(comparison_df.style.format({comp_ratio: "{:.2%}" if comp_ratio in percentage_columns else "{:.2f}"})) 
                
                # Visual Comparison
                st.markdown(f"<h3 class='sub-header' style='margin-top: 1rem;'>📊 Visual Comparison - {comp_ratio.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
                fig_comp = px.bar(comparison_df, y=comp_ratio, 
                                  title=f"{comp_ratio.replace('_', ' ').title()} Comparison - {comp_year} {comp_period}",
                                  labels={comp_ratio: comp_ratio.replace('_', ' ').title()},
                                  text_auto=True)
                fig_comp.update_layout(yaxis_tickformat='.1%' if comp_ratio in percentage_columns else '.2f')
                st.plotly_chart(fig_comp, use_container_width=True)
            else:
                st.warning(f"No comparable data found for {comp_ratio} in {comp_year} {comp_period}.")
    else:
        st.warning("No data loaded. Cannot perform company comparison.")

# --- Financial Health Check --- (Example Structure)
elif analysis_type == "Financial Health Check":
    st.markdown("<h2 class='sub-header'>🩺 Financial Health Assessment</h2>", unsafe_allow_html=True)
    st.markdown("Comprehensive health analysis using multiple financial indicators.")

    if companies and years and periods:
        hc_company = st.selectbox("Select Company for Health Check:", companies, key="hc_company")
        hc_year = st.selectbox("Select Year:", years, key="hc_year")
        hc_period = st.selectbox("Select Period:", periods, key="hc_period")

        if st.button("🔍 Perform Health Check"):
            st.subheader(f"Health Check Results for {hc_company} - {hc_year} {hc_period}")
            summary, data_dict = financial_ai.perform_health_check(hc_company, hc_year, hc_period)
            
            if data_dict:
                st.markdown(summary) # Display the summary (potential issues)
                # Optionally display the data used for the check
                st.markdown("**Data Used for Check:**")
                # Filter to show only relevant ratios
                display_ratios = ['current_ratio', 'debt_to_equity', 'roa', 'net_profit_margin', 'roe', 'debt_to_assets']
                display_dict = {k: data_dict[k] for k in display_ratios if k in data_dict}
                st.json(display_dict) # Show the key ratios used
            else:
                 st.error(f"Could not perform health check for {hc_company}, {hc_year}, {hc_period}. Data unavailable.")
    else:
        st.warning("No data loaded. Cannot perform health check.")

# --- Ratio Prediction --- (Example Structure)
elif analysis_type == "Ratio Prediction":
    st.markdown("<h2 class='sub-header'>🔮 Financial Ratio Prediction</h2>", unsafe_allow_html=True)
    st.markdown("Predict future financial performance using AI and historical trends.")

    if companies and years:
        pred_company = st.selectbox("Company for Prediction:", companies, key="pred_company")
        # Predict for the next year
        current_max_year = max(years) if years else datetime.now().year
        pred_year = st.number_input("Predict for Year:", min_value=current_max_year + 1, value=current_max_year + 1, step=1, key="pred_year")
        pred_method = "AI Model (Advanced)" if ai_system_info['status'] == 'AI_MODELS_ACTIVE' else "Mathematical Fallback"
        st.info(f"Prediction Method: {pred_method}")

        # Display recent historical data
        st.markdown(f"**Recent Historical Data: {pred_company} (Annual)**")
        hist_data = financial_data_df[
            (financial_data_df['company'] == pred_company) &
            (financial_data_df['period_type'].str.lower() == 'annual')
        ].sort_values('year', ascending=False).head(3)
        
        if not hist_data.empty:
            cols = st.columns(len(hist_data))
            for i, year_hist in enumerate(hist_data['year']):
                with cols[i]:
                    st.metric("Year", str(year_hist))
                    roe_hist = hist_data[hist_data['year'] == year_hist]['roe'].iloc[0]
                    roa_hist = hist_data[hist_data['year'] == year_hist]['roa'].iloc[0]
                    st.metric("ROE", f"{roe_hist:.2%}")
                    st.metric("ROA", f"{roa_hist:.2%}")
        else:
             st.warning("No recent annual data found for context.")

        if st.button("🚀 Generate Predictions"):
            st.subheader(f"Predicted Ratios for {pred_company} - {pred_year} (Annual)")
            predicted_data_dict = financial_ai.predict_ratios(pred_company, pred_year, 'Annual')

            if predicted_data_dict:
                # Display key predicted ratios
                pred_ratios_disp = ['roe', 'roa', 'net_profit_margin', 'current_ratio', 'debt_to_equity']
                cols_pred = st.columns(len(pred_ratios_disp))
                for i, ratio in enumerate(pred_ratios_disp):
                     with cols_pred[i]:
                        value = predicted_data_dict.get(ratio)
                        display_value = f"{value:.2%}" if pd.notna(value) and ratio in percentage_columns else (f"{value:.2f}" if pd.notna(value) else "N/A")
                        st.metric(label=ratio.replace('_', ' ').title(), value=display_value)
                
                # Optionally show more details or confidence if available from AI
                # st.write("Full Predicted Data:", predicted_data_dict)
            else:
                st.error(f"Could not generate predictions for {pred_company}, {pred_year}.")
    else:
        st.warning("No data loaded. Cannot perform ratio prediction.")

# --- Custom Financial Analysis --- (FIXED Input Handling)
elif analysis_type == "Custom Financial Analysis":
    st.markdown("<h2 class='sub-header'>⚙️ Custom Financial Analysis</h2>", unsafe_allow_html=True)
    st.markdown("Input your own financial ratios for AI analysis (or fallback calculation).")

    if companies and years:
        st.markdown("**Enter Financial Data**")
        col1, col2 = st.columns(2)
        
        with col1:
            cust_company = st.text_input("Company Type (e.g., Almarai, Savola, NADEC, or Other):", value="Almarai", key="cust_comp")
            cust_year = st.number_input("Year:", min_value=2010, max_value=datetime.now().year + 5, value=datetime.now().year, key="cust_year")
            cust_period = st.selectbox("Period:", ["Annual", "Quarterly"], key="cust_period")
            
            st.markdown("**Profitability Ratios (%)**")
            # FIXED: Input as percentage (0-100), will be converted later
            cust_gross_margin = st.slider("Gross Margin (%):", 0, 100, 38, key="cust_gm")
            cust_npm = st.slider("Net Profit Margin (%):", -50, 100, 10, key="cust_npm") # Allow negative NPM
            cust_roa = st.slider("Return on Assets (ROA %):", -50, 50, 8, key="cust_roa")

        with col2:
            st.markdown("**Financial Health Ratios**")
            cust_current_ratio = st.slider("Current Ratio:", 0.0, 5.0, 1.5, step=0.01, key="cust_cr")
            cust_debt_equity = st.slider("Debt-to-Equity:", 0.0, 5.0, 0.8, step=0.01, key="cust_de")
            cust_debt_assets = st.slider("Debt-to-Assets (%):", 0, 100, 45, key="cust_da")
            
            st.markdown("**Optional**")
            cust_roe_input_type = st.radio("Provide ROE?", ["Manual ROE (%)", "Use AI/Fallback to Predict ROE"], key="cust_roe_type")
            cust_roe = None
            if cust_roe_input_type == "Manual ROE (%)":
                cust_roe = st.slider("Manual ROE (%):", -50, 100, 12, key="cust_roe_manual")

        if st.button("📈 Analyze Custom Data"):
            st.subheader(f"Custom Analysis Results for {cust_company} - {cust_year} {cust_period}")
            
            # Prepare input dictionary
            custom_input = {
                'Company': cust_company,
                'Year': cust_year,
                'Period_Type': cust_period,
                # FIXED: Pass slider values directly, conversion happens inside predictor/handler
                'Gross_Margin': cust_gross_margin,
                'Net_Profit_Margin': cust_npm,
                'ROA': cust_roa,
                'Current_Ratio': cust_current_ratio,
                'Debt_to_Equity': cust_debt_equity,
                'Debt_to_Assets': cust_debt_assets,
                # Pass ROE only if manually provided
                'ROE': cust_roe if cust_roe is not None else np.nan # Use NaN if not provided
            }

            # Use the predictor/fallback mechanism
            # The predict_all_ratios method inside ComprehensiveRatioPredictor handles conversion and prediction
            if ai_system_info['status'] == 'AI_MODELS_ACTIVE' and financial_ai.predictor:
                st.info("Using AI Model for analysis...")
                try:
                    predictions, analyzed_data = financial_ai.predictor.predict_all_ratios(custom_input)
                    
                    st.markdown("**Analyzed / Predicted Ratios:**")
                    # Display the results (original + predicted)
                    disp_ratios = ['roe', 'roa', 'net_profit_margin', 'gross_margin', 'current_ratio', 'debt_to_equity', 'debt_to_assets']
                    results_disp = {}
                    for r in disp_ratios:
                        val = analyzed_data.get(r)
                        is_predicted = r in predictions
                        conf = predictions[r]['confidence'] if is_predicted else 'N/A'
                        val_str = f"{val:.2%}" if pd.notna(val) and r in percentage_columns else (f"{val:.2f}" if pd.notna(val) else "N/A")
                        results_disp[r.replace('_',' ').title()] = f"{val_str} {'(Predicted - ' + conf + ')' if is_predicted else '(Input)'}"
                    st.json(results_disp)
                    
                    # Generate recommendation based on the analyzed data
                    rec, rec_style, _ = financial_ai.generate_recommendation(cust_company, cust_year, cust_period) # This might need adjustment if generate_recommendation relies on historical data not present here
                    st.markdown(f"**Recommendation:** <span class='{rec_style}' style='padding: 5px; border-radius: 5px;'>{rec}</span>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error during custom AI analysis: {e}")
                    st.warning("Falling back to basic display of input data.")
                    st.json({k: v for k, v in custom_input.items() if k not in ['Company', 'Year', 'Period_Type']})
            else:
                st.warning("AI Model not active. Displaying input data only. Fallback calculations for custom input are limited.")
                # Just display the input data as fallback for custom analysis is complex
                st.json({k: v for k, v in custom_input.items() if k not in ['Company', 'Year', 'Period_Type']})
    else:
        st.warning("No data loaded. Cannot perform custom analysis.")

# ============================================================================
# Footer / Additional Info
# ============================================================================
st.markdown("---")
st.markdown("<h3 class='sub-header'>ℹ️ Financial AI Assistant Information</h3>", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("**Data Coverage**")
    if not financial_data_df.empty:
        min_year, max_year = financial_data_df['year'].min(), financial_data_df['year'].max()
        st.write(f"- Period: {min_year}-{max_year}")
        st.write(f"- Companies: {len(companies)}")
        st.write(f"- Records: {len(financial_data_df)}")
    else:
        st.write("- Data not loaded.")
with col2:
    st.markdown("**AI Models**")
    st.write(f"- Status: {ai_system_info['status'].replace('_', ' ')}")
    if ai_system_info['status'] == 'AI_MODELS_ACTIVE':
        st.write(f"- Models Loaded: {ai_system_info['model_count']}")
        # st.write(f"- Available Ratios: {', '.join(ai_system_info['models'])}")
    else:
        st.write(f"- Reason: {ai_system_info['message']}")
with col3:
    st.markdown("**Capabilities**")
    st.write("- ROE Prediction")
    st.write("- Investment Recommendations")
    st.write("- Company Comparison")
    st.write("- Financial Health Assessment")
    st.write("- Custom Analysis")

with st.expander("Technical Details"):
    st.write("This application uses machine learning models (potentially XGBoost, Scikit-learn) trained on historical financial data to provide insights and predictions. When AI models fail to load or during prediction errors, it falls back to basic mathematical calculations or historical data retrieval.")
    st.write("Data Cleaning involves standardizing column names, handling mixed percentage/decimal formats, and converting data types.")
    st.write("Prediction confidence (High, Medium, Low) is estimated based on the availability of required input features for the AI model.")

st.markdown("<hr style='margin-top: 2rem;'>", unsafe_allow_html=True)
st.caption("Saudi Food Sector Financial AI Assistant | Powered by Advanced Machine Learning")

