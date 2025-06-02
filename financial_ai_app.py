import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings

# --- Streamlit Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Check for Optional Dependencies (like XGBoost) AFTER page config ---
xgboost_present = False
xgboost_error_message = ""
try:
    import xgboost
    xgboost_present = True
except ImportError:
    xgboost_error_message = "XGBoost library not found. AI prediction features requiring XGBoost will use fallbacks. Please install it (`pip install xgboost`) for full AI capabilities."
    # Display the error prominently after page config
    st.error(xgboost_error_message)

warnings.filterwarnings("ignore")

# ============================================================================
# FIXED: ComprehensiveRatioPredictor class with robust feature handling
# ============================================================================

class ComprehensiveRatioPredictor:
    """Predicts all financial ratios using trained models - FIXED VERSION"""

    def __init__(self, models_dict, company_encoder=None):
        self.models = models_dict
        self.le_company = company_encoder
        self.available_ratios = list(models_dict.keys()) if models_dict else []

    def predict_all_ratios(self, input_data, prediction_method="iterative"):
        """Predict all financial ratios from partial input - FIXED LOGIC"""
        results = {}
        input_copy = input_data.copy()

        # Handle company encoding if needed
        if self.le_company and "Company" in input_copy:
            try:
                # Check if the company is known by the encoder
                if hasattr(self.le_company, "classes_") and input_copy["Company"] in self.le_company.classes_:
                    input_copy["Company_Encoded"] = self.le_company.transform([input_copy["Company"]])[0]
                else:
                    # Handle unknown companies - assign a default encoding (e.g., -1 or 0)
                    input_copy["Company_Encoded"] = 0 # Assuming 0 is a safe default
                    # st.warning(f"Company \'{input_copy['Company']}\' not recognized by the model encoder. Using default encoding.") # Avoid st calls inside class
            except Exception as e:
                # st.error(f"Error encoding company: {e}") # Avoid st calls inside class
                input_copy["Company_Encoded"] = 0 # Fallback encoding

        # Convert percentage inputs (assuming they come in as 0-100) to decimals (0-1)
        percentage_columns = ["Gross_Margin", "Net_Profit_Margin", "ROA", "ROE", "Debt_to_Assets"]
        ratio_columns = ["Current_Ratio", "Debt_to_Equity"]

        for key, value in input_copy.items():
            try:
                # Ensure value is numeric before processing
                numeric_value = pd.to_numeric(value, errors="coerce")
                if pd.isna(numeric_value):
                    input_copy[key] = np.nan # Keep as NaN if conversion fails
                    continue

                # Convert percentage columns (expected range 0-100) to decimal (0-1)
                if key in percentage_columns:
                     if abs(numeric_value) > 1: # Check absolute value in case of negative percentages like -10%
                         input_copy[key] = numeric_value / 100.0
                     else:
                         input_copy[key] = numeric_value # Assume already decimal
                # Keep ratio columns as they are (already numeric)
                elif key in ratio_columns:
                    input_copy[key] = numeric_value
                # Handle other potential numeric columns like Year, Quarter
                elif key in ["Year", "Quarter", "Company_Encoded"]:
                     input_copy[key] = numeric_value

            except Exception as e:
                 # st.warning(f"Could not process input for {key}: {value}. Error: {e}. Setting to NaN.") # Avoid st calls
                 input_copy[key] = np.nan

        # Iterative prediction approach
        if prediction_method == "iterative" and self.models:
            max_iterations = 3
            for iteration in range(max_iterations):
                predicted_in_iteration = False
                for ratio in self.available_ratios:
                    # Predict only if the ratio is missing or NaN
                    if ratio not in input_copy or pd.isna(input_copy.get(ratio)):
                        if ratio in self.models:
                            model_info = self.models[ratio]
                            if isinstance(model_info, dict) and "model" in model_info:
                                model = model_info["model"]
                                features = model_info.get("features", [])

                                # Check if enough features are available
                                available_features_values = {f: input_copy.get(f) for f in features if f in input_copy and pd.notna(input_copy.get(f))}
                                min_required_features = max(1, int(len(features) * 0.5))

                                if len(available_features_values) >= min_required_features:
                                    try:
                                        feature_vector = [available_features_values.get(f, 0) for f in features]
                                        if hasattr(model, "predict"):
                                            # Check if the model is an XGBoost model and if XGBoost is present
                                            model_type_str = str(type(model)).lower()
                                            if "xgboost" in model_type_str and not xgboost_present:
                                                 # Cannot predict with this model, skip
                                                 # st.warning(f"Skipping prediction for {ratio}: XGBoost model requires missing library.") # Avoid st calls
                                                 continue

                                            prediction = model.predict([feature_vector])[0]
                                            input_copy[ratio] = prediction
                                            predicted_in_iteration = True

                                            feature_availability_ratio = len(available_features_values) / len(features) if features else 1
                                            confidence = "High" if feature_availability_ratio >= 0.8 else ("Medium" if feature_availability_ratio >= 0.5 else "Low")

                                            results[ratio] = {
                                                "predicted_value": prediction,
                                                "confidence": confidence,
                                                "iteration": iteration + 1,
                                                "features_used": list(available_features_values.keys()),
                                                "features_missing": [f for f in features if f not in available_features_values]
                                            }
                                    except Exception as e:
                                        # st.warning(f"Could not predict {ratio} in iteration {iteration+1}. Error: {e}") # Avoid st calls
                                        if ratio not in input_copy:
                                             input_copy[ratio] = np.nan
                            # else: # Model info not in expected format - handled in loading
                                 # st.warning(f"Model information for ratio \'{ratio}\' is not in the expected format.")
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
    .recommendation-buy {
        background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; padding: 0.5rem; border-radius: 0.5rem; display: inline-block; margin-top: 5px;
    }
    .recommendation-sell {
        background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; padding: 0.5rem; border-radius: 0.5rem; display: inline-block; margin-top: 5px;
    }
    .recommendation-hold {
        background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; padding: 0.5rem; border-radius: 0.5rem; display: inline-block; margin-top: 5px;
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
    predictor_path = "comprehensive_ratio_predictor.pkl"
    encoder_path = "company_encoder.pkl"
    ai_status = {"status": "UNKNOWN", "message": "", "predictor": None, "encoder": None, "models": [], "model_count": 0}

    try:
        # Attempt to load the main predictor
        comprehensive_predictor = joblib.load(predictor_path)
        ai_status["predictor"] = comprehensive_predictor

        # Attempt to load the company encoder
        try:
            company_encoder = joblib.load(encoder_path)
            ai_status["encoder"] = company_encoder
        except FileNotFoundError:
            ai_status["message"] += f"Optional encoder file '{encoder_path}' not found. "
            company_encoder = None
        except Exception as e:
            ai_status["message"] += f"Error loading company encoder '{encoder_path}': {str(e)[:100]}. "
            company_encoder = None

        # Verify the loaded predictor has models
        if hasattr(comprehensive_predictor, "models") and comprehensive_predictor.models:
            available_models = list(comprehensive_predictor.models.keys())
            ai_status.update({
                "status": "AI_MODELS_ACTIVE",
                "message": f"✅ Successfully loaded AI system with {len(available_models)} models.",
                "models": available_models,
                "model_count": len(available_models)
            })
        else:
            ai_status.update({
                "status": "FALLBACK_MODE",
                "message": f"⚠️ Loaded predictor from '{predictor_path}' but found no valid models inside. Using fallback calculations."
            })

    except FileNotFoundError:
        ai_status.update({
            "status": "FALLBACK_MODE",
            "message": f"❌ Main AI model file '{predictor_path}' not found. Using fallback calculations."
        })
    except ModuleNotFoundError as e:
        # Specific handling for missing dependencies like xgboost
        # The main error message is already shown via st.error() at the top
        ai_status.update({
            "status": "FALLBACK_MODE",
            "message": f"❌ Model loading failed due to missing library: '{e.name}'. Using fallback calculations."
        })
    except Exception as e:
        ai_status.update({
            "status": "FALLBACK_MODE",
            "message": f"❌ Error loading AI model from '{predictor_path}': {str(e)[:150]}. Using fallback calculations."
        })

    # Display status message in the sidebar (avoid duplicating xgboost error)
    if ai_status["status"] == "AI_MODELS_ACTIVE":
        st.sidebar.success(ai_status["message"])
        st.sidebar.info(f"Available AI Models: {', '.join(ai_status['models'])}")
    else:
        st.sidebar.error("AI System Status: Using Mathematical Fallbacks")
        # Show the specific loading error, unless it was the already displayed xgboost error
        if "missing library: 'xgboost'" not in ai_status["message"]:
             st.sidebar.warning(ai_status["message"])
        elif not xgboost_present: # If xgboost was the issue, confirm fallback
             st.sidebar.warning("Using fallback calculations due to missing XGBoost.")
        else: # Other loading error
             st.sidebar.warning(ai_status["message"])


    return ai_status

# ============================================================================
# FIXED: Data loading and cleaning functions with robust error handling
# ============================================================================

@st.cache_data
def load_financial_data():
    """Load and prepare financial data with FIXED cleaning and error handling"""
    possible_filenames = [
        "SavolaAlmaraiNADECFinancialRatiosCSV.csv", # Check local first
        "/home/ubuntu/upload/SavolaAlmaraiNADECFinancialRatiosCSV.csv", # Check upload dir if needed
        "Savola Almarai NADEC Financial Ratios CSV.csv", # With spaces
        "Savola_Almarai_NADEC_Financial_Ratios_CSV.csv", # With underscores
        "financial_data.csv",
        "data.csv"
    ]
    df = None
    loaded_filename = None

    for filename in possible_filenames:
        try:
            # Try reading with default encoding, fallback to latin1 if UTF-8 fails
            try:
                df = pd.read_csv(filename)
            except UnicodeDecodeError:
                df = pd.read_csv(filename, encoding='latin1')
            loaded_filename = filename
            st.success(f"✅ Data loaded successfully from: {loaded_filename}")
            break
        except FileNotFoundError:
            continue
        except Exception as e:
            st.warning(f"⚠️ Could not read file '{filename}'. Error: {e}")
            continue

    if df is None:
        st.error("❌ Critical Error: Could not find or load the financial data CSV file. App functionality will be limited.")
        return pd.DataFrame()

    try:
        df_cleaned = clean_financial_data(df.copy())
        st.info(f"Data cleaned: {df_cleaned.shape[0]} records found.")
        return df_cleaned
    except Exception as e:
        st.error(f"❌ Error during data cleaning: {e}. Returning raw data.")
        return df

def clean_financial_data(df):
    """Clean financial data - Robust percentage/decimal handling"""
    # Standardize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^\w_]', '', regex=True)
    # Rename specific columns if needed (e.g., if standardization creates conflicts)
    # df = df.rename(columns={'old_name': 'new_name'})

    # Ensure essential columns exist
    required_cols = ['company', 'year', 'period_type']
    if not all(col in df.columns for col in required_cols):
        # Attempt to derive if possible
        if 'period' in df.columns:
             if 'year' not in df.columns:
                 df['year'] = df['period'].astype(str).str.extract(r'(\d{4})').astype(float).fillna(method='ffill').astype(int)
             if 'period_type' not in df.columns:
                 df['period_type'] = df['period'].astype(str).apply(lambda x: 'Annual' if 'Annual' in x else ('Quarterly' if 'Q' in x else 'Unknown'))
             if 'quarter' not in df.columns:
                 df['quarter'] = df['period'].astype(str).str.extract(r'Q(\d)').fillna(0).astype(int)
        else:
            missing = [col for col in required_cols if col not in df.columns]
            st.error(f"Cannot proceed. Missing critical columns: {missing}. Please check the CSV.")
            return pd.DataFrame()

    # Define numeric columns (standardized names)
    numeric_columns = [
        'gross_margin', 'net_profit_margin', 'roa', 'roe', 'debt_to_assets',
        'current_ratio', 'debt_to_equity'
    ]
    percentage_columns = ['gross_margin', 'net_profit_margin', 'roa', 'roe', 'debt_to_assets']

    for col in numeric_columns:
        if col in df.columns:
            df[col] = df[col].astype(str)
            df[col] = df[col].str.replace('%', '', regex=False).str.replace(',', '', regex=False).str.strip()
            # Handle potential empty strings after cleaning before numeric conversion
            df[col].replace('', np.nan, inplace=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')

            if col in percentage_columns:
                # Convert if absolute value > 1 (e.g., 15 or -15 for 15% or -15%)
                df[col] = df[col].apply(lambda x: x / 100.0 if pd.notna(x) and abs(x) > 1 else x)
        else:
            st.warning(f"Numeric column '{col}' not found in the data. Adding as empty.")
            df[col] = np.nan

    # Ensure 'year' is integer
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce').fillna(0).astype(int)

    # Ensure 'company' is string and stripped
    if 'company' in df.columns:
        df['company'] = df['company'].astype(str).str.strip()

    # Drop rows where essential identifiers are missing AFTER cleaning attempts
    df.dropna(subset=['company', 'year', 'period_type'], inplace=True)

    # Final check for NaNs in key numeric columns - optional: fill or report
    # nan_counts = df[numeric_columns].isnull().sum()
    # if nan_counts.sum() > 0:
    #     st.warning(f"NaN values found after cleaning in numeric columns: \n{nan_counts[nan_counts > 0]}")

    return df

# Sample data function (optional)
def create_sample_data():
    # ... (keep existing sample data generation logic if needed) ...
    st.warning("Using generated sample data. Results may not be accurate.")
    pass

# ============================================================================
# Enhanced Financial AI Class with Fallback Logic
# ============================================================================

class EnhancedFinancialAI:
    def __init__(self, ai_system_status, financial_data):
        self.ai_status = ai_system_status
        self.df = financial_data
        self.predictor = ai_system_status.get("predictor")
        self.encoder = ai_system_status.get("encoder")
        # Pass the xgboost status to the predictor if it needs it
        if self.predictor:
             self.predictor.xgboost_present = xgboost_present # Add flag if predictor needs it

    def _get_historical_data(self, company, year, period_type):
        """Retrieve historical data for a specific company, year, and period."""
        if self.df.empty:
            return pd.Series(dtype="float64")
        try:
            data = self.df[
                (self.df["company"].str.lower() == company.lower()) &
                (self.df["year"] == year) &
                (self.df["period_type"].str.lower() == period_type.lower())
            ]
            return data.iloc[0] if not data.empty else pd.Series(dtype="float64")
        except Exception as e:
            # st.error(f"Error retrieving historical data: {e}") # Avoid st calls
            return pd.Series(dtype="float64")

    def _mathematical_fallback_prediction(self, company, year, period_type):
        """Basic fallback using historical averages or simple rules."""
        if self.df.empty:
             return pd.Series(dtype="float64")
        # Try last year's annual data
        last_year_data = self._get_historical_data(company, year - 1, "Annual")
        if not last_year_data.empty:
            return last_year_data
        # Fallback: company average (annual data only for stability)
        company_avg = self.df[
            (self.df["company"].str.lower() == company.lower()) &
            (self.df["period_type"].str.lower() == "annual")
        ].select_dtypes(include=np.number).mean()
        return company_avg if not company_avg.empty else pd.Series(dtype="float64")

    def predict_ratios(self, company, year, period_type="Annual"):
        """Predict ratios using AI if available, otherwise use fallback."""
        if self.ai_status["status"] == "AI_MODELS_ACTIVE" and self.predictor:
            # Prepare input data for AI model
            historical_data = self._get_historical_data(company, year, period_type)
            if historical_data.empty:
                 historical_data = self._get_historical_data(company, year - 1, "Annual")

            input_dict = historical_data.to_dict() if not historical_data.empty else {}
            input_dict["Company"] = company
            input_dict["Year"] = year
            input_dict["Period_Type"] = period_type
            input_dict.setdefault("Quarter", 0 if period_type == "Annual" else 1) # Example default

            try:
                predictions, filled_data = self.predictor.predict_all_ratios(input_dict)
                return filled_data
            except Exception as e:
                # st.error(f"Error during AI prediction: {e}. Falling back...") # Avoid st calls
                fallback_data = self._mathematical_fallback_prediction(company, year, period_type)
                return fallback_data.to_dict() if not fallback_data.empty else {}
        else:
            fallback_data = self._mathematical_fallback_prediction(company, year, period_type)
            return fallback_data.to_dict() if not fallback_data.empty else {}

    def perform_health_check(self, company, year, period_type="Annual"):
        """Perform financial health check based on key ratios."""
        data_dict = self.predict_ratios(company, year, period_type)
        if not data_dict:
             return "Could not retrieve or predict data for health check.", {}

        thresholds = {
            "current_ratio": (1.5, 3.0), # Adjusted range
            "debt_to_equity": (0.0, 1.5), # Adjusted range (lower is often better)
            "roa": (0.05, float("inf")), # Minimum acceptable
            "net_profit_margin": (0.1, float("inf"))
        }
        issues = []
        # Use .get(key, default_value) for safety
        cr = data_dict.get("current_ratio", 0)
        de = data_dict.get("debt_to_equity", float("inf"))
        roa = data_dict.get("roa", -float("inf"))
        npm = data_dict.get("net_profit_margin", -float("inf"))

        if not (thresholds["current_ratio"][0] <= cr <= thresholds["current_ratio"][1]):
            issues.append(f"Current Ratio ({cr:.2f}) outside ideal range {thresholds['current_ratio']}")
        if not (thresholds["debt_to_equity"][0] <= de <= thresholds["debt_to_equity"][1]):
            issues.append(f"Debt-to-Equity ({de:.2f}) outside ideal range {thresholds['debt_to_equity']}")
        if roa < thresholds["roa"][0]:
            issues.append(f"ROA ({roa:.2%}) below minimum {thresholds['roa'][0]:.1%}")
        if npm < thresholds["net_profit_margin"][0]:
            issues.append(f"Net Profit Margin ({npm:.2%}) below minimum {thresholds['net_profit_margin'][0]:.1%}")

        if not issues:
            summary = f"✅ Strong Financial Health for {company} ({year} {period_type}). Key ratios within healthy ranges."
        else:
            summary = f"⚠️ Potential Financial Health Concerns for {company} ({year} {period_type}):\n- " + "\n- ".join(issues)

        return summary, data_dict

    def generate_recommendation(self, company, year, period_type="Annual"):
        """Generate investment recommendation based on AI/fallback data."""
        data_dict = self.predict_ratios(company, year, period_type)
        if not data_dict:
            return "Insufficient data", "recommendation-hold", {}

        score = 0
        roe = data_dict.get("roe", 0)
        roa = data_dict.get("roa", 0)
        npm = data_dict.get("net_profit_margin", 0)
        de = data_dict.get("debt_to_equity", 1.0)

        if roe > 0.15: score += 2
        elif roe > 0.10: score += 1
        if roa > 0.07: score += 2
        elif roa > 0.04: score += 1
        if npm > 0.12: score += 1
        if de < 1.0: score += 1
        elif de > 2.0: score -= 1

        if score >= 5: rec, style = "Strong Buy", "recommendation-buy"
        elif score >= 3: rec, style = "Buy", "recommendation-buy"
        elif score >= 1: rec, style = "Hold", "recommendation-hold"
        else: rec, style = "Sell", "recommendation-sell"

        rec_source = "(AI Assisted)" if self.ai_status["status"] == "AI_MODELS_ACTIVE" else "(Fallback Based)"
        return f"{rec} {rec_source}", style, data_dict

# ============================================================================
# Load Data and AI System
# ============================================================================

ai_system_info = load_comprehensive_ai_system()
financial_data_df = load_financial_data()

# Initialize the AI/Fallback Handler
if not financial_data_df.empty:
    financial_ai = EnhancedFinancialAI(ai_system_info, financial_data_df)
    # Use standardized 'company' column for unique list
    companies = sorted(financial_data_df['company'].unique())
    years = sorted(financial_data_df['year'].unique(), reverse=True)
    periods = ['Annual', 'Quarterly']
else:
    st.error("Application cannot function without financial data. Please ensure the CSV file is accessible and readable.")
    companies, years, periods = [], [], []
    financial_ai = None

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

def format_value(value, ratio_name):
    """Helper function to format numeric values consistently."""
    # Use standardized names
    percentage_cols_std = ['gross_margin', 'net_profit_margin', 'roa', 'roe', 'debt_to_assets']
    if pd.isna(value):
        return "N/A"
    elif ratio_name in percentage_cols_std:
        return f"{value:.2%}"
    else: # Ratios like current_ratio, debt_to_equity
        return f"{value:.2f}"

if financial_ai is None:
    st.warning("Financial data could not be loaded or is empty. Please check the CSV file.")

# --- Dashboard ---
elif analysis_type == "Dashboard":
    st.markdown("<h2 class='sub-header'>📊 Financial AI Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("Overview of Saudi Food Sector Companies (Almarai, Savola, NADEC)")

    if not financial_data_df.empty and companies:
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
                    # Use standardized 'company' column for filtering
                    company_data = latest_annual_data[latest_annual_data['company'].str.lower() == company.lower()]
                    if not company_data.empty:
                        # Use standardized column names
                        roe = company_data['roe'].iloc[0]
                        roa = company_data['roa'].iloc[0]
                        npm = company_data['net_profit_margin'].iloc[0]
                        st.metric("ROE", format_value(roe, 'roe'))
                        st.metric("ROA", format_value(roa, 'roa'))
                        st.metric("Net Profit Margin", format_value(npm, 'net_profit_margin'))

                        rec, rec_style, _ = financial_ai.generate_recommendation(company, latest_year, 'Annual')
                        st.markdown(f'<div class="{rec_style}">{rec}</div>', unsafe_allow_html=True)
                    else:
                        st.warning(f"No {latest_year} data for {company}.")

            # Sector Trends Plot
            st.markdown("<h3 class='sub-header' style='margin-top: 2rem;'>📈 Sector ROE Trends (Annual)</h3>", unsafe_allow_html=True)
            annual_data = financial_data_df[financial_data_df['period_type'].str.lower() == 'annual']
            if not annual_data.empty:
                fig_trends = px.line(annual_data, x='year', y='roe', color='company',
                                     title="Return on Equity (ROE) Over Time",
                                     labels={'year': 'Year', 'roe': 'ROE', 'company': 'Company'},
                                     markers=True)
                fig_trends.update_layout(yaxis_tickformat='.1%')
                st.plotly_chart(fig_trends, use_container_width=True)
            else:
                 st.warning("No annual data available for trend plot.")
        else:
            st.warning(f"No annual data found for the latest year ({latest_year}).")
    else:
        st.warning("No financial data available for dashboard.")

# --- Individual Company Analysis ---
elif analysis_type == "Individual Company Analysis":
    st.markdown("<h2 class='sub-header'>🏢 Individual Company Analysis</h2>", unsafe_allow_html=True)
    st.markdown("Deep dive into specific company performance.")

    if companies and years and periods:
        sel_company = st.selectbox("Select Company:", companies)
        sel_year = st.selectbox("Select Year:", years)
        sel_period = st.selectbox("Select Period:", periods)

        if st.button("📊 Generate Analysis"):
            st.subheader(f"Analysis for {sel_company} - {sel_year} {sel_period}")
            data_dict = financial_ai.predict_ratios(sel_company, sel_year, sel_period)

            if data_dict:
                profitability_ratios = ['gross_margin', 'net_profit_margin', 'roa', 'roe']
                health_ratios = ['current_ratio', 'debt_to_equity', 'debt_to_assets']

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("**Profitability Ratios**")
                    for ratio in profitability_ratios:
                        value = data_dict.get(ratio)
                        st.metric(label=ratio.replace('_', ' ').title(), value=format_value(value, ratio))
                with col2:
                    st.markdown("**Financial Health Ratios**")
                    for ratio in health_ratios:
                        value = data_dict.get(ratio)
                        st.metric(label=ratio.replace('_', ' ').title(), value=format_value(value, ratio))
            else:
                st.warning(f"Could not retrieve or predict data for {sel_company}, {sel_year}, {sel_period}.")
    else:
        st.warning("No data loaded. Cannot perform individual analysis.")

# --- Company Comparison ---
elif analysis_type == "Company Comparison":
    st.markdown("<h2 class='sub-header'>🆚 Company Comparison</h2>", unsafe_allow_html=True)
    st.markdown("Side-by-side financial performance comparison.")

    if companies and years and periods:
        # Get available numeric columns AFTER cleaning
        numeric_cols_available = financial_data_df.select_dtypes(include=np.number).columns.drop(['year', 'quarter'], errors='ignore')
        if not numeric_cols_available.empty:
            comp_year = st.selectbox("Comparison Year:", years, key="comp_year")
            comp_period = st.selectbox("Comparison Period:", periods, key="comp_period")
            # Default to 'roe' if available, otherwise first numeric column
            default_ratio_index = list(numeric_cols_available).index('roe') if 'roe' in numeric_cols_available else 0
            comp_ratio = st.selectbox("Ratio to Compare:", numeric_cols_available, index=default_ratio_index, key="comp_ratio")

            if st.button("🔍 Compare Companies"):
                st.subheader(f"Comparison for {comp_year} {comp_period} - Ratio: {comp_ratio.replace('_', ' ').title()}")
                comparison_data = []
                for company in companies:
                    data_dict = financial_ai.predict_ratios(company, comp_year, comp_period)
                    ratio_value = data_dict.get(comp_ratio, np.nan)
                    comparison_data.append({'Company': company, comp_ratio: ratio_value})

                comparison_df = pd.DataFrame(comparison_data).set_index('Company')
                comparison_df.dropna(inplace=True)

                if not comparison_df.empty:
                    st.dataframe(comparison_df.style.format({comp_ratio: lambda x: format_value(x, comp_ratio)}))

                    st.markdown(f"<h3 class='sub-header' style='margin-top: 1rem;'>📊 Visual Comparison - {comp_ratio.replace('_', ' ').title()}</h3>", unsafe_allow_html=True)
                    fig_comp = px.bar(comparison_df.reset_index(), x='Company', y=comp_ratio,
                                      title=f"{comp_ratio.replace('_', ' ').title()} Comparison - {comp_year} {comp_period}",
                                      labels={comp_ratio: comp_ratio.replace('_', ' ').title(), 'Company': 'Company'},
                                      text_auto=True)
                    # Use format_value logic for axis ticks if possible (simplified here)
                    is_percentage = comp_ratio in ['gross_margin', 'net_profit_margin', 'roa', 'roe', 'debt_to_assets']
                    fig_comp.update_layout(yaxis_tickformat='.1%' if is_percentage else '.2f')
                    fig_comp.update_traces(texttemplate='%{y:.2%}' if is_percentage else '%{y:.2f}', textposition='outside')
                    st.plotly_chart(fig_comp, use_container_width=True)
                else:
                    st.warning(f"No comparable data found for {comp_ratio} in {comp_year} {comp_period}.")
        else:
             st.warning("No numeric ratios available in the data for comparison.")
    else:
        st.warning("No data loaded. Cannot perform company comparison.")

# --- Financial Health Check ---
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
                st.markdown(summary)
                st.markdown("**Data Used for Check:**")
                display_ratios = ['current_ratio', 'debt_to_equity', 'roa', 'net_profit_margin', 'roe', 'debt_to_assets']
                display_dict_formatted = {k.replace('_',' ').title(): format_value(data_dict.get(k), k) for k in display_ratios if k in data_dict}
                st.json(display_dict_formatted)
            else:
                 st.error(f"Could not perform health check for {hc_company}, {hc_year}, {hc_period}. Data unavailable.")
    else:
        st.warning("No data loaded. Cannot perform health check.")

# --- Ratio Prediction ---
elif analysis_type == "Ratio Prediction":
    st.markdown("<h2 class='sub-header'>🔮 Financial Ratio Prediction</h2>", unsafe_allow_html=True)
    st.markdown("Predict future financial performance using AI and historical trends.")

    if companies and years:
        pred_company = st.selectbox("Company for Prediction:", companies, key="pred_company")
        current_max_year = max(years) if years else datetime.now().year
        pred_year = st.number_input("Predict for Year:", min_value=current_max_year + 1, value=current_max_year + 1, step=1, key="pred_year")
        pred_method = "AI Model (Advanced)" if ai_system_info["status"] == "AI_MODELS_ACTIVE" else "Mathematical Fallback"
        st.info(f"Prediction Method: {pred_method}")

        st.markdown(f"**Recent Historical Data: {pred_company} (Annual)**")
        hist_data = financial_data_df[
            (financial_data_df['company'].str.lower() == pred_company.lower()) &
            (financial_data_df['period_type'].str.lower() == 'annual')
        ].sort_values('year', ascending=False).head(3)

        if not hist_data.empty:
            cols = st.columns(len(hist_data))
            for i, row in hist_data.iterrows():
                with cols[i]:
                    st.metric("Year", str(row['year']))
                    st.metric("ROE", format_value(row.get('roe'), 'roe'))
                    st.metric("ROA", format_value(row.get('roa'), 'roa'))
        else:
             st.warning("No recent annual data found for context.")

        if st.button("🚀 Generate Predictions"):
            st.subheader(f"Predicted Ratios for {pred_company} - {pred_year} (Annual)")
            predicted_data_dict = financial_ai.predict_ratios(pred_company, pred_year, "Annual")

            if predicted_data_dict:
                pred_ratios_disp = ['roe', 'roa', 'net_profit_margin', 'current_ratio', 'debt_to_equity']
                # Determine number of columns based on available predictions
                valid_preds = {k: predicted_data_dict.get(k) for k in pred_ratios_disp if pd.notna(predicted_data_dict.get(k))}
                if valid_preds:
                    cols_pred = st.columns(len(valid_preds))
                    i = 0
                    for ratio, value in valid_preds.items():
                        with cols_pred[i]:
                            st.metric(label=ratio.replace('_', ' ').title(), value=format_value(value, ratio))
                            i += 1
                else:
                    st.warning("Prediction resulted in no valid ratio values.")
                # Optionally show raw prediction details
                # st.write("Full Predicted Data:", predicted_data_dict)
            else:
                st.error(f"Could not generate predictions for {pred_company}, {pred_year}.")
    else:
        st.warning("No data loaded. Cannot perform ratio prediction.")

# --- Custom Financial Analysis ---
elif analysis_type == "Custom Financial Analysis":
    st.markdown("<h2 class='sub-header'>⚙️ Custom Financial Analysis</h2>", unsafe_allow_html=True)
    st.markdown("Input your own financial ratios for AI analysis (or fallback calculation).")

    if companies:
        st.markdown("**Enter Financial Data**")
        col1, col2 = st.columns(2)

        with col1:
            cust_company = st.text_input("Company Name (e.g., Almarai, Savola, NADEC, or Other):", value="Almarai", key="cust_comp")
            cust_year = st.number_input("Year:", min_value=2010, max_value=datetime.now().year + 5, value=datetime.now().year, key="cust_year")
            cust_period = st.selectbox("Period:", ["Annual", "Quarterly"], key="cust_period")

            st.markdown("**Profitability Ratios (%)**")
            cust_gross_margin = st.slider("Gross Margin (%):", 0, 100, 38, key="cust_gm")
            cust_npm = st.slider("Net Profit Margin (%):", -50, 100, 10, key="cust_npm")
            cust_roa = st.slider("Return on Assets (ROA %):", -50, 50, 8, key="cust_roa")

        with col2:
            st.markdown("**Financial Health Ratios**")
            cust_current_ratio = st.slider("Current Ratio:", 0.0, 5.0, 1.5, step=0.01, key="cust_cr")
            cust_debt_equity = st.slider("Debt-to-Equity:", 0.0, 5.0, 0.8, step=0.01, key="cust_de")
            cust_debt_assets = st.slider("Debt-to-Assets (%):", 0, 100, 45, key="cust_da")

            st.markdown("**Optional**")
            cust_roe_input_type = st.radio("Provide ROE?", ["Manual ROE (%)", "Use AI/Fallback to Predict ROE"], index=1, key="cust_roe_type")
            cust_roe = None
            if cust_roe_input_type == "Manual ROE (%)":
                cust_roe = st.slider("Manual ROE (%):", -50, 100, 12, key="cust_roe_manual")

        if st.button("📈 Analyze Custom Data"):
            st.subheader(f"Custom Analysis Results for {cust_company} - {cust_year} {cust_period}")

            custom_input = {
                "Company": cust_company,
                "Year": cust_year,
                "Period_Type": cust_period,
                # Pass slider values directly (as percentages 0-100 or ratios)
                "Gross_Margin": cust_gross_margin,
                "Net_Profit_Margin": cust_npm,
                "ROA": cust_roa,
                "Current_Ratio": cust_current_ratio,
                "Debt_to_Equity": cust_debt_equity,
                "Debt_to_Assets": cust_debt_assets,
                "ROE": cust_roe if cust_roe is not None else np.nan
            }

            # Use the predictor directly for custom analysis
            if ai_system_info["status"] == "AI_MODELS_ACTIVE" and financial_ai.predictor:
                st.info("Using AI Model for analysis...")
                try:
                    # The predictor handles conversion from % to decimal internally now
                    predictions, analyzed_data = financial_ai.predictor.predict_all_ratios(custom_input)

                    st.markdown("**Analyzed / Predicted Ratios:**")
                    disp_ratios = ['roe', 'roa', 'net_profit_margin', 'gross_margin', 'current_ratio', 'debt_to_equity', 'debt_to_assets']
                    results_disp = {}
                    for r in disp_ratios:
                        val = analyzed_data.get(r)
                        is_predicted = r in predictions
                        conf = predictions[r]['confidence'] if is_predicted else 'N/A'
                        val_str = format_value(val, r)
                        source = f"(Predicted - {conf})" if is_predicted else "(Input)"
                        # Handle case where input was NaN and prediction failed
                        if pd.isna(val) and r not in custom_input: source = "(Not Available)"
                        elif pd.isna(val) and pd.isna(custom_input.get(r)): source = "(Input: N/A)"

                        results_disp[r.replace('_',' ').title()] = f"{val_str} {source}"
                    st.json(results_disp)

                    # Recommendation based on analyzed data (might be less accurate without history)
                    # We need a way to generate recommendation purely from the final dict
                    # Let's adapt generate_recommendation or create a simpler version
                    # For now, just show the data
                    # rec, rec_style, _ = financial_ai.generate_recommendation_from_dict(analyzed_data)
                    # st.markdown(f"**Recommendation:** <span class='{rec_style}'>{rec}</span>", unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error during custom AI analysis: {e}")
                    st.warning("Displaying input data only.")
                    st.json({k: v for k, v in custom_input.items() if k not in ['Company', 'Year', 'Period_Type']})
            else:
                st.warning("AI Model not active. Displaying input data only. Fallback calculations for custom input are limited.")
                st.json({k: v for k, v in custom_input.items() if k not in ['Company', 'Year', 'Period_Type']})
    else:
        st.warning("No base company data loaded. Custom analysis section requires initial data load.")

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
    else:
        # Display the specific error message from loading
        st.write(f"- Details: {ai_system_info['message']}")
        if not xgboost_present:
             st.write(f"- Action: Install XGBoost for full AI features.")
with col3:
    st.markdown("**Capabilities**")
    st.write("- Ratio Prediction (AI/Fallback)")
    st.write("- Investment Recommendations")
    st.write("- Company Comparison")
    st.write("- Financial Health Assessment")
    st.write("- Custom Analysis (AI/Input)")

with st.expander("Technical Details"):
    st.write("This application uses machine learning models (including potentially XGBoost, Scikit-learn) trained on historical financial data. Ensure all required libraries from `requirements.txt` are installed.")
    st.write("Fallback mechanisms use historical averages or simple rules when AI models are unavailable or fail.")
    st.write("Data Cleaning standardizes names, handles mixed percentage/decimal formats (assuming inputs > 1 are percentages), and manages data types.")
    st.write("Prediction confidence is estimated based on input feature availability.")

st.markdown("<hr style='margin-top: 2rem;'>", unsafe_allow_html=True)
st.caption("Saudi Food Sector Financial AI Assistant | Powered by Advanced Machine Learning")

