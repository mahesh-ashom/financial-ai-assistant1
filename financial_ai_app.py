import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Financial AI Assistant", 
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CRITICAL: Define ComprehensiveRatioPredictor class for pickle loading
# ============================================================================

class ComprehensiveRatioPredictor:
    """Predicts all financial ratios using trained models"""

    def __init__(self, models_dict, company_encoder=None):
        self.models = models_dict
        self.le_company = company_encoder
        self.available_ratios = list(models_dict.keys()) if models_dict else []

    def predict_all_ratios(self, input_data, prediction_method='iterative'):
        """Predict all financial ratios from partial input"""
        results = {}
        input_copy = input_data.copy()

        # Handle company encoding if needed
        if self.le_company and 'Company' in input_copy:
            try:
                if hasattr(self.le_company, 'classes_') and input_copy['Company'] in self.le_company.classes_:
                    input_copy['Company_Encoded'] = self.le_company.transform([input_copy['Company']])[0]
                else:
                    input_copy['Company_Encoded'] = 0
            except:
                input_copy['Company_Encoded'] = 0

        # Clean input values
        for key, value in input_copy.items():
            if isinstance(value, str) and key in self.available_ratios:
                try:
                    cleaned_value = str(value).replace('%', '').replace(',', '').strip()
                    input_copy[key] = float(cleaned_value)
                    
                    # Convert percentages to decimals if needed
                    if key in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 'Debt-to-Assets'] and input_copy[key] > 1:
                        input_copy[key] = input_copy[key] / 100
                except:
                    input_copy[key] = np.nan

        # Iterative prediction approach
        if prediction_method == 'iterative' and self.models:
            for iteration in range(3):
                for ratio in self.available_ratios:
                    if ratio not in input_copy or pd.isna(input_copy.get(ratio)):
                        if ratio in self.models:
                            model_info = self.models[ratio]
                            if isinstance(model_info, dict) and 'model' in model_info:
                                model = model_info['model']
                                features = model_info.get('features', [])

                                # Check available features
                                available_features = [f for f in features
                                                    if f in input_copy and not pd.isna(input_copy.get(f))]

                                if len(available_features) >= len(features) * 0.5:
                                    try:
                                        # Create feature vector
                                        feature_values = []
                                        for feature in features:
                                            if feature in input_copy and not pd.isna(input_copy.get(feature)):
                                                feature_values.append(input_copy[feature])
                                            else:
                                                feature_values.append(0)

                                        # Make prediction
                                        if hasattr(model, 'predict'):
                                            prediction = model.predict([feature_values])[0]
                                            input_copy[ratio] = prediction

                                            confidence = 'High' if len(available_features) >= len(features) * 0.8 else 'Medium'
                                            results[ratio] = {
                                                'predicted_value': prediction,
                                                'confidence': confidence,
                                                'iteration': iteration + 1
                                            }
                                    except Exception as e:
                                        pass

        return results, input_copy

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .recommendation-buy {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .recommendation-sell {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .recommendation-hold {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }
    .stProgress .st-bo {
        background-color: #e9ecef;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 Financial AI Assistant</h1>', unsafe_allow_html=True)
st.markdown("**Saudi Food Sector Investment Analysis System**")
st.markdown("*Analyze Almarai, Savola, and NADEC with AI-powered insights*")

# ============================================================================
# Enhanced AI System Loading with Multiple Fallbacks
# ============================================================================

@st.cache_resource
def load_comprehensive_ai_system():
    """Load the comprehensive AI system with enhanced error handling"""
    try:
        # Try to load the main comprehensive predictor
        comprehensive_predictor = joblib.load('comprehensive_ratio_predictor.pkl')
        
        # Try to load company encoder
        try:
            company_encoder = joblib.load('company_encoder.pkl')
        except:
            company_encoder = None
        
        # Check if the system has models
        if hasattr(comprehensive_predictor, 'models') and comprehensive_predictor.models:
            available_models = list(comprehensive_predictor.models.keys())
            
            return {
                'status': 'AI_MODELS_ACTIVE',
                'comprehensive_predictor': comprehensive_predictor,
                'company_encoder': company_encoder,
                'available_models': available_models,
                'model_count': len(available_models)
            }
        else:
            return {'status': 'FALLBACK_MODE', 'error': 'No models found in predictor'}
            
    except Exception as e:
        # Try to load individual model files as backup
        try:
            individual_models = {}
            model_files = [
                ('roe_prediction_model.pkl', 'ROE'),
                ('investment_model.pkl', 'Investment'),
                ('company_status_model.pkl', 'Status'),
                ('model_roa.pkl', 'ROA'),
                ('model_net_profit_margin.pkl', 'Net Profit Margin'),
                ('model_gross_margin.pkl', 'Gross Margin')
            ]
            
            for model_file, ratio_name in model_files:
                try:
                    model = joblib.load(model_file)
                    individual_models[ratio_name] = model
                except:
                    continue
            
            if individual_models:
                predictor = ComprehensiveRatioPredictor(individual_models, None)
                
                return {
                    'status': 'PARTIAL_AI_ACTIVE',
                    'comprehensive_predictor': predictor,
                    'company_encoder': None,
                    'available_models': list(individual_models.keys()),
                    'model_count': len(individual_models)
                }
            
        except:
            pass
        
        return {
            'status': 'FALLBACK_MODE', 
            'error': f'Model loading failed: {str(e)[:100]}'
        }

# Load financial data with multiple fallback options
@st.cache_data
def load_financial_data():
    """Load and prepare financial data with comprehensive cleaning"""
    try:
        # Try multiple possible filenames
        possible_filenames = [
            'Savola Almarai NADEC Financial Ratios CSV.csv.csv',
            'Savola Almarai NADEC Financial Ratios CSV.csv', 
            'Savola_Almarai_NADEC_Financial_Ratios_CSV.csv',
            'financial_data.csv',
            'data.csv'
        ]
        
        df = None
        loaded_filename = None
        
        for filename in possible_filenames:
            try:
                df = pd.read_csv(filename)
                loaded_filename = filename
                break
            except:
                continue
        
        if df is None:
            st.warning("⚠️ CSV file not found. Using sample data for demonstration.")
            return create_sample_data()
        
        st.success(f"✅ Data loaded from: {loaded_filename}")
        
        # Clean the data
        df = clean_financial_data(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

def clean_financial_data(df):
    """Clean financial data - handle percentages correctly"""
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Define columns that should be treated as percentages vs ratios
    percentage_columns = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 'Debt-to-Assets']
    ratio_columns = ['Current Ratio', 'Debt-to-Equity']
    
    # Clean percentage columns
    for col in percentage_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Convert to decimal if values appear to be in percentage format (>1)
            if df[col].max() > 1:
                df[col] = df[col] / 100
    
    # Clean ratio columns
    for col in ratio_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(',', '').str.strip()
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Create derived columns if missing
    if 'Period_Type' not in df.columns and 'Period' in df.columns:
        df['Period_Type'] = df['Period'].apply(
            lambda x: 'Annual' if 'Annual' in str(x) or (isinstance(x, (int, float)) and x == int(x)) else 'Quarterly'
        )
    
    if 'Year' not in df.columns and 'Period' in df.columns:
        df['Year'] = df['Period'].str.extract(r'(\d{4})').astype(float).fillna(2023).astype(int)
    
    if 'Quarter' not in df.columns and 'Period' in df.columns:
        df['Quarter'] = df['Period'].str.extract(r'Q(\d)').fillna(0).astype(int)
    
    # Fill missing values with median for each company
    for company in df['Company'].unique():
        company_mask = df['Company'] == company
        for col in percentage_columns + ratio_columns:
            if col in df.columns:
                company_median = df.loc[company_mask, col].median()
                if not pd.isna(company_median):
                    df.loc[company_mask, col] = df.loc[company_mask, col].fillna(company_median)
    
    return df

def create_sample_data():
    """Create sample data for demonstration purposes"""
    companies = ['Almarai', 'Savola', 'NADEC']
    years = [2020, 2021, 2022, 2023]
    quarters = [1, 2, 3, 4]
    
    data = []
    
    # Base financial ratios for each company (in decimal format)
    base_ratios = {
        'Almarai': {
            'Gross Margin': 0.309, 'Net Profit Margin': 0.105, 'ROA': 0.057, 'ROE': 0.115,
            'Current Ratio': 1.40, 'Debt-to-Equity': 1.03, 'Debt-to-Assets': 0.51
        },
        'Savola': {
            'Gross Margin': 0.203, 'Net Profit Margin': 0.045, 'ROA': 0.040, 'ROE': 0.126,
            'Current Ratio': 0.84, 'Debt-to-Equity': 2.14, 'Debt-to-Assets': 0.68
        },
        'NADEC': {
            'Gross Margin': 0.370, 'Net Profit Margin': 0.094, 'ROA': 0.059, 'ROE': 0.084,
            'Current Ratio': 1.96, 'Debt-to-Equity': 0.42, 'Debt-to-Assets': 0.30
        }
    }
    
    for company in companies:
        for year in years:
            # Annual data
            annual_ratios = base_ratios[company].copy()
            # Add some year-over-year variation
            for ratio in annual_ratios:
                annual_ratios[ratio] *= (1 + np.random.normal(0, 0.1))
                # Ensure positive values
                annual_ratios[ratio] = max(0, annual_ratios[ratio])
            
            data.append({
                'Company': company,
                'Period': f'{year} Annual',
                'Period_Type': 'Annual',
                'Year': year,
                'Quarter': 0,
                **annual_ratios
            })
            
            # Quarterly data
            for quarter in quarters:
                quarterly_ratios = base_ratios[company].copy()
                # Add quarterly variation
                for ratio in quarterly_ratios:
                    quarterly_ratios[ratio] *= (1 + np.random.normal(0, 0.15))
                    # Ensure positive values
                    quarterly_ratios[ratio] = max(0, quarterly_ratios[ratio])
                
                data.append({
                    'Company': company,
                    'Period': f'{year} Q{quarter}',
                    'Period_Type': 'Quarterly',
                    'Year': year,
                    'Quarter': quarter,
                    **quarterly_ratios
                })
    
    return pd.DataFrame(data)

# ============================================================================
# Enhanced Financial AI Class
# ============================================================================

class EnhancedFinancialAI:
    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.status = ai_system['status']
        
        if self.status in ['AI_MODELS_ACTIVE', 'PARTIAL_AI_ACTIVE']:
            self.comprehensive_predictor = ai_system['comprehensive_predictor']
            self.company_encoder = ai_system['company_encoder']
        else:
            self.comprehensive_predictor = None
            self.company_encoder = None
    
    def comprehensive_analysis(self, company_data):
        """Perform comprehensive analysis using AI or fallbacks"""
        
        if self.status in ['AI_MODELS_ACTIVE', 'PARTIAL_AI_ACTIVE'] and self.comprehensive_predictor:
            try:
                # Use the comprehensive AI system
                predictions, complete_data = self.comprehensive_predictor.predict_all_ratios(
                    company_data, 
                    prediction_method='iterative'
                )
                
                # Extract predicted ROE
                predicted_roe = predictions.get('ROE', {}).get('predicted_value', 
                                                             self._estimate_roe(company_data))
                
                # Create investment recommendation based on AI predictions
                investment_score = self._calculate_ai_investment_score(complete_data, predictions)
                
                if investment_score >= 70:
                    investment_rec, confidence = "Strong Buy", 0.85
                elif investment_score >= 60:
                    investment_rec, confidence = "Buy", 0.75
                elif investment_score >= 40:
                    investment_rec, confidence = "Hold", 0.65
                else:
                    investment_rec, confidence = "Sell", 0.55
                
                # Determine company status
                company_status = self._get_ai_company_status(complete_data)
                
                return {
                    'predicted_roe': predicted_roe,
                    'investment_recommendation': investment_rec,
                    'investment_confidence': confidence,
                    'company_status': company_status,
                    'investment_score': investment_score,
                    'ai_predictions': predictions,
                    'prediction_method': 'AI_COMPREHENSIVE_SYSTEM'
                }
                
            except Exception as e:
                st.warning(f"AI prediction failed: {e}. Using fallback method.")
                return self._fallback_analysis(company_data)
        else:
            return self._fallback_analysis(company_data)
    
    def _estimate_roe(self, data):
        """Estimate ROE using DuPont formula if not available"""
        roa = data.get('ROA', 0.05)
        debt_equity = data.get('Debt-to-Equity', 0.8)
        equity_multiplier = 1 + debt_equity
        return roa * equity_multiplier
    
    def _calculate_ai_investment_score(self, complete_data, predictions):
        """Calculate investment score using AI predictions"""
        score = 0
        
        # Use predicted ROE
        roe = predictions.get('ROE', {}).get('predicted_value', complete_data.get('ROE', 0))
        if roe > 0.15: score += 30
        elif roe > 0.10: score += 20
        elif roe > 0.05: score += 10
        
        # Use predicted ROA
        roa = predictions.get('ROA', {}).get('predicted_value', complete_data.get('ROA', 0))
        if roa > 0.08: score += 25
        elif roa > 0.05: score += 15
        elif roa > 0.02: score += 5
        
        # Use predicted Net Profit Margin
        npm = predictions.get('Net Profit Margin', {}).get('predicted_value', 
                                                           complete_data.get('Net Profit Margin', 0))
        if npm > 0.10: score += 20
        elif npm > 0.05: score += 10
        
        # Debt ratios
        debt_equity = predictions.get('Debt-to-Equity', {}).get('predicted_value', 
                                                               complete_data.get('Debt-to-Equity', 1))
        if debt_equity < 0.5: score += 15
        elif debt_equity < 1.0: score += 10
        elif debt_equity < 1.5: score += 5
        
        return min(100, score)
    
    def _get_ai_company_status(self, complete_data):
        """Determine company status using AI predictions"""
        roe = complete_data.get('ROE', 0)
        npm = complete_data.get('Net Profit Margin', 0)
        
        if roe > 0.15 and npm > 0.15:
            return 'Excellent'
        elif roe > 0.10 and npm > 0.10:
            return 'Good'
        elif roe > 0.05 and npm > 0.05:
            return 'Average'
        else:
            return 'Poor'
    
    def _fallback_analysis(self, company_data):
        """Fallback to mathematical calculations"""
        predicted_roe = self._estimate_roe(company_data)
        investment_score = self._calculate_fallback_score(company_data)
        
        if investment_score >= 70:
            investment_rec, confidence = "Buy", 0.70
        elif investment_score >= 50:
            investment_rec, confidence = "Hold", 0.65
        else:
            investment_rec, confidence = "Sell", 0.60
        
        return {
            'predicted_roe': predicted_roe,
            'investment_recommendation': investment_rec,
            'investment_confidence': confidence,
            'company_status': 'Average',
            'investment_score': investment_score,
            'prediction_method': 'MATHEMATICAL_FALLBACK'
        }
    
    def _calculate_fallback_score(self, data):
        """Calculate fallback investment score"""
        score = 0
        roe = data.get('ROE', self._estimate_roe(data))
        
        if roe > 0.15: score += 35
        elif roe > 0.10: score += 25
        elif roe > 0.05: score += 15
        
        roa = data.get('ROA', 0.05)
        if roa > 0.08: score += 25
        elif roa > 0.05: score += 15
        
        npm = data.get('Net Profit Margin', 0.08)
        if npm > 0.10: score += 20
        elif npm > 0.05: score += 10
        
        return min(100, score)

# ============================================================================
# Load Data and Initialize AI System
# ============================================================================

# Load data and initialize AI system
df = load_financial_data()
ai_system = load_comprehensive_ai_system()
enhanced_financial_ai = EnhancedFinancialAI(ai_system)

# ============================================================================
# Sidebar Navigation and AI Status
# ============================================================================

# Sidebar for navigation and AI status
st.sidebar.title("🎯 Navigation")

# AI System Status Display
st.sidebar.subheader("🤖 AI System Status")

if ai_system['status'] == 'AI_MODELS_ACTIVE':
    st.sidebar.success("🤖 **AI MODELS ACTIVE**")
    st.sidebar.write(f"✅ Comprehensive AI System Loaded")
elif ai_system['status'] == 'PARTIAL_AI_ACTIVE':
    st.sidebar.info("🤖 **PARTIAL AI ACTIVE**")
    st.sidebar.write(f"✅ Some AI Models Loaded")
else:
    st.sidebar.warning("⚠️ **Using Mathematical Fallbacks**")
    st.sidebar.write(f"Error: {ai_system.get('error', 'Unknown error')}")

st.sidebar.write(f"✅ Available Models: {ai_system['model_count']}")

# Show available AI capabilities
with st.sidebar.expander("🔍 System Details", expanded=False):
    if ai_system['status'] != 'FALLBACK_MODE':
        st.write("**Available Models:**")
        for model in ai_system['available_models']:
            st.write(f"• {model}")
    
    st.write("**Capabilities:**")
    st.write("• ROE Prediction")
    st.write("• Investment Recommendations")
    st.write("• Financial Health Assessment")
    st.write("• Risk Analysis")

# Main navigation
page = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["🏠 Dashboard", "📊 Company Analysis", "🔮 Quick Prediction", "🏥 Health Check", "⚖️ Comparison", "🎯 Custom Analysis"]
)

# ============================================================================
# Dashboard Page
# ============================================================================

if page == "🏠 Dashboard":
    st.header("📊 Financial AI Dashboard")
    st.markdown("*Overview of Saudi Food Sector Performance*")
    
    if not df.empty:
        # Key metrics overview
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_companies = df['Company'].nunique()
            st.metric("Companies Analyzed", total_companies)
        
        with col2:
            date_range = f"{df['Year'].min()}-{df['Year'].max()}"
            st.metric("Data Period", date_range)
        
        with col3:
            total_records = len(df)
            st.metric("Financial Records", total_records)
        
        with col4:
            avg_roe = df['ROE'].mean()
            st.metric("Avg Sector ROE", f"{avg_roe:.1%}")
        
        # Latest performance summary
        st.subheader("🏆 Latest Company Performance")
        
        # Get latest data for each company
        latest_year = df['Year'].max()
        latest_data = df[(df['Year'] == latest_year) & (df['Period_Type'] == 'Annual')]
        
        if not latest_data.empty:
            # Create performance comparison
            performance_cols = st.columns(len(latest_data))
            
            for i, (_, company_data) in enumerate(latest_data.iterrows()):
                with performance_cols[i]:
                    company = company_data['Company']
                    roe = company_data['ROE']
                    
                    # Generate recommendation using enhanced AI
                    recommendation_result = enhanced_financial_ai.comprehensive_analysis(company_data.to_dict())
                    recommendation = recommendation_result['investment_recommendation']
                    
                    st.markdown("---")
                st.subheader("🎯 AI Investment Analysis")
                
                # Display results
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("AI Predicted ROE", f"{results['predicted_roe']:.1%}")
                
                with col_b:
                    rec = results['investment_recommendation']
                    if rec in ["Strong Buy", "Buy"]:
                        st.success(f"📈 {rec}")
                    elif "Hold" in rec:
                        st.warning(f"⚖️ {rec}")
                    else:
                        st.error(f"📉 {rec}")
                
                with col_c:
                    confidence = results['investment_confidence']
                    st.metric("AI Confidence", f"{confidence:.0%}")
                
                # Investment score and status
                score = results['investment_score']
                status = results['company_status']
                
                col_d, col_e = st.columns(2)
                
                with col_d:
                    st.metric("Investment Score", f"{score}/100")
                    st.progress(score / 100)
                
                with col_e:
                    if status == "Excellent":
                        st.success(f"🌟 Company Status: {status}")
                    elif status == "Good":
                        st.info(f"👍 Company Status: {status}")
                    elif status == "Average":
                        st.warning(f"📊 Company Status: {status}")
                    else:
                        st.error(f"⚠️ Company Status: {status}")

# ============================================================================
# Quick Prediction Page
# ============================================================================

elif page == "🔮 Quick Prediction":
    st.header("🔮 Quick Financial Prediction")
    st.markdown("*Get instant AI predictions with minimal input*")
    
    st.subheader("📝 Quick Input Form")
    
    quick_col1, quick_col2 = st.columns(2)
    
    with quick_col1:
        quick_company = st.selectbox("Company:", ["Almarai", "Savola", "NADEC", "Other"], key="quick_company")
        quick_roe = st.number_input("Current ROE (%)", min_value=0.0, max_value=50.0, value=10.0, step=0.1) / 100
        quick_roa = st.number_input("Current ROA (%)", min_value=0.0, max_value=30.0, value=5.0, step=0.1) / 100
    
    with quick_col2:
        quick_npm = st.number_input("Net Profit Margin (%)", min_value=0.0, max_value=30.0, value=8.0, step=0.1) / 100
        quick_debt_equity = st.number_input("Debt-to-Equity Ratio", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
        quick_current_ratio = st.number_input("Current Ratio", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
    
    if st.button("⚡ Get Quick Prediction", type="primary"):
        quick_data = {
            'Company': quick_company,
            'ROE': quick_roe,
            'ROA': quick_roa,
            'Net Profit Margin': quick_npm,
            'Debt-to-Equity': quick_debt_equity,
            'Current Ratio': quick_current_ratio,
            'Year': 2024,
            'Quarter': 0
        }
        
        with st.spinner("⚡ Generating quick prediction..."):
            results = enhanced_financial_ai.comprehensive_analysis(quick_data)
            
            st.markdown("---")
            st.subheader("⚡ Quick Analysis Results")
            
            quick_result_col1, quick_result_col2, quick_result_col3 = st.columns(3)
            
            with quick_result_col1:
                st.metric("Investment Score", f"{results['investment_score']}/100")
                st.progress(results['investment_score'] / 100)
            
            with quick_result_col2:
                rec = results['investment_recommendation']
                color = "🟢" if rec in ["Strong Buy", "Buy"] else "🟡" if "Hold" in rec else "🔴"
                st.metric("Recommendation", f"{color} {rec}")
            
            with quick_result_col3:
                st.metric("Confidence", f"{results['investment_confidence']:.0%}")

# ============================================================================
# Custom Analysis Page
# ============================================================================

elif page == "🎯 Custom Analysis":
    st.header("🎯 Custom Financial Analysis")
    st.markdown("*Input your own financial ratios for AI analysis*")
    
    # Custom input form
    st.subheader("📝 Enter Financial Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Company Information")
        custom_company = st.selectbox("Company Type:", ["Almarai", "Savola", "NADEC", "Custom Company"])
        custom_year = st.number_input("Year:", min_value=2020, max_value=2030, value=2024)
        custom_quarter = st.selectbox("Period:", ["Annual", "Q1", "Q2", "Q3", "Q4"])
        
        st.markdown("#### Profitability Ratios")
        gross_margin = st.slider("Gross Margin (%)", 0, 100, 30) / 100
        net_profit_margin = st.slider("Net Profit Margin (%)", 0, 50, 10) / 100
        roa = st.slider("Return on Assets (%)", 0, 30, 8) / 100
    
    with col2:
        st.markdown("#### Financial Health Ratios")
        current_ratio = st.slider("Current Ratio", 0.0, 5.0, 1.5, 0.1)
        debt_to_equity = st.slider("Debt-to-Equity", 0.0, 5.0, 0.8, 0.1)
        debt_to_assets = st.slider("Debt-to-Assets (%)", 0, 100, 45) / 100
        
        st.markdown("#### Optional")
        manual_roe = st.slider("Manual ROE (%) - Optional", 0, 50, 12) / 100
        use_manual_roe = st.checkbox("Use Manual ROE (skip AI prediction)")
    
    # Analysis button
    if st.button("🔍 ANALYZE CUSTOM DATA", type="primary", key="custom_analysis"):
        # Prepare custom data
        custom_data = {
            'Company': custom_company,
            'Year': custom_year,
            'Quarter': 0 if custom_quarter == "Annual" else int(custom_quarter[1]),
            'Gross Margin': gross_margin,
            'Net Profit Margin': net_profit_margin,
            'ROA': roa,
            'Current Ratio': current_ratio,
            'Debt-to-Equity': debt_to_equity,
            'Debt-to-Assets': debt_to_assets
        }
        
        # Add manual ROE if specified
        if use_manual_roe:
            custom_data['ROE'] = manual_roe
        
        with st.spinner("🤖 AI is analyzing your data..."):
            # Get AI analysis
            results = enhanced_financial_ai.comprehensive_analysis(custom_data)
            
            # Add AI status message
            if results.get('prediction_method') == 'AI_COMPREHENSIVE_SYSTEM':
                st.success("🎯 **AI Analysis Complete!** (Using trained models)")
            else:
                st.info("📊 **Mathematical Analysis** (AI models not available)")
            
            st.markdown("---")
            st.subheader(f"🎯 Analysis Results: {custom_company}")
            
            # Create results display
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            
            with result_col1:
                roe_display = manual_roe if use_manual_roe else results['predicted_roe']
                st.metric("ROE", f"{roe_display:.1%}", help="Return on Equity")
            
            with result_col2:
                rec = results['investment_recommendation']
                color = "🟢" if rec in ["Strong Buy", "Buy"] else "🟡" if "Hold" in rec else "🔴"
                st.metric("Investment Rec", f"{color} {rec}")
            
            with result_col3:
                st.metric("AI Confidence", f"{results['investment_confidence']:.0%}")
            
            with result_col4:
                st.metric("Investment Score", f"{results['investment_score']}/100")
            
            # Detailed analysis
            st.markdown("#### 📊 Detailed Analysis")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.markdown("**✅ Strengths:**")
                strengths = []
                if roa > 0.08:
                    strengths.append(f"Strong ROA ({roa:.1%})")
                if net_profit_margin > 0.10:
                    strengths.append(f"Good profitability ({net_profit_margin:.1%})")
                if current_ratio > 1.5:
                    strengths.append(f"Strong liquidity ({current_ratio:.2f})")
                if debt_to_equity < 0.8:
                    strengths.append(f"Conservative debt ({debt_to_equity:.2f})")
                
                if strengths:
                    for strength in strengths:
                        st.write(f"• {strength}")
                else:
                    st.write("• Focus on improving key metrics")
            
            with analysis_col2:
                st.markdown("**⚠️ Areas for Improvement:**")
                concerns = []
                if roa < 0.05:
                    concerns.append(f"Low ROA ({roa:.1%}) - improve asset efficiency")
                if net_profit_margin < 0.05:
                    concerns.append(f"Low margins ({net_profit_margin:.1%}) - reduce costs")
                if current_ratio < 1.2:
                    concerns.append(f"Liquidity risk ({current_ratio:.2f}) - improve cash flow")
                if debt_to_equity > 1.2:
                    concerns.append(f"High leverage ({debt_to_equity:.2f}) - reduce debt")
                
                if concerns:
                    for concern in concerns:
                        st.write(f"• {concern}")
                else:
                    st.write("• Strong performance across all metrics!")

# ============================================================================
# Health Check Page
# ============================================================================

elif page == "🏥 Health Check":
    st.header("🏥 Financial Health Assessment")
    st.markdown("*Comprehensive health analysis using multiple financial indicators*")
    
    if not df.empty:
        # Health check settings
        health_company = st.selectbox("Select Company for Health Check:", sorted(df['Company'].unique()))
        
        # Get latest data
        company_data = df[df['Company'] == health_company]
        annual_data = company_data[company_data['Period_Type'] == 'Annual'].sort_values('Year')
        
        if not annual_data.empty:
            latest_data = annual_data.iloc[-1]
            
            # Health assessment
            if st.button("🔍 Perform Health Check", type="primary"):
                with st.spinner("🏥 Analyzing financial health..."):
                    # Get AI analysis
                    results = enhanced_financial_ai.comprehensive_analysis(latest_data.to_dict())
                    
                    st.markdown("---")
                    st.subheader(f"🏥 Health Report: {health_company}")
                    st.markdown(f"*Assessment Period: {int(latest_data['Year'])} Annual*")
                    
                    # Overall health score
                    health_score = results['investment_score']
                    
                    health_col1, health_col2 = st.columns([1, 2])
                    
                    with health_col1:
                        st.metric("Overall Health Score", f"{health_score}/100")
                        st.progress(health_score / 100)
                        
                        # Health grade
                        if health_score >= 80:
                            st.success("🌟 Grade: A (Excellent)")
                        elif health_score >= 70:
                            st.success("👍 Grade: B (Good)")
                        elif health_score >= 60:
                            st.info("📊 Grade: C (Average)")
                        elif health_score >= 50:
                            st.warning("⚠️ Grade: D (Below Average)")
                        else:
                            st.error("🚨 Grade: F (Poor)")
                    
                    with health_col2:
                        st.markdown("#### 📊 Health Indicators")
                        
                        # Individual health checks
                        health_indicators = [
                            ("Profitability", latest_data.get('ROE', 0), 0.12, "ROE"),
                            ("Asset Efficiency", latest_data.get('ROA', 0), 0.08, "ROA"),
                            ("Profit Margins", latest_data.get('Net Profit Margin', 0), 0.10, "NPM"),
                            ("Liquidity", latest_data.get('Current Ratio', 0), 1.2, "CR"),
                            ("Leverage", latest_data.get('Debt-to-Equity', 0), 1.0, "D/E", True)
                        ]
                        
                        for indicator, value, benchmark, code, *lower_better in health_indicators:
                            is_lower_better = len(lower_better) > 0 and lower_better[0]
                            
                            if pd.notna(value):
                                if is_lower_better:
                                    status = "✅ Healthy" if value <= benchmark else "⚠️ Risk" if value <= benchmark * 1.5 else "🚨 High Risk"
                                else:
                                    status = "✅ Healthy" if value >= benchmark else "⚠️ Below Par" if value >= benchmark * 0.7 else "🚨 Poor"
                                
                                # Display values correctly
                                if code in ["ROE", "ROA", "NPM"]:
                                    value_str = f"{value:.1%}"
                                else:
                                    value_str = f"{value:.2f}"
                                
                                st.write(f"**{indicator}:** {value_str} {status}")
                            else:
                                st.write(f"**{indicator}:** Data not available")

# ============================================================================
# Comparison Page
# ============================================================================

elif page == "⚖️ Comparison":
    st.header("⚖️ Company Comparison Analysis")
    st.markdown("*Side-by-side financial performance comparison*")
    
    if not df.empty:
        # Comparison settings
        comp_col1, comp_col2 = st.columns(2)
        
        with comp_col1:
            available_years = sorted(df['Year'].unique(), reverse=True)
            comp_year = st.selectbox("Comparison Year:", available_years)
        
        with comp_col2:
            comp_period = st.selectbox("Period:", ["Annual"])
        
        # Get comparison data
        comparison_data = df[(df['Year'] == comp_year) & (df['Period_Type'] == 'Annual')]
        
        if not comparison_data.empty:
            st.subheader(f"📊 Company Comparison - {comp_year} {comp_period}")
            
            # Create comparison table
            metrics_to_compare = ['ROE', 'ROA', 'Net Profit Margin', 'Current Ratio', 'Debt-to-Equity']
            available_metrics = [m for m in metrics_to_compare if m in comparison_data.columns]
            
            if available_metrics:
                # Prepare display data
                display_data = comparison_data[['Company'] + available_metrics].copy()
                
                # Format for display
                for metric in available_metrics:
                    if metric in ['ROE', 'ROA', 'Net Profit Margin']:
                        display_data[f"{metric} (%)"] = (display_data[metric] * 100).round(1)
                        display_data = display_data.drop(columns=[metric])
                    else:
                        display_data[metric] = display_data[metric].round(2)
                
                # Display comparison table
                st.dataframe(display_data.set_index('Company'), use_container_width=True)
                
                # Comparison charts
                st.subheader("📈 Visual Comparison")
                
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # ROE comparison
                    if 'ROE' in comparison_data.columns:
                        roe_data = comparison_data[['Company', 'ROE']].copy()
                        fig_roe = px.bar(roe_data, x='Company', y='ROE',
                                        title=f"ROE Comparison - {comp_year} {comp_period}",
                                        color='ROE', color_continuous_scale='viridis')
                        fig_roe.update_layout(yaxis_tickformat='.1%', showlegend=False)
                        st.plotly_chart(fig_roe, use_container_width=True)
                
                with chart_col2:
                    # Current Ratio comparison
                    if 'Current Ratio' in comparison_data.columns:
                        cr_data = comparison_data[['Company', 'Current Ratio']].copy()
                        fig_cr = px.bar(cr_data, x='Company', y='Current Ratio',
                                       title=f"Liquidity Comparison - {comp_year} {comp_period}",
                                       color='Current Ratio', color_continuous_scale='plasma')
                        fig_cr.update_layout(showlegend=False)
                        st.plotly_chart(fig_cr, use_container_width=True)
                
                # Performance ranking
                st.subheader("🏆 Performance Ranking")
                
                # Calculate overall scores for ranking
                ranking_data = []
                
                for _, company_row in comparison_data.iterrows():
                    company_dict = company_row.to_dict()
                    results = enhanced_financial_ai.comprehensive_analysis(company_dict)
                    
                    ranking_data.append({
                        'Company': company_row['Company'],
                        'Overall Score': results['investment_score'],
                        'Investment Rec': results['investment_recommendation'],
                        'AI Confidence': f"{results['investment_confidence']:.0%}"
                    })
                
                ranking_df = pd.DataFrame(ranking_data).sort_values('Overall Score', ascending=False)
                
                # Display ranking
                rank_cols = st.columns(len(ranking_df))
                
                for i, (_, company_data) in enumerate(ranking_df.iterrows()):
                    with rank_cols[i]:
                        position = i + 1
                        medal = "🥇" if position == 1 else "🥈" if position == 2 else "🥉"
                        
                        st.markdown(f"### {medal} {company_data['Company']}")
                        st.metric("Score", f"{company_data['Overall Score']}/100")
                        
                        rec = company_data['Investment Rec']
                        if rec in ["Strong Buy", "Buy"]:
                            st.success(f"📈 {rec}")
                        elif "Hold" in rec:
                            st.warning(f"⚖️ {rec}")
                        else:
                            st.error(f"📉 {rec}")

# ============================================================================
# Footer with Enhanced Information
# ============================================================================

st.markdown("---")
st.markdown("### 🤖 Financial AI Assistant Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("**📊 Data Coverage**")
    if not df.empty:
        st.write(f"• Period: {df['Year'].min()}-{df['Year'].max()}")
        st.write(f"• Companies: {df['Company'].nunique()}")
        st.write(f"• Records: {len(df)}")
    else:
        st.write("• No data loaded")

with info_col2:
    st.markdown("**🤖 AI Models**")
    if ai_system['status'] == 'AI_MODELS_ACTIVE':
        st.write(f"• Status: ✅ Full AI Active")
        st.write(f"• Models: {ai_system['model_count']} loaded")
    elif ai_system['status'] == 'PARTIAL_AI_ACTIVE':
        st.write(f"• Status: ⚠️ Partial AI Active")
        st.write(f"• Models: {ai_system['model_count']} loaded")
    else:
        st.write(f"• Status: 📊 Mathematical Fallback")
        st.write(f"• Models: 0 AI models")

with info_col3:
    st.markdown("**📈 Capabilities**")
    st.write("• ROE Prediction")
    st.write("• Investment Recommendations")
    st.write("• Financial Health Assessment")
    st.write("• Company Comparison")
    st.write("• Custom Analysis")

# Model status expander
with st.expander("🔧 Technical Details", expanded=False):
    st.markdown("**AI System Status:**")
    if ai_system['status'] != 'FALLBACK_MODE':
        st.write("**Available Models:**")
        for model in ai_system['available_models']:
            st.write(f"• {model}: ✅ Loaded")
        st.write(f"**Total Models**: {ai_system['model_count']}")
    else:
        st.write(f"**Status**: {ai_system['status']}")
        st.write(f"**Error**: {ai_system.get('error', 'Unknown error')}")
        st.write("**Note**: Upload model files to activate AI features")
    
    st.markdown("**Features:**")
    st.write("• Real-time financial analysis")
    st.write("• Machine learning predictions")
    st.write("• Interactive visualizations")
    st.write("• Historical trend analysis")
    st.write("• Risk assessment")

# Help section
with st.expander("📖 How to Use", expanded=False):
    st.markdown("""
    **Getting Started:**
    1. **Dashboard**: Overview of all companies and sector trends
    2. **Company Analysis**: Deep dive into specific company performance
    3. **Quick Prediction**: Fast analysis with minimal input
    4. **Custom Analysis**: Enter your own financial ratios
    5. **Health Check**: Comprehensive financial health assessment
    6. **Comparison**: Side-by-side company comparison
    
    **AI Features:**
    - Upload model files (.pkl) to activate full AI capabilities
    - Without AI models, the system uses mathematical fallbacks
    - AI provides more accurate predictions and recommendations
    
    **Tips:**
    - All percentages are automatically handled (enter 30 for 30%)
    - Ratios like Current Ratio and Debt-to-Equity stay as decimals
    - Green recommendations = Buy, Yellow = Hold, Red = Sell
    """)

st.markdown("---")
st.markdown("*Saudi Food Sector Financial AI Assistant | Powered by Advanced Machine Learning*")(f"### {company}")
                    st.metric("ROE", f"{roe:.1%}")
                    
                    # Color-coded recommendation
                    if recommendation in ["Strong Buy", "Buy"]:
                        st.success(f"📈 {recommendation}")
                    elif "Hold" in recommendation:
                        st.warning(f"⚖️ {recommendation}")
                    else:
                        st.error(f"📉 {recommendation}")
        
        # Sector trends chart
        st.subheader("📈 Sector ROE Trends")
        
        # Create trend chart
        annual_data = df[df['Period_Type'] == 'Annual'].groupby(['Year', 'Company'])['ROE'].mean().reset_index()
        
        if not annual_data.empty:
            fig = px.line(
                annual_data, 
                x='Year', 
                y='ROE', 
                color='Company',
                title="ROE Performance Over Time",
                labels={'ROE': 'Return on Equity (%)', 'Year': 'Year'},
                markers=True
            )
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# Company Analysis Page
# ============================================================================

elif page == "📊 Company Analysis":
    st.header("📊 Individual Company Analysis")
    st.markdown("*Deep dive into specific company performance*")
    
    if not df.empty:
        # Company selection
        available_companies = sorted(df['Company'].unique())
        company = st.selectbox("Select Company:", available_companies)
        
        # Get available periods
        company_data = df[df['Company'] == company]
        available_years = sorted(company_data['Year'].unique(), reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.selectbox("Select Year:", available_years)
        
        with col2:
            # Check available period types for the year
            year_data = company_data[company_data['Year'] == year]
            available_periods = []
            
            if not year_data[year_data['Period_Type'] == 'Annual'].empty:
                available_periods.append("Annual")
            
            quarterly_data = year_data[year_data['Period_Type'] == 'Quarterly']
            for q in sorted(quarterly_data['Quarter'].unique()):
                if q > 0:
                    available_periods.append(f"Q{int(q)}")
            
            if available_periods:
                period = st.selectbox("Select Period:", available_periods)
            else:
                st.error(f"No data available for {company} in {year}")
                st.stop()
        
        # Get selected data
        if period == "Annual":
            selected_data = year_data[year_data['Period_Type'] == 'Annual'].iloc[0]
        else:
            quarter_num = int(period[1])
            selected_data = year_data[year_data['Quarter'] == quarter_num].iloc[0]
        
        # Display analysis
        st.subheader(f"📈 {company} - {year} {period}")
        
        # Financial metrics display
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💰 Profitability")
            if pd.notna(selected_data.get('ROE')):
                st.metric("ROE", f"{selected_data['ROE']:.1%}")
            if pd.notna(selected_data.get('ROA')):
                st.metric("ROA", f"{selected_data['ROA']:.1%}")
            if pd.notna(selected_data.get('Net Profit Margin')):
                st.metric("Net Profit Margin", f"{selected_data['Net Profit Margin']:.1%}")
        
        with col2:
            st.markdown("#### ⚖️ Financial Health")
            if pd.notna(selected_data.get('Current Ratio')):
                st.metric("Current Ratio", f"{selected_data['Current Ratio']:.2f}")
            if pd.notna(selected_data.get('Debt-to-Equity')):
                st.metric("Debt-to-Equity", f"{selected_data['Debt-to-Equity']:.2f}")
        
        with col3:
            st.markdown("#### 📊 Efficiency")
            if pd.notna(selected_data.get('Gross Margin')):
                st.metric("Gross Margin", f"{selected_data['Gross Margin']:.1%}")
            if pd.notna(selected_data.get('Debt-to-Assets')):
                st.metric("Debt-to-Assets", f"{selected_data['Debt-to-Assets']:.1%}")
        
        # AI Analysis Button
        if st.button("🤖 Generate AI Analysis", type="primary", key="company_analysis"):
            with st.spinner("Analyzing financial data..."):
                analysis_data = selected_data.to_dict()
                results = enhanced_financial_ai.comprehensive_analysis(analysis_data)
                
                # Add AI status message
                if results.get('prediction_method') == 'AI_COMPREHENSIVE_SYSTEM':
                    st.success("🎯 **AI Analysis Complete!** (Using trained models)")
                else:
                    st.info("📊 **Mathematical Analysis** (AI models not available)")
                
                st.markdown
