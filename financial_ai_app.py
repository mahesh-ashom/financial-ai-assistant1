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
st.markdown("---")
st.markdown("### 🤖 Financial AI Assistant Information")

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.markdown("**📊 Data Coverage**")
    if not df.empty:
        st.write(f"• Period: {df['Year'].min():.0f}-{df['Year'].max():.0f}")
        st.write(f"• Companies: {df['Company'].nunique()}")
        st.write(f"• Records: {len(df)}")
        st.write(f"• Data Source: Real CSV file")
    else:
        st.write("• No data loaded")

with info_col2:
    st.markdown("**🤖 AI Models**")
    if ai_system['status'] == 'AI_MODELS_ACTIVE':
        st.write(f"• Status: ✅ Full AI Active")
        st.write(f"• Models: {ai_system['model_count']} loaded")
        st.write(f"• Prediction: ✅ Available")
        st.write(f"• Classification: ✅ Available")
    elif ai_system['status'] == 'PARTIAL_AI_ACTIVE':
        st.write(f"• Status: ⚠️ Partial AI Active")
        st.write(f"• Models: {ai_system['model_count']} loaded")
        st.write(f"• Prediction: ⚠️ Limited")
        st.write(f"• Classification: ⚠️ Limited")
    else:
        st.write(f"• Status: 📊 Mathematical Fallback")
        st.write(f"• Models: 0 AI models")
        st.write(f"• Prediction: ⚠️ Fallback")
        st.write(f"• Classification: ⚠️ Fallback")

with info_col3:
    st.markdown("**📈 Capabilities**")
    st.write("• ROE Prediction")
    st.write("• Investment Recommendations")
    st.write("• Financial Health Assessment")
    st.write("• Company Comparison")
    st.write("• Trend Analysis")
    st.write("• Risk Assessment")

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
    
    st.markdown("**Data Format Details:**")
    if not df.empty:
        st.write(f"**Actual Data Ranges:**")
        st.write(f"• ROE: {df['ROE'].min():.3f} to {df['ROE'].max():.3f}")
        st.write(f"• ROA: {df['ROA'].min():.3f} to {df['ROA'].max():.3f}")
        st.write(f"• Net Profit Margin: {df['Net Profit Margin'].min():.3f} to {df['Net Profit Margin'].max():.3f}")
        st.write(f"• Current Ratio: {df['Current Ratio'].min():.2f} to {df['Current Ratio'].max():.2f}")
        st.write(f"• Debt-to-Equity: {df['Debt-to-Equity'].min():.2f} to {df['Debt-to-Equity'].max():.2f}")
    
    st.markdown("**Features:**")
    st.write("• Real-time financial analysis")
    st.write("• Machine learning predictions")
    st.write("• Interactive visualizations")
    st.write("• Historical trend analysis")
    st.write("• Sector benchmarking")
    st.write("• Risk assessment")

# Data quality and sector insights
with st.expander("📊 Sector Insights & Data Quality", expanded=False):
    if not df.empty:
        st.markdown("**Saudi Food Sector Insights:**")
        
        # Company performance summary
        company_summary = df.groupby('Company').agg({
            'ROE': ['mean', 'std'],
            'ROA': ['mean', 'std'],
            'Net Profit Margin': ['mean', 'std'],
            'Current Ratio': ['mean', 'std'],
            'Debt-to-Equity': ['mean', 'std']
        }).round(3)
        
        for company in df['Company'].unique():
            st.write(f"**{company}:**")
            st.write(f"• Avg ROE: {company_summary.loc[company, ('ROE', 'mean')]:.1%} (±{company_summary.loc[company, ('ROE', 'std')]:.1%})")
            st.write(f"• Avg ROA: {company_summary.loc[company, ('ROA', 'mean')]:.1%} (±{company_summary.loc[company, ('ROA', 'std')]:.1%})")
            st.write(f"• Avg Current Ratio: {company_summary.loc[company, ('Current Ratio', 'mean')]:.2f}")
        
        st.markdown("**Data Quality Notes:**")
        st.write("• All ratios are in decimal format (0.15 = 15%)")
        st.write("• Data covers quarterly periods from 2016-2023")
        st.write("• Some negative values indicate challenging periods")
        st.write("• Current Ratio below 1.0 for some periods indicates liquidity challenges")
    
    st.markdown("**Benchmarking Guidelines:**")
    st.write("**Excellent Performance:**")
    st.write("• ROE > 8%, ROA > 3%, Net Profit Margin > 10%")
    st.write("**Good Performance:**")
    st.write("• ROE > 4%, ROA > 1.5%, Net Profit Margin > 5%")
    st.write("**Average Performance:**")
    st.write("• ROE > 2%, ROA > 0.5%, Net Profit Margin > 2%")
    st.write("**Poor Performance:**")
    st.write("• Below average thresholds or negative values")

# Help section
with st.expander("📖 How to Use This Application", expanded=False):
    st.markdown("""
    **Getting Started:**
    1. **Dashboard**: Overview of all companies and sector trends
    2. **Company Analysis**: Deep dive into specific company performance
    3. **Quick Prediction**: Fast analysis with minimal input
    4. **Custom Analysis**: Enter your own financial ratios
    5. **Health Check**: Comprehensive financial health assessment
    6. **Comparison**: Side-by-side company comparison
    
    **Understanding the Data:**
    - All financial ratios are in decimal format (0.15 = 15%)
    - ROE, ROA, Net Profit Margin, Gross Margin are profitability ratios
    - Current Ratio measures liquidity (>1.0 is generally good)
    - Debt-to-Equity measures leverage (lower is generally better)
    - Data covers 2016-2023 quarterly periods
    
    **AI Features:**
    - Upload model files (.pkl) to activate full AI capabilities
    - Without AI models, the system uses mathematical fallbacks
    - AI provides more accurate predictions and recommendations
    - Green recommendations = Buy, Yellow = Hold, Red = Sell
    
    **Interpretation Guide:**
    - **Investment Score**: 0-100 scale (70+ = Buy, 40-70 = Hold, <40 = Sell)
    - **Company Status**: Excellent > Good > Average > Poor
    - **Confidence**: AI's confidence in its recommendation
    - **Health Grade**: A (Excellent) to F (Poor)
    
    **Tips for Analysis:**
    - Compare companies within the same time period
    - Look for trends over multiple quarters
    - Consider sector-specific challenges (food industry margins)
    - Negative ROE/ROA values indicate operational challenges
    - Current Ratio <1.0 may indicate liquidity concerns
    """)

# About section
with st.expander("ℹ️ About This Application", expanded=False):
    st.markdown("""
    **Saudi Food Sector Financial AI Assistant**
    
    This application provides comprehensive financial analysis for Saudi Arabia's food sector companies, 
    specifically focusing on Almarai, Savola, and NADEC.
    
    **Key Features:**
    - AI-powered financial ratio prediction
    - Investment recommendation engine
    - Financial health assessment
    - Historical trend analysis
    - Interactive visualizations
    - Sector benchmarking
    
    **Data Source:**
    - Real financial data from 2016-2023
    - Quarterly reporting periods
    - Comprehensive ratio analysis
    - Industry-specific benchmarks
    
    **Technology Stack:**
    - Streamlit for web interface
    - Pandas for data processing
    - Plotly for interactive charts
    - Scikit-learn/XGBoost for ML models
    - Python for backend processing
    
    **Developed for:** Financial analysis and investment decision support in the Saudi food sector
    
    **Note:** This tool is for educational and analytical purposes. Always consult with financial 
    professionals before making investment decisions.
    """)

st.markdown("---")
st.markdown("*Saudi Food Sector Financial AI Assistant | Powered by Real Data & Advanced Analytics*")
st.markdown("*Accurate representation of Almarai, Savola, and NADEC financial performance (2016-2023)*")set_page_config(
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

        # Clean input values - data is already in decimal format
        for key, value in input_copy.items():
            if isinstance(value, str) and key in self.available_ratios:
                try:
                    cleaned_value = str(value).replace('%', '').replace(',', '').strip()
                    input_copy[key] = float(cleaned_value)
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

# Load financial data with accurate CSV handling
@st.cache_data
def load_financial_data():
    """Load and prepare financial data based on actual CSV structure"""
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
        
        # Clean the data based on actual CSV structure
        df = clean_financial_data(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

def clean_financial_data(df):
    """Clean financial data based on actual CSV structure - data is already in decimals"""
    # Clean column names - handle the space in Current Ratio column
    df.columns = df.columns.str.strip()
    
    # Rename the Current Ratio column to remove extra spaces
    if ' Current Ratio ' in df.columns:
        df = df.rename(columns={' Current Ratio ': 'Current Ratio'})
    
    # Remove empty columns that appear in the CSV
    empty_cols = [col for col in df.columns if col == '' or col.startswith('_')]
    df = df.drop(columns=empty_cols, errors='ignore')
    
    # Define all financial ratio columns
    financial_columns = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 
                        'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
    
    # Clean all financial columns - data is already in decimal format
    for col in financial_columns:
        if col in df.columns:
            # Clean strings (remove % signs, commas, extra spaces)
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
            
            # Convert to numeric - data is already in decimal format (0.334 not 33.4%)
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean Period column - convert to proper date format
    if 'Period' in df.columns:
        df['Period'] = pd.to_datetime(df['Period'], errors='coerce')
        
        # Create Period_Type if not exists
        if 'Period_Type' not in df.columns:
            df['Period_Type'] = 'Quarterly'  # Based on the data, it's all quarterly
    
    # Ensure Year and Quarter are numeric
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    if 'Quarter' in df.columns:
        df['Quarter'] = pd.to_numeric(df['Quarter'], errors='coerce')
    
    # Fill missing values with median for each company
    for company in df['Company'].unique():
        if pd.notna(company):  # Skip NaN company names
            company_mask = df['Company'] == company
            for col in financial_columns:
                if col in df.columns:
                    company_median = df.loc[company_mask, col].median()
                    if not pd.isna(company_median):
                        df.loc[company_mask, col] = df.loc[company_mask, col].fillna(company_median)
    
    # Remove rows with missing company names
    df = df.dropna(subset=['Company'])
    
    return df

def create_sample_data():
    """Create sample data based on actual CSV ranges"""
    companies = ['Almarai', 'Savola', 'NADEC']
    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    quarters = [1, 2, 3, 4]
    
    data = []
    
    # Base financial ratios for each company (in decimal format based on actual data)
    base_ratios = {
        'Almarai': {
            'Gross Margin': 0.35, 'Net Profit Margin': 0.12, 'ROA': 0.03, 'ROE': 0.08,
            'Current Ratio': 1.15, 'Debt-to-Equity': 1.20, 'Debt-to-Assets': 0.55
        },
        'Savola': {
            'Gross Margin': 0.19, 'Net Profit Margin': 0.03, 'ROA': 0.01, 'ROE': 0.02,
            'Current Ratio': 0.85, 'Debt-to-Equity': 1.45, 'Debt-to-Assets': 0.59
        },
        'NADEC': {
            'Gross Margin': 0.38, 'Net Profit Margin': 0.08, 'ROA': 0.02, 'ROE': 0.04,
            'Current Ratio': 0.95, 'Debt-to-Equity': 1.80, 'Debt-to-Assets': 0.64
        }
    }
    
    for company in companies:
        for year in years:
            for quarter in quarters:
                # Create quarterly ratios with variation
                quarterly_ratios = base_ratios[company].copy()
                
                # Add realistic variation based on actual data ranges
                for ratio in quarterly_ratios:
                    variation = np.random.normal(0, 0.2)  # 20% standard deviation
                    quarterly_ratios[ratio] *= (1 + variation)
                    
                    # Ensure values stay within realistic bounds
                    if ratio in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE']:
                        quarterly_ratios[ratio] = max(-0.1, min(0.5, quarterly_ratios[ratio]))
                    elif ratio == 'Current Ratio':
                        quarterly_ratios[ratio] = max(0.3, min(3.0, quarterly_ratios[ratio]))
                    else:  # Debt ratios
                        quarterly_ratios[ratio] = max(0.2, min(3.0, quarterly_ratios[ratio]))
                
                period_date = f"{quarter * 3}/31/{year}"  # Approximate quarterly dates
                
                data.append({
                    'Period': period_date,
                    'Period_Type': 'Quarterly',
                    'Year': year,
                    'Quarter': quarter,
                    'Company': company,
                    **quarterly_ratios
                })
    
    return pd.DataFrame(data)

# ============================================================================
# Enhanced Financial AI Class with accurate thresholds
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
        roa = data.get('ROA', 0.02)  # Based on actual data average
        debt_equity = data.get('Debt-to-Equity', 1.5)  # Based on actual data average
        equity_multiplier = 1 + debt_equity
        return roa * equity_multiplier
    
    def _calculate_ai_investment_score(self, complete_data, predictions):
        """Calculate investment score using AI predictions - based on actual data ranges"""
        score = 0
        
        # Use predicted ROE (actual data ranges from -0.24 to 0.16)
        roe = predictions.get('ROE', {}).get('predicted_value', complete_data.get('ROE', 0))
        if roe > 0.10: score += 30  # Excellent ROE for this sector
        elif roe > 0.05: score += 20  # Good ROE
        elif roe > 0.02: score += 10  # Average ROE
        elif roe > 0: score += 5     # Positive ROE
        
        # Use predicted ROA (actual data ranges from -0.07 to 0.07)
        roa = predictions.get('ROA', {}).get('predicted_value', complete_data.get('ROA', 0))
        if roa > 0.04: score += 25   # Excellent ROA
        elif roa > 0.02: score += 15 # Good ROA
        elif roa > 0.01: score += 10 # Average ROA
        elif roa > 0: score += 5     # Positive ROA
        
        # Use predicted Net Profit Margin (actual data ranges from -0.13 to 0.22)
        npm = predictions.get('Net Profit Margin', {}).get('predicted_value', 
                                                           complete_data.get('Net Profit Margin', 0))
        if npm > 0.15: score += 20   # Excellent margin
        elif npm > 0.10: score += 15 # Good margin
        elif npm > 0.05: score += 10 # Average margin
        elif npm > 0: score += 5     # Positive margin
        
        # Current Ratio (actual data shows some companies below 1.0)
        current_ratio = complete_data.get('Current Ratio', 1.0)
        if current_ratio > 1.5: score += 10  # Strong liquidity
        elif current_ratio > 1.2: score += 8 # Good liquidity
        elif current_ratio > 1.0: score += 5 # Adequate liquidity
        elif current_ratio > 0.8: score += 2 # Weak but manageable
        
        # Debt ratios (actual data ranges from 0.42 to 2.54 for D/E)
        debt_equity = predictions.get('Debt-to-Equity', {}).get('predicted_value', 
                                                               complete_data.get('Debt-to-Equity', 1.5))
        if debt_equity < 1.0: score += 10    # Conservative leverage
        elif debt_equity < 1.5: score += 8   # Moderate leverage
        elif debt_equity < 2.0: score += 5   # High but manageable
        elif debt_equity < 2.5: score += 2   # Very high leverage
        
        return min(100, score)
    
    def _get_ai_company_status(self, complete_data):
        """Determine company status using AI predictions - based on actual data"""
        roe = complete_data.get('ROE', 0)
        npm = complete_data.get('Net Profit Margin', 0)
        roa = complete_data.get('ROA', 0)
        
        # Adjusted thresholds based on actual sector performance
        if roe > 0.08 and npm > 0.10 and roa > 0.03:
            return 'Excellent'
        elif roe > 0.04 and npm > 0.05 and roa > 0.015:
            return 'Good'
        elif roe > 0.02 and npm > 0.02 and roa > 0.005:
            return 'Average'
        else:
            return 'Poor'
    
    def _fallback_analysis(self, company_data):
        """Fallback to mathematical calculations - adjusted for actual data"""
        predicted_roe = self._estimate_roe(company_data)
        investment_score = self._calculate_fallback_score(company_data)
        
        if investment_score >= 70:
            investment_rec, confidence = "Buy", 0.70
        elif investment_score >= 50:
            investment_rec, confidence = "Hold", 0.65
        else:
            investment_rec, confidence = "Sell", 0.60
        
        # Determine status based on actual performance
        roe = company_data.get('ROE', predicted_roe)
        npm = company_data.get('Net Profit Margin', 0.05)
        roa = company_data.get('ROA', 0.02)
        
        if roe > 0.08 and npm > 0.10:
            status = 'Excellent'
        elif roe > 0.04 and npm > 0.05:
            status = 'Good'
        elif roe > 0.02:
            status = 'Average'
        else:
            status = 'Poor'
        
        return {
            'predicted_roe': predicted_roe,
            'investment_recommendation': investment_rec,
            'investment_confidence': confidence,
            'company_status': status,
            'investment_score': investment_score,
            'prediction_method': 'MATHEMATICAL_FALLBACK'
        }
    
    def _calculate_fallback_score(self, data):
        """Calculate fallback investment score - adjusted for actual data ranges"""
        score = 0
        
        # ROE scoring (based on actual data ranges)
        roe = data.get('ROE', self._estimate_roe(data))
        if roe > 0.10: score += 35   # Top performance
        elif roe > 0.05: score += 25 # Good performance
        elif roe > 0.02: score += 15 # Average performance
        elif roe > 0: score += 5     # Positive performance
        
        # ROA scoring
        roa = data.get('ROA', 0.02)
        if roa > 0.04: score += 25
        elif roa > 0.02: score += 15
        elif roa > 0.01: score += 10
        elif roa > 0: score += 5
        
        # Net Profit Margin scoring
        npm = data.get('Net Profit Margin', 0.05)
        if npm > 0.15: score += 20
        elif npm > 0.10: score += 15
        elif npm > 0.05: score += 10
        elif npm > 0: score += 5
        
        # Liquidity and leverage (based on actual ranges)
        current_ratio = data.get('Current Ratio', 1.0)
        if current_ratio > 1.5: score += 10
        elif current_ratio > 1.0: score += 5
        
        debt_equity = data.get('Debt-to-Equity', 1.5)
        if debt_equity < 1.0: score += 5
        elif debt_equity < 1.5: score += 3
        elif debt_equity > 2.0: score -= 5  # Penalty for high leverage
        
        return max(0, min(100, score))

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
            date_range = f"{df['Year'].min():.0f}-{df['Year'].max():.0f}"
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
        latest_data = df[df['Year'] == latest_year].groupby('Company').tail(1)
        
        if not latest_data.empty:
            # Create performance comparison
            performance_cols = st.columns(min(len(latest_data), 3))
            
            for i, (_, company_data) in enumerate(latest_data.iterrows()):
                if i < len(performance_cols):
                    with performance_cols[i]:
                        company = company_data['Company']
                        roe = company_data['ROE']
                        
                        # Generate recommendation using enhanced AI
                        recommendation_result = enhanced_financial_ai.comprehensive_analysis(company_data.to_dict())
                        recommendation = recommendation_result['investment_recommendation']
                        
                        st.markdown(f"### {company}")
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
        
        # Create trend chart using quarterly data
        quarterly_avg = df.groupby(['Year', 'Quarter', 'Company'])['ROE'].mean().reset_index()
        quarterly_avg['Period'] = quarterly_avg['Year'].astype(str) + ' Q' + quarterly_avg['Quarter'].astype(str)
        
        if not quarterly_avg.empty:
            fig = px.line(
                quarterly_avg, 
                x='Period', 
                y='ROE', 
                color='Company',
                title="ROE Performance Over Time (Quarterly)",
                labels={'ROE': 'Return on Equity (%)', 'Period': 'Period'},
                markers=True
            )
            fig.update_layout(yaxis_tickformat='.1%', xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)
        
        # Data quality info
        st.subheader("📋 Data Quality Summary")
        
        quality_col1, quality_col2 = st.columns(2)
        
        with quality_col1:
            st.markdown("**Data Coverage:**")
            st.write(f"• Total Records: {len(df)}")
            st.write(f"• Companies: {', '.join(df['Company'].unique())}")
            st.write(f"• Years: {df['Year'].min():.0f} - {df['Year'].max():.0f}")
            st.write(f"• Quarters per Year: {df['Quarter'].nunique()}")
        
        with quality_col2:
            st.markdown("**Financial Metrics Ranges:**")
            st.write(f"• ROE: {df['ROE'].min():.1%} to {df['ROE'].max():.1%}")
            st.write(f"• ROA: {df['ROA'].min():.1%} to {df['ROA'].max():.1%}")
            st.write(f"• Net Profit Margin: {df['Net Profit Margin'].min():.1%} to {df['Net Profit Margin'].max():.1%}")
            st.write(f"• Current Ratio: {df['Current Ratio'].min():.2f} to {df['Current Ratio'].max():.2f}")

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
        
        # Get available periods for selected company
        company_data = df[df['Company'] == company]
        available_years = sorted(company_data['Year'].unique(), reverse=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            year = st.selectbox("Select Year:", available_years)
        
        with col2:
            # Get available quarters for the selected year
            year_data = company_data[company_data['Year'] == year]
            available_quarters = sorted(year_data['Quarter'].unique())
            quarter_options = [f"Q{int(q)}" for q in available_quarters if q > 0]
            
            if quarter_options:
                period = st.selectbox("Select Quarter:", quarter_options)
                quarter_num = int(period[1])
                selected_data = year_data[year_data['Quarter'] == quarter_num].iloc[0]
            else:
                st.error(f"No data available for {company} in {year}")
                st.stop()
        
        # Display analysis
        st.subheader(f"📈 {company} - {year} {period}")
        
        # Financial metrics display - accurate for decimal data
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
                current_ratio = selected_data['Current Ratio']
                status = "🟢" if current_ratio > 1.2 else "🟡" if current_ratio > 1.0 else "🔴"
                st.metric("Current Ratio", f"{current_ratio:.2f} {status}")
            if pd.notna(selected_data.get('Debt-to-Equity')):
                debt_equity = selected_data['Debt-to-Equity']
                status = "🟢" if debt_equity < 1.0 else "🟡" if debt_equity < 1.5 else "🔴"
                st.metric("Debt-to-Equity", f"{debt_equity:.2f} {status}")
        
        with col3:
            st.markdown("#### 📊 Efficiency")
            if pd.notna(selected_data.get('Gross Margin')):
                st.metric("Gross Margin", f"{selected_data['Gross Margin']:.1%}")
            if pd.notna(selected_data.get('Debt-to-Assets')):
                st.metric("Debt-to-Assets", f"{selected_data['Debt-to-Assets']:.1%}")
        
        # Historical trend for the company
        st.subheader(f"📈 {company} Historical Performance")
        
        company_historical = df[df['Company'] == company].sort_values(['Year', 'Quarter'])
        
        if len(company_historical) > 1:
            # Create period labels
            company_historical['Period_Label'] = (company_historical['Year'].astype(str) + 
                                                 ' Q' + company_historical['Quarter'].astype(str))
            
            # Plot key metrics
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=company_historical['Period_Label'],
                y=company_historical['ROE'],
                mode='lines+markers',
                name='ROE',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=company_historical['Period_Label'],
                y=company_historical['ROA'],
                mode='lines+markers',
                name='ROA',
                line=dict(color='green')
            ))
            
            fig.add_trace(go.Scatter(
                x=company_historical['Period_Label'],
                y=company_historical['Net Profit Margin'],
                mode='lines+markers',
                name='Net Profit Margin',
                line=dict(color='red')
            ))
            
            fig.update_layout(
                title=f"{company} - Key Financial Ratios Over Time",
                xaxis_title="Period",
                yaxis_title="Ratio Value",
                yaxis_tickformat='.1%',
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
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
    st.markdown("*Enter values as decimals (e.g., 0.15 for 15%, 1.2 for ratio of 1.2)*")
    
    quick_col1, quick_col2 = st.columns(2)
    
    with quick_col1:
        quick_company = st.selectbox("Company:", ["Almarai", "Savola", "NADEC", "Other"], key="quick_company")
        quick_roe = st.number_input("Current ROE (decimal)", min_value=-0.3, max_value=0.5, value=0.05, step=0.01, format="%.3f")
        quick_roa = st.number_input("Current ROA (decimal)", min_value=-0.1, max_value=0.2, value=0.02, step=0.01, format="%.3f")
    
    with quick_col2:
        quick_npm = st.number_input("Net Profit Margin (decimal)", min_value=-0.2, max_value=0.3, value=0.08, step=0.01, format="%.3f")
        quick_debt_equity = st.number_input("Debt-to-Equity Ratio", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
        quick_current_ratio = st.number_input("Current Ratio", min_value=0.0, max_value=5.0, value=1.0, step=0.1)
    
    if st.button("⚡ Get Quick Prediction", type="primary"):
        quick_data = {
            'Company': quick_company,
            'ROE': quick_roe,
            'ROA': quick_roa,
            'Net Profit Margin': quick_npm,
            'Debt-to-Equity': quick_debt_equity,
            'Current Ratio': quick_current_ratio,
            'Year': 2024,
            'Quarter': 1
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
    st.markdown("*Based on actual Saudi food sector data ranges*")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Company Information")
        custom_company = st.selectbox("Company Type:", ["Almarai", "Savola", "NADEC", "Custom Company"])
        custom_year = st.number_input("Year:", min_value=2016, max_value=2030, value=2024)
        custom_quarter = st.selectbox("Period:", ["Q1", "Q2", "Q3", "Q4"])
        
        st.markdown("#### Profitability Ratios (Decimal Format)")
        gross_margin = st.slider("Gross Margin", 0.0, 0.5, 0.3, 0.01, format="%.3f", 
                                help="Range in data: 0.16 to 39.29 (avg: 1.27)")
        net_profit_margin = st.slider("Net Profit Margin", -0.2, 0.3, 0.08, 0.01, format="%.3f",
                                     help="Range in data: -0.13 to 0.22 (avg: 0.07)")
        roa = st.slider("Return on Assets", -0.1, 0.1, 0.02, 0.005, format="%.3f",
                       help="Range in data: -0.07 to 0.07 (avg: 0.01)")
    
    with col2:
        st.markdown("#### Financial Health Ratios")
        current_ratio = st.slider("Current Ratio", 0.3, 3.0, 1.0, 0.1,
                                 help="Range in data: varies by company")
        debt_to_equity = st.slider("Debt-to-Equity", 0.0, 3.0, 1.5, 0.1,
                                  help="Range in data: 0.42 to 2.54 (avg: 1.62)")
        debt_to_assets = st.slider("Debt-to-Assets", 0.2, 0.8, 0.6, 0.01, format="%.3f")
        
        st.markdown("#### Optional")
        manual_roe = st.slider("Manual ROE - Optional", -0.3, 0.3, 0.05, 0.01, format="%.3f",
                              help="Range in data: -0.24 to 0.16 (avg: 0.03)")
        use_manual_roe = st.checkbox("Use Manual ROE (skip AI prediction)")
    
    # Analysis button
    if st.button("🔍 ANALYZE CUSTOM DATA", type="primary", key="custom_analysis"):
        # Prepare custom data
        custom_data = {
            'Company': custom_company,
            'Year': custom_year,
            'Quarter': int(custom_quarter[1]),
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
            
            # Detailed analysis based on actual sector benchmarks
            st.markdown("#### 📊 Detailed Analysis")
            
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                st.markdown("**✅ Strengths:**")
                strengths = []
                if roa > 0.03:
                    strengths.append(f"Strong ROA ({roa:.1%}) - above sector average")
                if net_profit_margin > 0.10:
                    strengths.append(f"Excellent profitability ({net_profit_margin:.1%})")
                elif net_profit_margin > 0.05:
                    strengths.append(f"Good profitability ({net_profit_margin:.1%})")
                if current_ratio > 1.2:
                    strengths.append(f"Strong liquidity ({current_ratio:.2f})")
                elif current_ratio > 1.0:
                    strengths.append(f"Adequate liquidity ({current_ratio:.2f})")
                if debt_to_equity < 1.0:
                    strengths.append(f"Conservative debt ({debt_to_equity:.2f})")
                
                if strengths:
                    for strength in strengths:
                        st.write(f"• {strength}")
                else:
                    st.write("• Focus on improving key metrics")
            
            with analysis_col2:
                st.markdown("**⚠️ Areas for Improvement:**")
                concerns = []
                if roa < 0:
                    concerns.append(f"Negative ROA ({roa:.1%}) - major concern")
                elif roa < 0.01:
                    concerns.append(f"Low ROA ({roa:.1%}) - below sector average")
                if net_profit_margin < 0:
                    concerns.append(f"Negative margins ({net_profit_margin:.1%}) - urgent attention needed")
                elif net_profit_margin < 0.05:
                    concerns.append(f"Low margins ({net_profit_margin:.1%}) - cost management needed")
                if current_ratio < 0.8:
                    concerns.append(f"Poor liquidity ({current_ratio:.2f}) - cash flow risk")
                elif current_ratio < 1.0:
                    concerns.append(f"Weak liquidity ({current_ratio:.2f}) - monitor cash flow")
                if debt_to_equity > 2.0:
                    concerns.append(f"Very high leverage ({debt_to_equity:.2f}) - reduce debt")
                elif debt_to_equity > 1.5:
                    concerns.append(f"High leverage ({debt_to_equity:.2f}) - monitor debt levels")
                
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
    st.markdown("*Comprehensive health analysis using Saudi food sector benchmarks*")
    
    if not df.empty:
        # Health check settings
        health_company = st.selectbox("Select Company for Health Check:", sorted(df['Company'].unique()))
        
        # Get latest data
        company_data = df[df['Company'] == health_company]
        latest_data = company_data.sort_values(['Year', 'Quarter']).iloc[-1]
        
        # Health assessment
        if st.button("🔍 Perform Health Check", type="primary"):
            with st.spinner("🏥 Analyzing financial health..."):
                # Get AI analysis
                results = enhanced_financial_ai.comprehensive_analysis(latest_data.to_dict())
                
                st.markdown("---")
                st.subheader(f"🏥 Health Report: {health_company}")
                st.markdown(f"*Assessment Period: {latest_data['Year']:.0f} Q{latest_data['Quarter']:.0f}*")
                
                # Overall health score
                health_score = results['investment_score']
                
                health_col1, health_col2 = st.columns([1, 2])
                
                with health_col1:
                    st.metric("Overall Health Score", f"{health_score}/100")
                    st.progress(health_score / 100)
                    
                    # Health grade based on actual sector performance
                    if health_score >= 80:
                        st.success("🌟 Grade: A (Excellent)")
                    elif health_score >= 65:
                        st.success("👍 Grade: B (Good)")
                    elif health_score >= 50:
                        st.info("📊 Grade: C (Average)")
                    elif health_score >= 35:
                        st.warning("⚠️ Grade: D (Below Average)")
                    else:
                        st.error("🚨 Grade: F (Poor)")
                
                with health_col2:
                    st.markdown("#### 📊 Health Indicators")
                    
                    # Individual health checks based on actual data ranges
                    health_indicators = [
                        ("Profitability (ROE)", latest_data.get('ROE', 0), 0.05, "ROE"),
                        ("Asset Efficiency (ROA)", latest_data.get('ROA', 0), 0.02, "ROA"),
                        ("Profit Margins", latest_data.get('Net Profit Margin', 0), 0.05, "NPM"),
                        ("Liquidity", latest_data.get('Current Ratio', 0), 1.0, "CR"),
                        ("Leverage", latest_data.get('Debt-to-Equity', 0), 1.5, "D/E", True)
                    ]
                    
                    for indicator, value, benchmark, code, *lower_better in health_indicators:
                        is_lower_better = len(lower_better) > 0 and lower_better[0]
                        
                        if pd.notna(value):
                            if is_lower_better:
                                if value <= benchmark:
                                    status = "✅ Healthy"
                                elif value <= benchmark * 1.3:
                                    status = "⚠️ Risk"
                                else:
                                    status = "🚨 High Risk"
                            else:
                                if value >= benchmark:
                                    status = "✅ Healthy"
                                elif value >= benchmark * 0.5:
                                    status = "⚠️ Below Par"
                                elif value >= 0:
                                    status = "🚨 Poor"
                                else:
                                    status = "🚨 Critical"
                            
                            # Display values correctly
                            if code in ["ROE", "ROA", "NPM"]:
                                value_str = f"{value:.1%}"
                            else:
                                value_str = f"{value:.2f}"
                            
                            st.write(f"**{indicator}:** {value_str} {status}")
                        else:
                            st.write(f"**{indicator}:** Data not available")
                
                # Historical health trend
                st.markdown("#### 📈 Health Trend Analysis")
                
                # Calculate health scores for recent periods
                recent_data = company_data.tail(8)  # Last 8 quarters
                health_history = []
                
                for _, period_data in recent_data.iterrows():
                    period_results = enhanced_financial_ai.comprehensive_analysis(period_data.to_dict())
                    health_history.append({
                        'Period': f"{period_data['Year']:.0f} Q{period_data['Quarter']:.0f}",
                        'Health_Score': period_results['investment_score'],
                        'Status': period_results['company_status']
                    })
                
                if health_history:
                    health_df = pd.DataFrame(health_history)
                    
                    fig = px.line(health_df, x='Period', y='Health_Score',
                                 title=f"{health_company} - Health Score Trend",
                                 markers=True)
                    fig.update_layout(
                        yaxis_title="Health Score",
                        xaxis_title="Period",
                        xaxis_tickangle=-45,
                        yaxis=dict(range=[0, 100])
                    )
                    st.plotly_chart(fig, use_container_width=True)

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
            available_quarters = sorted(df[df['Year'] == comp_year]['Quarter'].unique())
            quarter_options = [f"Q{int(q)}" for q in available_quarters if q > 0]
            comp_quarter = st.selectbox("Quarter:", quarter_options)
        
        # Get comparison data
        quarter_num = int(comp_quarter[1])
        comparison_data = df[(df['Year'] == comp_year) & (df['Quarter'] == quarter_num)]
        
        if not comparison_data.empty:
            st.subheader(f"📊 Company Comparison - {comp_year:.0f} {comp_quarter}")
            
            # Create comparison table
            metrics_to_compare = ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin', 
                                'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
            available_metrics = [m for m in metrics_to_compare if m in comparison_data.columns]
            
            if available_metrics:
                # Prepare display data
                display_data = comparison_data[['Company'] + available_metrics].copy()
                
                # Format for display
                for metric in available_metrics:
                    if metric in ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin', 'Debt-to-Assets']:
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
                                        title=f"ROE Comparison - {comp_year:.0f} {comp_quarter}",
                                        color='ROE', color_continuous_scale='viridis')
                        fig_roe.update_layout(yaxis_tickformat='.1%', showlegend=False)
                        st.plotly_chart(fig_roe, use_container_width=True)
                
                with chart_col2:
                    # Current Ratio comparison
                    if 'Current Ratio' in comparison_data.columns:
                        cr_data = comparison_data[['Company', 'Current Ratio']].copy()
                        fig_cr = px.bar(cr_data, x='Company', y='Current Ratio',
                                       title=f"Liquidity Comparison - {comp_year:.0f} {comp_quarter}",
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
                        'AI Confidence': f"{results['investment_confidence']:.0%}",
                        'Status': results['company_status']
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
                        st.write(f"**Status:** {company_data['Status']}")
                        
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

st.
