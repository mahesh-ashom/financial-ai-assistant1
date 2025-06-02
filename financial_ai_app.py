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
# FIXED: ComprehensiveRatiPredictor class with correct calculation logic
# ============================================================================

class ComprehensiveRatiPredictor:
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
                if hasattr(self.le_company, 'classes_') and input_copy['Company'] in self.le_company.classes_:
                    input_copy['Company_Encoded'] = self.le_company.transform([input_copy['Company']])[0]
                else:
                    input_copy['Company_Encoded'] = 0
            except:
                input_copy['Company_Encoded'] = 0

        # FIXED: Clean input values with correct percentage handling
        percentage_columns = ['Gross_Margin', 'Net_Profit_Margin', 'ROA', 'ROE', 'Debt_to_Assets']
        
        for key, value in input_copy.items():
            if isinstance(value, str) and key in percentage_columns:
                try:
                    cleaned_value = str(value).replace('%', '').replace(',', '').strip()
                    numeric_value = float(cleaned_value)
                    
                    # FIXED: Smart conversion - only if value > 1 (meaning percentage form like 8.9)
                    if numeric_value > 1:
                        input_copy[key] = numeric_value / 100  # Convert 8.9 to 0.089
                    else:
                        input_copy[key] = numeric_value        # Keep 0.089 as 0.089
                except:
                    input_copy[key] = np.nan

        # Iterative prediction approach (unchanged)
        if prediction_method == 'iterative' and self.models:
            for iteration in range(3):
                for ratio in self.available_ratios:
                    if ratio not in input_copy or pd.isna(input_copy.get(ratio)):
                        if ratio in self.models:
                            model_info = self.models[ratio]
                            if isinstance(model_info, dict) and 'model' in model_info:
                                model = model_info['model']
                                features = model_info.get('features', [])

                                available_features = [f for f in features
                                                    if f in input_copy and not pd.isna(input_copy.get(f))]

                                if len(available_features) >= len(features) * 0.5:
                                    try:
                                        feature_values = []
                                        for feature in features:
                                            if feature in input_copy and not pd.isna(input_copy.get(feature)):
                                                feature_values.append(input_copy[feature])
                                            else:
                                                feature_values.append(0)

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
    """Load the comprehensive AI system from Colab with enhanced error handling"""
    try:
        comprehensive_predictor = joblib.load('comprehensive_ratio_predictor.pkl')
        
        try:
            company_encoder = joblib.load('company_encoder.pkl')
        except:
            company_encoder = None
        
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
        try:
            individual_models = {}
            model_files = [
                ('model_roe.pkl', 'ROE'),
                ('model_roa.pkl', 'ROA'),
                ('model_net_profit_margin.pkl', 'Net_Profit_Margin'),
                ('model_gross_margin.pkl', 'Gross_Margin'),
                ('model_debt_to_equity.pkl', 'Debt_to_Equity'),
                ('model_debt_to_assets.pkl', 'Debt_to_Assets')
            ]
            
            for model_file, ratio_name in model_files:
                try:
                    model = joblib.load(model_file)
                    individual_models[ratio_name] = model
                except:
                    continue
            
            if individual_models:
                predictor = ComprehensiveRatiPredictor(individual_models, None)
                
                return {
                    'status': 'AI_MODELS_ACTIVE',
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

# ============================================================================
# FIXED: Data loading and cleaning functions
# ============================================================================

@st.cache_data
def load_financial_data():
    """Load and prepare financial data with FIXED cleaning"""
    try:
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
        
        # Clean the data with FIXED logic
        df = clean_financial_data(df)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

def clean_financial_data(df):
    """Clean financial data - FIXED: Keep original decimal values"""
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # FIXED: Define column types clearly
    numeric_columns = ['Gross_Margin', 'Net_Profit_Margin', 'ROA', 'ROE', 'Debt_to_Assets', 'Current_Ratio', 'Debt_to_Equity']
    
    for col in numeric_columns:
        if col in df.columns:
            # Clean strings (remove % signs, commas, spaces)
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
            
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # FIXED: Only convert to decimal if value > 1 (percentage form)
            if col in ['Gross_Margin', 'Net_Profit_Margin', 'ROA', 'ROE', 'Debt_to_Assets']:
                df[col] = df[col].apply(lambda x: x/100 if pd.notna(x) and x > 1 else x)
    
    # Create derived columns if missing
    if 'Period_Type' not in df.columns and 'Period' in df.columns:
        df['Period_Type'] = df['Period'].apply(
            lambda x: 'Annual' if 'Annual' in str(x) else 'Quarterly'
        )
    
    if 'Year' not in df.columns and 'Period' in df.columns:
        df['Year'] = df['Period'].str.extract(r'(\d{4})').astype(float).fillna(2023).astype(int)
    
    if 'Quarter' not in df.columns and 'Period' in df.columns:
        df['Quarter'] = df['Period'].str.extract(r'Q(\d)').fillna(0).astype(int)
    
    return df

def create_sample_data():
    """FIXED: Create sample data with consistent decimal format"""
    companies = ['Almarai', 'Savola', 'NADEC']
    years = [2020, 2021, 2022, 2023]
    quarters = [1, 2, 3, 4]
    
    data = []
    
    # FIXED: Base ratios stored as decimals (0.115 = 11.5%)
    base_ratios = {
        'Almarai': {
            'Gross_Margin': 0.309,      # 30.9%
            'Net_Profit_Margin': 0.105, # 10.5%
            'ROA': 0.057,               # 5.7%
            'ROE': 0.115,               # 11.5%
            'Current_Ratio': 1.40,      # Ratio
            'Debt_to_Equity': 1.03,     # Ratio
            'Debt_to_Assets': 0.51      # 51%
        },
        'Savola': {
            'Gross_Margin': 0.203,      # 20.3%
            'Net_Profit_Margin': 0.045, # 4.5%
            'ROA': 0.040,               # 4.0%
            'ROE': 0.126,               # 12.6%
            'Current_Ratio': 0.84,      # Ratio
            'Debt_to_Equity': 2.14,     # Ratio
            'Debt_to_Assets': 0.68      # 68%
        },
        'NADEC': {
            'Gross_Margin': 0.370,      # 37.0%
            'Net_Profit_Margin': 0.094, # 9.4%
            'ROA': 0.059,               # 5.9%
            'ROE': 0.084,               # 8.4%
            'Current_Ratio': 1.96,      # Ratio
            'Debt_to_Equity': 0.42,     # Ratio
            'Debt_to_Assets': 0.30      # 30%
        }
    }
    
    for company in companies:
        for year in years:
            # Annual data
            annual_ratios = base_ratios[company].copy()
            # Add some year-over-year variation
            for ratio in annual_ratios:
                annual_ratios[ratio] *= (1 + np.random.normal(0, 0.1))
            
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
# FIXED: Enhanced Financial AI Class with correct calculation logic
# ============================================================================

class EnhancedFinancialAI:
    def __init__(self, ai_system):
        self.ai_system = ai_system
        self.status = ai_system['status']
        
        if self.status == 'AI_MODELS_ACTIVE':
            self.comprehensive_predictor = ai_system['comprehensive_predictor']
            self.company_encoder = ai_system['company_encoder']
        else:
            self.comprehensive_predictor = None
            self.company_encoder = None
    
    def comprehensive_analysis(self, company_data):
        """Perform comprehensive analysis using AI or fallbacks - FIXED"""
        
        if self.status == 'AI_MODELS_ACTIVE' and self.comprehensive_predictor:
            try:
                predictions, complete_data = self.comprehensive_predictor.predict_all_ratios(
                    company_data, 
                    prediction_method='iterative'
                )
                
                predicted_roe = predictions.get('ROE', {}).get('predicted_value', company_data.get('ROA', 0.05) * 1.5)
                investment_score = self._calculate_ai_investment_score(complete_data, predictions)
                
                if investment_score >= 70:
                    investment_rec, confidence = "Strong Buy", 0.85
                elif investment_score >= 60:
                    investment_rec, confidence = "Buy", 0.75
                elif investment_score >= 40:
                    investment_rec, confidence = "Hold", 0.65
                else:
                    investment_rec, confidence = "Sell", 0.55
                
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
    
    def _calculate_ai_investment_score(self, complete_data, predictions):
        """FIXED: Calculate investment score using correct decimal thresholds"""
        score = 0
        
        # FIXED: Use decimal thresholds (0.15 = 15%)
        roe = predictions.get('ROE', {}).get('predicted_value', complete_data.get('ROE', 0))
        if roe > 0.15: score += 30      # 15%
        elif roe > 0.10: score += 20    # 10%
        elif roe > 0.05: score += 10    # 5%
        
        roa = predictions.get('ROA', {}).get('predicted_value', complete_data.get('ROA', 0))
        if roa > 0.08: score += 25      # 8%
        elif roa > 0.05: score += 15    # 5%
        elif roa > 0.02: score += 5     # 2%
        
        npm = predictions.get('Net_Profit_Margin', {}).get('predicted_value', complete_data.get('Net_Profit_Margin', 0))
        if npm > 0.10: score += 20      # 10%
        elif npm > 0.05: score += 10    # 5%
        
        # Debt ratios (these stay as ratios)
        debt_equity = predictions.get('Debt_to_Equity', {}).get('predicted_value', complete_data.get('Debt_to_Equity', 1))
        if debt_equity < 0.5: score += 15
        elif debt_equity < 1.0: score += 10
        elif debt_equity < 1.5: score += 5
        
        return min(100, score)
    
    def _get_ai_company_status(self, complete_data):
        """FIXED: Determine company status using decimal thresholds"""
        roe = complete_data.get('ROE', 0)
        npm = complete_data.get('Net_Profit_Margin', 0)
        
        if roe > 0.15 and npm > 0.15:      # 15%
            return 'Excellent'
        elif roe > 0.10 and npm > 0.10:    # 10%
            return 'Good'
        elif roe > 0.05 and npm > 0.05:    # 5%
            return 'Average'
        else:
            return 'Poor'
    
    def _fallback_analysis(self, company_data):
        """FIXED: Fallback to mathematical calculations with decimal logic"""
        roa = company_data.get('ROA', 0.05)
        npm = company_data.get('Net_Profit_Margin', 0.08)
        equity_multiplier = 1 + company_data.get('Debt_to_Equity', 0.8)
        
        predicted_roe = roa * equity_multiplier
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
        """FIXED: Calculate fallback investment score with decimal thresholds"""
        score = 0
        roe = data.get('ROE', data.get('ROA', 0.05) * 1.5)
        
        # FIXED: Use decimal thresholds
        if roe > 0.15: score += 35       # 15%
        elif roe > 0.10: score += 25     # 10%
        elif roe > 0.05: score += 15     # 5%
        
        roa = data.get('ROA', 0.05)
        if roa > 0.08: score += 25       # 8%
        elif roa > 0.05: score += 15     # 5%
        
        npm = data.get('Net_Profit_Margin', 0.08)
        if npm > 0.10: score += 20       # 10%
        elif npm > 0.05: score += 10     # 5%
        
        return min(100, score)

# ============================================================================
# FIXED: Utility functions for consistent display
# ============================================================================

def display_percentage_metric(label, value, help_text=None):
    """FIXED: Display percentage metrics consistently"""
    if pd.isna(value):
        st.metric(label, "N/A", help=help_text)
    else:
        # Value is in decimal format (0.089), display as percentage
        st.metric(label, f"{value:.1%}", help=help_text)

def display_ratio_metric(label, value, help_text=None):
    """Display ratio metrics consistently"""
    if pd.isna(value):
        st.metric(label, "N/A", help=help_text)
    else:
        # Value is already a ratio, display as number
        st.metric(label, f"{value:.2f}", help=help_text)

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

st.sidebar.title("🎯 Navigation")

# AI System Status Display
st.sidebar.subheader("🤖 AI System Status")

if ai_system['status'] == 'AI_MODELS_ACTIVE':
    st.sidebar.success("🤖 **AI MODELS ACTIVE**")
    st.sidebar.write(f"✅ Comprehensive AI System Loaded")
    st.sidebar.write(f"✅ Available Models: {ai_system['model_count']}")
    st.sidebar.write(f"✅ Company Encoder: {'Ready' if ai_system['company_encoder'] else 'Fallback'}")
    
    with st.sidebar.expander("🔍 AI Model Details", expanded=False):
        st.write("**Available AI Models:**")
        for model in ai_system['available_models']:
            st.write(f"• {model} Prediction")
        st.write("**Capabilities:**")
        st.write("• ROE Prediction")
        st.write("• Investment Recommendations")
        st.write("• Financial Health Assessment")
        st.write("• Risk Analysis")
else:
    st.sidebar.warning("⚠️ **Using Mathematical Fallbacks**")
    st.sidebar.write(f"Error: {ai_system.get('error', 'Unknown error')}")
    st.sidebar.write("Upload comprehensive_ratio_predictor.pkl to activate AI")

# Main navigation
page = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["🏠 Dashboard", "📊 Company Analysis", "🔮 Ratio Prediction", "🏥 Health Check", "⚖️ Comparison", "🎯 Custom Analysis"]
)

# ============================================================================
# FIXED: Dashboard Page
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
            # FIXED: Display as percentage
            display_percentage_metric("Avg Sector ROE", avg_roe)
        
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
                    
                    st.markdown(f"### {company}")
                    # FIXED: Display ROE as percentage
                    display_percentage_metric("ROE", roe)
                    
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
            # FIXED: Display Y-axis as percentage
            fig.update_layout(yaxis_tickformat='.1%')
            st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# FIXED: Company Analysis Page
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
        
        # FIXED: Financial metrics display with correct formatting
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 💰 Profitability")
            display_percentage_metric("ROE", selected_data.get('ROE'), "Return on Equity")
            display_percentage_metric("ROA", selected_data.get('ROA'), "Return on Assets")
            display_percentage_metric("Net Profit Margin", selected_data.get('Net_Profit_Margin'), "Net Profit Margin")
        
        with col2:
            st.markdown("#### ⚖️ Financial Health")
            display_ratio_metric("Current Ratio", selected_data.get('Current_Ratio'), "Liquidity measure")
            display_ratio_metric("Debt-to-Equity", selected_data.get('Debt_to_Equity'), "Leverage ratio")
        
        with col3:
            st.markdown("#### 📊 Efficiency")
            display_percentage_metric("Gross Margin", selected_data.get('Gross_Margin'), "Gross profit efficiency")
            display_percentage_metric("Debt-to-Assets", selected_data.get('Debt_to_Assets'), "Asset financing structure")
        
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
                    display_percentage_metric("AI Predicted ROE", results['predicted_roe'])
                
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
# FIXED: Custom Analysis Page
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
        # FIXED: Input as percentage but store as decimal for calculations
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
            'Gross_Margin': gross_margin,
            'Net_Profit_Margin': net_profit_margin,
            'ROA': roa,
            'Current_Ratio': current_ratio,
            'Debt_to_Equity': debt_to_equity,
            'Debt_to_Assets': debt_to_assets
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
                display_percentage_metric("ROE", roe_display, "Return on Equity")
            
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
# FIXED: Ratio Prediction Page
# ============================================================================

elif page == "🔮 Ratio Prediction":
    st.header("🔮 Financial Ratio Prediction")
    st.markdown("*Predict future financial performance using AI and historical trends*")
    
    if not df.empty:
        # Prediction settings
        pred_col1, pred_col2 = st.columns(2)
        
        with pred_col1:
            available_companies = sorted(df['Company'].unique())
            pred_company = st.selectbox("Company for Prediction:", available_companies)
            prediction_year = st.selectbox("Predict for Year:", [2024, 2025, 2026, 2027])
        
        with pred_col2:
            prediction_method = st.selectbox("Prediction Method:", 
                                           ["AI Model (Advanced)", "Trend Analysis", "Conservative Estimate"])
        
        # Get historical data for context
        company_historical = df[df['Company'] == pred_company]
        annual_historical = company_historical[company_historical['Period_Type'] == 'Annual'].sort_values('Year')
        
        if len(annual_historical) >= 2:
            latest_data = annual_historical.iloc[-1]
            
            st.subheader(f"📊 Historical Context: {pred_company}")
            
            # Show recent trend
            recent_years = annual_historical.tail(3)
            
            trend_col1, trend_col2, trend_col3 = st.columns(3)
            
            for i, (_, year_data) in enumerate(recent_years.iterrows()):
                with [trend_col1, trend_col2, trend_col3][i]:
                    st.markdown(f"**{int(year_data['Year'])}**")
                    display_percentage_metric("ROE", year_data.get('ROE'))
                    display_percentage_metric("ROA", year_data.get('ROA'))
            
            # Prediction button
            if st.button("🎯 Generate Predictions", type="primary"):
                with st.spinner("🔮 Generating predictions..."):
                    # Create prediction data based on latest
                    prediction_data = latest_data.to_dict()
                    prediction_data['Year'] = prediction_year
                    
                    # Get AI prediction
                    if prediction_method == "AI Model (Advanced)":
                        results = enhanced_financial_ai.comprehensive_analysis(prediction_data)
                        predicted_roe = results['predicted_roe']
                        investment_rec = results['investment_recommendation']
                        confidence = results['investment_confidence']
                    else:
                        # Simple trend-based prediction
                        if len(annual_historical) >= 2:
                            recent_roe = annual_historical['ROE'].tail(2).values
                            if len(recent_roe) == 2 and not np.isnan(recent_roe).any():
                                growth_rate = (recent_roe[1] - recent_roe[0]) / recent_roe[0] if recent_roe[0] != 0 else 0
                                
                                if prediction_method == "Conservative Estimate":
                                    growth_rate *= 0.5
                                
                                predicted_roe = recent_roe[1] * (1 + growth_rate)
                                predicted_roe = max(0, min(predicted_roe, 0.5))
                            else:
                                predicted_roe = latest_data.get('ROE', 0.1)
                        else:
                            predicted_roe = latest_data.get('ROE', 0.1)
                        
                        # Simple recommendation based on predicted ROE
                        if predicted_roe > 0.15:
                            investment_rec = "Buy"
                            confidence = 0.75
                        elif predicted_roe > 0.08:
                            investment_rec = "Hold"
                            confidence = 0.70
                        else:
                            investment_rec = "Sell"
                            confidence = 0.65
                    
                    st.markdown("---")
                    st.subheader(f"🔮 {pred_company} - {prediction_year} Predictions")
                    
                    # Display predictions
                    pred_result_col1, pred_result_col2, pred_result_col3 = st.columns(3)
                    
                    with pred_result_col1:
                        current_roe = latest_data.get('ROE', 0)
                        roe_change = ((predicted_roe - current_roe) / current_roe * 100) if current_roe != 0 else 0
                        st.metric("Predicted ROE", f"{predicted_roe:.1%}", f"{roe_change:+.1f}%")
                    
                    with pred_result_col2:
                        st.metric("Investment Outlook", investment_rec)
                    
                    with pred_result_col3:
                        st.metric("Prediction Confidence", f"{confidence:.0%}")

# ============================================================================
# FIXED: Health Check Page
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
                        
                        # FIXED: Individual health checks with correct thresholds
                        health_indicators = [
                            ("Profitability", latest_data.get('ROE', 0), 0.12, "ROE"),
                            ("Asset Efficiency", latest_data.get('ROA', 0), 0.08, "ROA"),
                            ("Profit Margins", latest_data.get('Net_Profit_Margin', 0), 0.10, "NPM"),
                            ("Liquidity", latest_data.get('Current_Ratio', 0), 1.2, "CR"),
                            ("Leverage", latest_data.get('Debt_to_Equity', 0), 1.0, "D/E", True)
                        ]
                        
                        for indicator, value, benchmark, code, *lower_better in health_indicators:
                            is_lower_better = len(lower_better) > 0 and lower_better[0]
                            
                            if pd.notna(value):
                                if is_lower_better:
                                    status = "✅ Healthy" if value <= benchmark else "⚠️ Risk" if value <= benchmark * 1.5 else "🚨 High Risk"
                                else:
                                    status = "✅ Healthy" if value >= benchmark else "⚠️ Below Par" if value >= benchmark * 0.7 else "🚨 Poor"
                                
                                # FIXED: Display percentages correctly
                                if code in ["ROE", "ROA", "NPM"]:
                                    value_str = f"{value:.1%}"
                                else:
                                    value_str = f"{value:.2f}"
                                
                                st.write(f"**{indicator}:** {value_str} {status}")
                            else:
                                st.write(f"**{indicator}:** Data not available")

# ============================================================================
# FIXED: Comparison Page
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
            metrics_to_compare = ['ROE', 'ROA', 'Net_Profit_Margin', 'Current_Ratio', 'Debt_to_Equity']
            available_metrics = [m for m in metrics_to_compare if m in comparison_data.columns]
            
            if available_metrics:
                # FIXED: Prepare display data with correct percentage formatting
                display_data = comparison_data[['Company'] + available_metrics].copy()
                
                # Format for display
                for metric in available_metrics:
                    if metric in ['ROE', 'ROA', 'Net_Profit_Margin']:
                        # Convert to percentage for display
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
                    if 'Current_Ratio' in comparison_data.columns:
                        cr_data = comparison_data[['Company', 'Current_Ratio']].copy()
                        fig_cr = px.bar(cr_data, x='Company', y='Current_Ratio',
                                       title=f"Liquidity Comparison - {comp_year} {comp_period}",
                                       color='Current_Ratio', color_continuous_scale='plasma')
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
        st.write(f"• Models: {ai_system['model_count']} AI models loaded")
        st.write(f"• Prediction: ✅ Available")
        st.write(f"• Classification: ✅ Available")
    else:
        st.write(f"• Models: Using mathematical fallbacks")
        st.write(f"• Prediction: ⚠️ Fallback")
        st.write(f"• Classification: ⚠️ Fallback")

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
    if ai_system['status'] == 'AI_MODELS_ACTIVE':
        st.write("**Available AI Models:**")
        for model in ai_system['available_models']:
            st.write(f"• {model}: ✅ Loaded")
        st.write(f"**Total Models**: {ai_system['model_count']}")
    else:
        st.write(f"**Status**: {ai_system['status']}")
        st.write(f"**Error**: {ai_system.get('error', 'Unknown error')}")
    
    st.markdown("**Features:**")
    st.write("• Real-time financial analysis")
    st.write("• Machine learning predictions")
    st.write("• Interactive visualizations")
    st.write("• Historical trend analysis")
    st.write("• Risk assessment")

st.markdown("---")
st.markdown("*Saudi Food Sector Financial AI Assistant | Powered by Advanced Machine Learning*")
