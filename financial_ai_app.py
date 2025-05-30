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

# Load AI models with comprehensive error handling
@st.cache_resource
def load_comprehensive_ai_system():
    """Load the comprehensive AI system from Colab"""
    try:
        # Load the main comprehensive predictor
        comprehensive_predictor = joblib.load('comprehensive_ratio_predictor.pkl')
        
        # Load company encoder
        company_encoder = joblib.load('company_encoder.pkl')
        
        # Check if the system has models
        if hasattr(comprehensive_predictor, 'models') and len(comprehensive_predictor.models) > 0:
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
            
    except FileNotFoundError as e:
        return {'status': 'FALLBACK_MODE', 'error': f'Model files not found: {e}'}
    except Exception as e:
        return {'status': 'FALLBACK_MODE', 'error': f'Error loading models: {e}'}

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
                # Load CSV with minimal processing first
                df = pd.read_csv(filename)
                loaded_filename = filename
                break
            except:
                continue
        
        if df is None:
            # Create sample data if no file found
            st.warning("⚠️ CSV file not found. Using sample data for demonstration.")
            return create_sample_data()
        
        st.success(f"✅ Data loaded from: {loaded_filename}")
        
        # Show RAW data before any cleaning
        st.write("🔍 **RAW CSV DATA (first 3 rows):**")
        st.dataframe(df.head(3))
        
        # Show specific values for debugging
        if 'Gross Margin' in df.columns:
            st.write(f"**Raw Gross Margin values:** {df['Gross Margin'].head(5).tolist()}")
            st.write(f"**Gross Margin data type:** {df['Gross Margin'].dtype}")
        
        # DON'T CLEAN AT ALL - just return raw data
        return df
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return create_sample_data()

def clean_financial_data(df):
    """FINAL FIX: Just clean the data, NO percentage conversion at all"""
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Clean all numeric columns - JUST CLEAN, DON'T CONVERT
    all_numeric_columns = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 'Debt-to-Assets', 'Current Ratio', 'Debt-to-Equity']
    
    for col in all_numeric_columns:
        if col in df.columns:
            # Clean strings (remove % signs, commas, spaces)
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
            
            # Convert to numeric - THAT'S IT, NO OTHER CONVERSION
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df
    
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
        for col in percentage_columns + numeric_columns:
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
    
    # Base financial ratios for each company (realistic Saudi market values)
    base_ratios = {
        'Almarai': {
            'Gross Margin': 30.9, 'Net Profit Margin': 10.5, 'ROA': 5.7, 'ROE': 11.5,
            'Current Ratio': 1.40, 'Debt-to-Equity': 1.03, 'Debt-to-Assets': 51.0
        },
        'Savola': {
            'Gross Margin': 20.3, 'Net Profit Margin': 4.5, 'ROA': 4.0, 'ROE': 12.6,
            'Current Ratio': 0.84, 'Debt-to-Equity': 2.14, 'Debt-to-Assets': 68.0
        },
        'NADEC': {
            'Gross Margin': 37.0, 'Net Profit Margin': 9.4, 'ROA': 5.9, 'ROE': 8.4,
            'Current Ratio': 1.96, 'Debt-to-Equity': 0.42, 'Debt-to-Assets': 30.0
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

# Financial AI Class for predictions
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
        """Perform comprehensive analysis using AI or fallbacks"""
        
        if self.status == 'AI_MODELS_ACTIVE' and self.comprehensive_predictor:
            try:
                # Use the comprehensive AI system from Colab
                predictions, complete_data = self.comprehensive_predictor.predict_all_ratios(
                    company_data, 
                    prediction_method='iterative'
                )
                
                # Extract predicted ROE
                predicted_roe = predictions.get('ROE', {}).get('predicted_value', company_data.get('ROA', 0.05) * 1.5)
                
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
        npm = predictions.get('Net Profit Margin', {}).get('predicted_value', complete_data.get('Net Profit Margin', 0))
        if npm > 0.10: score += 20
        elif npm > 0.05: score += 10
        
        # Debt ratios
        debt_equity = predictions.get('Debt-to-Equity', {}).get('predicted_value', complete_data.get('Debt-to-Equity', 1))
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
        roa = company_data.get('ROA', 0.05)
        npm = company_data.get('Net Profit Margin', 0.08)
        equity_multiplier = 1 + company_data.get('Debt-to-Equity', 0.8)
        
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
        """Calculate fallback investment score"""
        score = 0
        roe = data.get('ROE', data.get('ROA', 0.05) * 1.5)
        
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

# Load data and models
df = load_financial_data()
ai_system = load_comprehensive_ai_system()

# Initialize enhanced AI system
enhanced_financial_ai = EnhancedFinancialAI(ai_system)

# Debug section (outside cached function)
if st.sidebar.checkbox("🔍 Show Data Debug Info", value=True):  # DEFAULT TO TRUE to see the problem
    st.sidebar.subheader("📊 Data Debug Information")
    
    if not df.empty:
        st.sidebar.write(f"**Total rows loaded:** {len(df)}")
        st.sidebar.write(f"**Columns:** {list(df.columns)}")
        st.sidebar.write(f"**Companies:** {list(df['Company'].unique())}")
        
        # Show first few rows of raw data
        st.sidebar.write("**First 3 rows of data:**")
        st.sidebar.dataframe(df.head(3))
        
        # Show sample data for Almarai 2023 to verify
        almarai_2023 = df[(df['Company'] == 'Almarai') & (df['Year'] == 2023) & (df['Period_Type'] == 'Annual')]
        st.sidebar.write(f"**Almarai 2023 Annual rows found:** {len(almarai_2023)}")
        
        if not almarai_2023.empty:
            sample_row = almarai_2023.iloc[0]
            st.sidebar.write("**Almarai 2023 Annual Values:**")
            st.sidebar.write(f"• Gross Margin: {sample_row.get('Gross Margin', 0):.1f}%")
            st.sidebar.write(f"• Net Profit Margin: {sample_row.get('Net Profit Margin', 0):.1f}%")
            st.sidebar.write(f"• ROE: {sample_row.get('ROE', 0):.1f}%")
            st.sidebar.write(f"• ROA: {sample_row.get('ROA', 0):.1f}%")
            st.sidebar.write(f"• Current Ratio: {sample_row.get('Current Ratio', 0):.2f}")
            st.sidebar.write(f"• Debt-to-Equity: {sample_row.get('Debt-to-Equity', 0):.2f}")
            
            # Show raw values for debugging
            st.sidebar.write("**Raw Values:**")
            raw_row = almarai_2023.iloc[0]
            st.sidebar.write(f"• Gross Margin Raw: {raw_row.get('Gross Margin', 0):.4f}")
            st.sidebar.write(f"• Expected: 30.9 for 30.9%")
            
            # Show the entire row for debugging
            st.sidebar.write("**Complete Row Data:**")
            st.sidebar.json(dict(sample_row))
        else:
            st.sidebar.error("❌ No Almarai 2023 Annual data found!")
            
            # Show what data we DO have for Almarai
            almarai_data = df[df['Company'] == 'Almarai']
            if not almarai_data.empty:
                st.sidebar.write("**Available Almarai data:**")
                for _, row in almarai_data.head(5).iterrows():
                    st.sidebar.write(f"Year: {row.get('Year', 'N/A')}, Period: {row.get('Period', 'N/A')}, Type: {row.get('Period_Type', 'N/A')}")
    else:
        st.sidebar.error("❌ No data loaded at all!")

# Sidebar for navigation and model status
st.sidebar.title("🎯 Navigation")

# AI System Status Display
st.sidebar.subheader("🤖 AI System Status")

if ai_system['status'] == 'AI_MODELS_ACTIVE':
    st.sidebar.success("🤖 **AI MODELS ACTIVE**")
    st.sidebar.write(f"✅ Comprehensive AI System Loaded")
    st.sidebar.write(f"✅ Available Models: {ai_system['model_count']}")
    st.sidebar.write(f"✅ Company Encoder: Ready")
    
    # Show available AI capabilities
    with st.sidebar.expander("🔍 AI Model Details", expanded=False):
        for model in ai_system['available_models']:
            st.write(f"• {model} Prediction")
else:
    st.sidebar.warning("⚠️ **Using Mathematical Fallbacks**")
    st.sidebar.write(f"Error: {ai_system.get('error', 'Unknown error')}")
    st.sidebar.write("Upload comprehensive_ratio_predictor.pkl to activate AI")

# Main navigation
page = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["🏠 Dashboard", "📊 Company Analysis", "🔮 Ratio Prediction", "🏥 Health Check", "⚖️ Comparison", "🎯 Custom Analysis"]
)

# Dashboard Page
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
                    
                    # Generate recommendation
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

# Company Analysis Page
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

# Custom Analysis Page  
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
            
            # Investment recommendation explanation
            st.markdown("#### 💡 Investment Recommendation Explanation")
            
            score = results['investment_score']
            if score >= 70:
                st.success("🌟 **Strong Buy Recommendation**: Outstanding financial performance with multiple strength indicators. Low risk profile with strong growth potential.")
            elif score >= 60:
                st.success("📈 **Buy Recommendation**: Good financial performance with solid fundamentals. Moderate risk with positive outlook.")
            elif score >= 40:
                st.warning("⚖️ **Hold Recommendation**: Average performance with mixed signals. Monitor for improvements before increasing position.")
            else:
                st.error("📉 **Sell Recommendation**: Below-average performance with concerning metrics. Consider reducing exposure until fundamentals improve.")

# Ratio Prediction Page
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
            confidence_level = st.slider("Confidence Level", 50, 95, 85)
        
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
                    if pd.notna(year_data.get('ROE')):
                        st.metric("ROE", f"{year_data['ROE']:.1%}")
                    if pd.notna(year_data.get('ROA')):
                        st.metric("ROA", f"{year_data['ROA']:.1%}")
            
            # Prediction button
            if st.button("🎯 Generate Predictions", type="primary"):
                with st.spinner("🔮 Generating predictions..."):
                    # Create prediction data based on latest
                    prediction_data = latest_data.to_dict()
                    prediction_data['Year'] = prediction_year
                    
                    # Get AI prediction
                    if prediction_method == "AI Model (Advanced)":
                        results = financial_ai.comprehensive_analysis(prediction_data)
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
                                    growth_rate *= 0.5  # More conservative
                                
                                predicted_roe = recent_roe[1] * (1 + growth_rate)
                                predicted_roe = max(0, min(predicted_roe, 0.5))  # Bound predictions
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
                    
                    # Prediction chart
                    st.subheader("📈 ROE Trend & Prediction")
                    
                    # Prepare chart data
                    chart_data = []
                    
                    # Historical data
                    for _, row in annual_historical.iterrows():
                        if pd.notna(row.get('ROE')):
                            chart_data.append({
                                'Year': int(row['Year']),
                                'ROE': row['ROE'],
                                'Type': 'Historical'
                            })
                    
                    # Predicted data
                    chart_data.append({
                        'Year': prediction_year,
                        'ROE': predicted_roe,
                        'Type': 'Predicted'
                    })
                    
                    chart_df = pd.DataFrame(chart_data)
                    
                    # Create line chart
                    fig = px.line(chart_df, x='Year', y='ROE', color='Type',
                                 title=f"{pred_company} ROE: Historical vs Predicted",
                                 markers=True)
                    fig.update_layout(yaxis_tickformat='.1%')
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Risk assessment
                    st.subheader("⚠️ Prediction Risk Assessment")
                    
                    risk_factors = []
                    if confidence < 0.7:
                        risk_factors.append("Low prediction confidence due to limited historical data")
                    if abs(roe_change) > 20:
                        risk_factors.append("High volatility predicted - monitor closely")
                    if len(annual_historical) < 3:
                        risk_factors.append("Limited historical data available for trend analysis")
                    
                    if risk_factors:
                        for risk in risk_factors:
                            st.warning(f"⚠️ {risk}")
                    else:
                        st.success("✅ Prediction based on stable historical patterns")
        
        else:
            st.warning(f"⚠️ Insufficient historical data for {pred_company}. Need at least 2 years of data for predictions.")

# Health Check Page
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
                    results = financial_ai.comprehensive_analysis(latest_data.to_dict())
                    
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
                            ("Leverage", latest_data.get('Debt-to-Equity', 0), 1.0, "D/E", True)  # Lower is better
                        ]
                        
                        for indicator, value, benchmark, code, *lower_better in health_indicators:
                            is_lower_better = len(lower_better) > 0 and lower_better[0]
                            
                            if pd.notna(value):
                                if is_lower_better:
                                    status = "✅ Healthy" if value <= benchmark else "⚠️ Risk" if value <= benchmark * 1.5 else "🚨 High Risk"
                                else:
                                    status = "✅ Healthy" if value >= benchmark else "⚠️ Below Par" if value >= benchmark * 0.7 else "🚨 Poor"
                                
                                if code in ["ROE", "ROA", "NPM"]:
                                    value_str = f"{value:.1%}"
                                else:
                                    value_str = f"{value:.2f}"
                                
                                st.write(f"**{indicator}:** {value_str} {status}")
                            else:
                                st.write(f"**{indicator}:** Data not available")
                    
                    # Trend analysis
                    if len(annual_data) >= 3:
                        st.subheader("📈 Health Trend Analysis")
                        
                        # Calculate trends for key metrics
                        recent_data = annual_data.tail(3)
                        
                        trend_metrics = ['ROE', 'ROA', 'Net Profit Margin']
                        trend_cols = st.columns(len(trend_metrics))
                        
                        for i, metric in enumerate(trend_metrics):
                            with trend_cols[i]:
                                if metric in recent_data.columns:
                                    values = recent_data[metric].values
                                    if len(values) >= 2 and not np.isnan(values).all():
                                        # Simple trend calculation
                                        trend = "📈 Improving" if values[-1] > values[-2] else "📉 Declining" if values[-1] < values[-2] else "➡️ Stable"
                                        
                                        st.metric(
                                            f"{metric} Trend",
                                            trend,
                                            f"{values[-1]:.1%}" if not np.isnan(values[-1]) else "N/A"
                                        )
                    
                    # Recommendations
                    st.subheader("💡 Health Improvement Recommendations")
                    
                    recommendations = []
                    
                    if latest_data.get('ROE', 0) < 0.10:
                        recommendations.append("🎯 Focus on improving return on equity through better profit margins or asset efficiency")
                    
                    if latest_data.get('Current Ratio', 0) < 1.2:
                        recommendations.append("💧 Improve liquidity by increasing current assets or reducing short-term liabilities")
                    
                    if latest_data.get('Debt-to-Equity', 0) > 1.0:
                        recommendations.append("⚖️ Consider debt reduction to improve financial stability")
                    
                    if latest_data.get('Net Profit Margin', 0) < 0.08:
                        recommendations.append("💰 Work on cost optimization to improve profit margins")
                    
                    if not recommendations:
                        recommendations.append("🌟 Excellent financial health! Continue current management strategies")
                    
                    for rec in recommendations:
                        st.write(f"• {rec}")
        
        else:
            st.error(f"❌ No annual data available for {health_company}")

# Comparison Page
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
            # Check available periods for the year
            year_data = df[df['Year'] == comp_year]
            available_periods = ["Annual"]
            
            quarterly_data = year_data[year_data['Period_Type'] == 'Quarterly']
            for q in sorted(quarterly_data['Quarter'].unique()):
                if q > 0:
                    available_periods.append(f"Q{int(q)}")
            
            comp_period = st.selectbox("Period:", available_periods)
        
        # Get comparison data
        if comp_period == "Annual":
            comparison_data = df[(df['Year'] == comp_year) & (df['Period_Type'] == 'Annual')]
        else:
            quarter_num = int(comp_period[1])
            comparison_data = df[(df['Year'] == comp_year) & (df['Quarter'] == quarter_num)]
        
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
                    results = financial_ai.comprehensive_analysis(company_dict)
                    
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
                        medal = "🥇" if position == 1 else "🥈" if position == 2 else "🥉" if position == 3 else f"{position}."
                        
                        st.markdown(f"### {medal} {company_data['Company']}")
                        st.metric("Score", f"{company_data['Overall Score']}/100")
                        
                        rec = company_data['Investment Rec']
                        if rec in ["Strong Buy", "Buy"]:
                            st.success(f"📈 {rec}")
                        elif "Hold" in rec:
                            st.warning(f"⚖️ {rec}")
                        else:
                            st.error(f"📉 {rec}")
                        
                        st.write(f"Confidence: {company_data['AI Confidence']}")
            
            else:
                st.warning("⚠️ No comparable metrics available for the selected period")
        
        else:
            st.error(f"❌ No data available for {comp_year} {comp_period}")

# Footer with information
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
    models_loaded = sum(1 for status in model_status.values() if "✅" in status)
    total_models = len(model_status)
    st.write(f"• Models: {models_loaded}/{total_models} loaded")
    st.write(f"• Prediction: {'✅ Available' if models['roe_model'] else '⚠️ Fallback'}")
    st.write(f"• Classification: {'✅ Available' if models['investment_model'] else '⚠️ Fallback'}")

with info_col3:
    st.markdown("**📈 Capabilities**")
    st.write("• ROE Prediction")
    st.write("• Investment Recommendations")
    st.write("• Financial Health Assessment")
    st.write("• Company Comparison")
    st.write("• Custom Analysis")

# Model status expander
with st.expander("🔧 Technical Details", expanded=False):
    st.markdown("**Model Status:**")
    for model_name, status in model_status.items():
        st.write(f"• {model_name}: {status}")
    
    st.markdown("**Features:**")
    st.write("• Real-time financial analysis")
    st.write("• Machine learning predictions")
    st.write("• Interactive visualizations")
    st.write("• Historical trend analysis")
    st.write("• Risk assessment")

st.markdown("---")
st.markdown("*Saudi Food Sector Financial AI Assistant | Powered by Advanced Machine Learning*")
