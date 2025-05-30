import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Financial AI Assistant", 
    page_icon="🤖",
    layout="wide"
)

# Title and description
st.title("🤖 Financial AI Assistant")
st.markdown("**Saudi Food Sector Investment Analysis System**")
st.markdown("Analyze Almarai, Savola, and NADEC with AI-powered insights - Enhanced Version")

# Load real financial data from your CSV
@st.cache_data
def load_financial_data():
    """Load and prepare real financial data from actual CSV file"""
    try:
        # Try to load the actual CSV file - try multiple possible filenames
        possible_filenames = [
            'Savola Almarai NADEC Financial Ratios CSV.csv.csv',
            'Savola Almarai NADEC Financial Ratios CSV.csv', 
            'Savola_Almarai_NADEC_Financial_Ratios_CSV.csv',
            'financial_data.csv'
        ]
        
        df = None
        for filename in possible_filenames:
            try:
                df = pd.read_csv(filename)
                st.success(f"✅ Loaded data from: {filename}")
                break
            except:
                continue
        
        if df is None:
            raise FileNotFoundError("Could not find CSV file with any expected filename")
        
        # Clean column names
        df.columns = df.columns.str.strip()
        
        # Convert percentage columns to decimal format if they're strings
        percentage_columns = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 'Debt-to-Assets']
        
        for col in percentage_columns:
            if col in df.columns:
                if df[col].dtype == 'object':
                    # Remove % sign and convert to decimal
                    df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '')
                    df[col] = pd.to_numeric(df[col], errors='coerce') / 100
                else:
                    # Already numeric, check if it needs conversion from percentage
                    if df[col].max() > 1:
                        df[col] = df[col] / 100
        
        # Ensure numeric columns are properly formatted
        numeric_columns = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create Period_Type column if it doesn't exist
        if 'Period_Type' not in df.columns:
            df['Period_Type'] = df['Period'].apply(lambda x: 'Annual' if 'Annual' in str(x) or (isinstance(x, (int, float)) and x == int(x)) else 'Quarterly')
        
        # Extract Year and Quarter if not present
        if 'Year' not in df.columns:
            df['Year'] = df['Period'].str.extract(r'(\d{4})').astype(int)
        
        if 'Quarter' not in df.columns:
            df['Quarter'] = df['Period'].str.extract(r'Q(\d)').fillna(0).astype(int)
        
        return df
        
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        # Fallback to empty dataframe
        return pd.DataFrame()

# Load AI models with error handling
@st.cache_resource
def load_ai_models():
    """Load AI models if available"""
    try:
        comprehensive_predictor = joblib.load('comprehensive_ratio_predictor.pkl')
        company_encoder = joblib.load('company_encoder.pkl')
        return comprehensive_predictor, company_encoder, True
    except:
        return None, None, False

# Load data and models
df = load_financial_data()
comprehensive_predictor, company_encoder, models_loaded = load_ai_models()

# Check if data loaded successfully
if df.empty:
    st.error("❌ Could not load financial data. Please ensure the CSV file is uploaded correctly.")
    st.stop()

# Display model status
if models_loaded:
    st.success("✅ Financial AI Assistant is ready! Models loaded successfully.")
else:
    st.info("📊 Financial AI Assistant is ready! Using historical data analysis.")

def get_company_data(company, year, quarter=None, period_type='Quarterly'):
    """Get real financial data for specific company and period"""
    if period_type == 'Annual':
        # For annual data, look for records with Annual in Period_Type or Quarter = 0
        mask = (
            (df['Company'] == company) & 
            (df['Year'] == year) & 
            (df['Period_Type'] == 'Annual')
        )
    else:
        # For quarterly data
        mask = (
            (df['Company'] == company) & 
            (df['Year'] == year) & 
            (df['Quarter'] == quarter) &
            (df['Period_Type'] == 'Quarterly')
        )
    
    data = df[mask]
    if not data.empty:
        return data.iloc[0].to_dict()
    return None

def get_available_periods(company):
    """Get all available periods for a company"""
    company_data = df[df['Company'] == company]
    
    # Get unique years
    years = sorted(company_data['Year'].unique())
    
    # Get available quarters by year
    quarters_by_year = {}
    annual_by_year = {}
    
    for year in years:
        year_data = company_data[company_data['Year'] == year]
        
        # Check for quarterly data
        quarterly_data = year_data[year_data['Period_Type'] == 'Quarterly']
        if not quarterly_data.empty:
            quarters_by_year[year] = sorted(quarterly_data['Quarter'].unique())
        
        # Check for annual data
        annual_data = year_data[year_data['Period_Type'] == 'Annual']
        if not annual_data.empty:
            annual_by_year[year] = True
    
    return years, quarters_by_year, annual_by_year

def predict_future_ratios(company, base_year=2023):
    """Predict future ratios based on historical trends from real data"""
    company_data = df[df['Company'] == company]
    
    # Get annual data for trend analysis
    annual_data = company_data[company_data['Period_Type'] == 'Annual'].sort_values('Year')
    
    if len(annual_data) < 2:
        return {}
    
    # Use last two years for trend calculation
    latest_annual = annual_data.iloc[-1]
    previous_annual = annual_data.iloc[-2]
    
    predictions = {}
    ratios = ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin', 'Current Ratio', 'Debt-to-Equity']
    
    for ratio in ratios:
        if ratio in latest_annual and ratio in previous_annual:
            try:
                # Calculate growth rate
                current_val = latest_annual[ratio]
                previous_val = previous_annual[ratio]
                
                if previous_val != 0:
                    growth_rate = (current_val - previous_val) / previous_val
                else:
                    growth_rate = 0
                
                # Apply conservative growth (50% of historical trend)
                predicted_value = current_val * (1 + growth_rate * 0.5)
                
                # Add bounds checking
                if ratio in ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin']:
                    predicted_value = max(0, min(predicted_value, 0.8))  # 0-80% range
                elif ratio == 'Current Ratio':
                    predicted_value = max(0.1, min(predicted_value, 10.0))  # 0.1-10.0 range
                elif ratio == 'Debt-to-Equity':
                    predicted_value = max(0, min(predicted_value, 5.0))   # 0-5.0 range
                    
                predictions[ratio] = {
                    'predicted_value': predicted_value,
                    'confidence': 'High' if abs(growth_rate) < 0.2 else 'Medium',
                    'historical_trend': growth_rate,
                    'current_value': current_val
                }
            except:
                continue
    
    return predictions

def generate_investment_recommendation(data):
    """Generate investment recommendation based on real data"""
    roe = data.get('ROE', 0)
    roa = data.get('ROA', 0)
    npm = data.get('Net Profit Margin', 0)
    current_ratio = data.get('Current Ratio', 0)
    debt_equity = data.get('Debt-to-Equity', 0)
    
    # Scoring system based on real Saudi market benchmarks
    score = 0
    
    # ROE scoring (35% weight) - adjusted for Saudi market
    if roe > 0.20: score += 35      # Exceptional
    elif roe > 0.15: score += 30    # Excellent  
    elif roe > 0.12: score += 25    # Very Good
    elif roe > 0.08: score += 15    # Good
    elif roe > 0.05: score += 8     # Fair
    elif roe > 0.02: score += 3     # Poor
    
    # ROA scoring (25% weight)
    if roa > 0.12: score += 25      # Exceptional
    elif roa > 0.08: score += 20    # Excellent
    elif roa > 0.06: score += 15    # Good
    elif roa > 0.04: score += 10    # Fair
    elif roa > 0.02: score += 5     # Poor
    
    # NPM scoring (20% weight)
    if npm > 0.15: score += 20      # Excellent
    elif npm > 0.10: score += 15    # Good
    elif npm > 0.05: score += 10    # Fair
    elif npm > 0.02: score += 5     # Poor
    
    # Liquidity scoring (10% weight)
    if current_ratio > 2.0: score += 10    # Very Strong
    elif current_ratio > 1.5: score += 8   # Strong
    elif current_ratio > 1.2: score += 6   # Good
    elif current_ratio > 1.0: score += 3   # Adequate
    
    # Leverage scoring (10% weight) - lower is better
    if debt_equity < 0.3: score += 10      # Very Conservative
    elif debt_equity < 0.5: score += 8     # Conservative
    elif debt_equity < 0.8: score += 6     # Moderate
    elif debt_equity < 1.2: score += 4     # Higher Risk
    elif debt_equity < 1.5: score += 2     # High Risk
    
    # Generate recommendation based on score
    if score >= 80:
        return "STRONG BUY", "🟢", f"Outstanding financial performance (Score: {score}/100)"
    elif score >= 65:
        return "BUY", "🟡", f"Strong financial performance (Score: {score}/100)"  
    elif score >= 50:
        return "HOLD", "🟠", f"Good performance, monitor trends (Score: {score}/100)"
    elif score >= 35:
        return "WEAK HOLD", "🟠", f"Below average performance (Score: {score}/100)"
    else:
        return "SELL", "🔴", f"Poor financial performance (Score: {score}/100)"

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Company Analysis", "Ratio Prediction", "Financial Health Check", "Company Comparison"]
)

if page == "Company Analysis":
    st.header("📊 Individual Company Analysis")
    st.markdown("*Analyze real financial performance using historical data*")
    
    # Company selection
    available_companies = sorted(df['Company'].unique())
    company = st.selectbox("Select Company:", available_companies)
    
    # Get available periods for selected company
    years, quarters_by_year, annual_by_year = get_available_periods(company)
    
    if not years:
        st.error(f"❌ No data available for {company}")
    else:
        # Period selection
        col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
        
        with col1:
            st.write("**Period Type:**")
            period_type = st.radio("", ["Quarterly", "Annual"], horizontal=True)
        
        with col2:
            available_years = []
            if period_type == "Annual":
                available_years = [year for year in years if annual_by_year.get(year, False)]
            else:
                available_years = [year for year in years if quarters_by_year.get(year, [])]
            
            if available_years:
                year = st.selectbox("Year:", available_years)
            else:
                st.error(f"No {period_type.lower()} data available")
                st.stop()
        
        with col3:
            if period_type == "Quarterly":
                available_quarters = quarters_by_year.get(year, [])
                if available_quarters:
                    quarter = st.selectbox("Quarter:", available_quarters)
                else:
                    st.error(f"No quarterly data for {year}")
                    st.stop()
            else:
                quarter = None
                st.markdown("**Annual Analysis**")
        
        with col4:
            if period_type == "Annual":
                st.markdown("**📅 Full Year**")
            else:
                st.markdown(f"**📅 Q{quarter} {year}**")
        
        # Get real data for selected period
        real_data = get_company_data(company, year, quarter, period_type)
        
        if real_data:
            # Display current financial performance
            st.subheader(f"📈 {company} Financial Performance")
            period_str = f"{year} Q{quarter}" if period_type == "Quarterly" else f"{year} Annual"
            st.markdown(f"**Period:** {period_str}")
            
            # Show real financial ratios
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 💰 Profitability Ratios")
                
                # Display actual ratios from CSV
                if 'Gross Margin' in real_data and pd.notna(real_data['Gross Margin']):
                    st.metric("Gross Margin", f"{real_data['Gross Margin']:.1%}")
                
                if 'Net Profit Margin' in real_data and pd.notna(real_data['Net Profit Margin']):
                    st.metric("Net Profit Margin", f"{real_data['Net Profit Margin']:.1%}")
                
                if 'ROA' in real_data and pd.notna(real_data['ROA']):
                    st.metric("Return on Assets (ROA)", f"{real_data['ROA']:.1%}")
                
                if 'ROE' in real_data and pd.notna(real_data['ROE']):
                    st.metric("Return on Equity (ROE)", f"{real_data['ROE']:.1%}")
                
            with col2:
                st.markdown("#### ⚖️ Financial Health Ratios")
                
                if 'Current Ratio' in real_data and pd.notna(real_data['Current Ratio']):
                    st.metric("Current Ratio", f"{real_data['Current Ratio']:.2f}")
                
                if 'Debt-to-Equity' in real_data and pd.notna(real_data['Debt-to-Equity']):
                    st.metric("Debt-to-Equity", f"{real_data['Debt-to-Equity']:.2f}")
                
                if 'Debt-to-Assets' in real_data and pd.notna(real_data['Debt-to-Assets']):
                    st.metric("Debt-to-Assets", f"{real_data['Debt-to-Assets']:.1%}")
            
            # Generate analysis and recommendation
            if st.button("🔍 Generate Investment Analysis", type="primary"):
                recommendation, color, explanation = generate_investment_recommendation(real_data)
                
                st.markdown("---")
                st.subheader("🎯 Investment Analysis Results")
                
                # Display recommendation with appropriate styling
                if recommendation == "STRONG BUY":
                    st.success(f"💰 **Investment Recommendation: {recommendation}**")
                elif recommendation == "BUY":
                    st.success(f"📈 **Investment Recommendation: {recommendation}**")
                elif recommendation in ["HOLD", "WEAK HOLD"]:
                    st.warning(f"⚖️ **Investment Recommendation: {recommendation}**")
                else:
                    st.error(f"📉 **Investment Recommendation: {recommendation}**")
                
                st.write(explanation)
                
                # Detailed analysis
                col_a, col_b = st.columns(2)
                
                with col_a:
                    st.markdown("#### 📊 Performance Highlights")
                    
                    # ROE analysis
                    roe_val = real_data.get('ROE', 0)
                    if roe_val > 0.15:
                        st.success(f"✅ Excellent ROE: {roe_val:.1%}")
                    elif roe_val > 0.08:
                        st.info(f"👍 Good ROE: {roe_val:.1%}")
                    else:
                        st.warning(f"⚠️ ROE needs improvement: {roe_val:.1%}")
                    
                    # Liquidity analysis
                    cr_val = real_data.get('Current Ratio', 0)
                    if cr_val > 1.5:
                        st.success(f"✅ Strong liquidity: {cr_val:.2f}")
                    elif cr_val > 1.0:
                        st.info(f"👍 Adequate liquidity: {cr_val:.2f}")
                    else:
                        st.warning(f"⚠️ Liquidity concerns: {cr_val:.2f}")
                
                with col_b:
                    st.markdown("#### 🚨 Risk Assessment")
                    
                    # Leverage analysis
                    de_val = real_data.get('Debt-to-Equity', 0)
                    if de_val < 0.5:
                        st.success(f"✅ Conservative debt: {de_val:.2f}")
                    elif de_val < 1.0:
                        st.info(f"👍 Moderate debt: {de_val:.2f}")
                    else:
                        st.warning(f"⚠️ High leverage: {de_val:.2f}")
                    
                    # Industry comparison
                    company_data = df[df['Company'] == company]
                    if len(company_data) > 1:
                        avg_roe = company_data['ROE'].mean()
                        current_roe = real_data.get('ROE', 0)
                        if current_roe > avg_roe:
                            st.success(f"✅ Above company average ROE")
                        else:
                            st.info(f"📊 Below company average ROE")
        
        else:
            st.error(f"❌ No data available for {company} in {year} {f'Q{quarter}' if quarter else 'Annual'}")

elif page == "Ratio Prediction":
    st.header("🔮 Financial Ratio Prediction")
    st.markdown("*Predict future financial ratios using real historical trends*")
    
    # Company selection for prediction
    available_companies = sorted(df['Company'].unique())
    company = st.selectbox("Company:", available_companies)
    
    # Get latest available data
    company_data = df[df['Company'] == company]
    annual_data = company_data[company_data['Period_Type'] == 'Annual'].sort_values('Year')
    
    if annual_data.empty:
        st.error(f"❌ No annual data available for {company}")
    else:
        latest_data = annual_data.iloc[-1]
        
        # Prediction options
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Prediction Settings")
            prediction_year = st.selectbox("Predict for Year:", [2024, 2025, 2026])
            prediction_type = st.radio("Prediction Type:", ["Conservative", "Trend-based", "Optimistic"])
        
        with col2:
            st.subheader("📈 Latest Available Data")
            st.write(f"**Base Year:** {int(latest_data['Year'])}")
            
            if 'ROE' in latest_data and pd.notna(latest_data['ROE']):
                st.metric("Current ROE", f"{latest_data['ROE']:.1%}")
            if 'ROA' in latest_data and pd.notna(latest_data['ROA']):
                st.metric("Current ROA", f"{latest_data['ROA']:.1%}")
        
        if st.button("🎯 Generate Predictions", type="primary"):
            predictions = predict_future_ratios(company, int(latest_data['Year']))
            
            if predictions:
                st.markdown("---")
                st.subheader(f"🔮 {company} - {prediction_year} Predictions")
                
                # Display predictions in organized format
                pred_col1, pred_col2, pred_col3 = st.columns(3)
                
                with pred_col1:
                    st.markdown("#### 💰 **Profitability Predictions**")
                    for ratio in ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin']:
                        if ratio in predictions:
                            pred = predictions[ratio]
                            current_val = pred['current_value']
                            predicted_val = pred['predicted_value']
                            change = ((predicted_val - current_val) / current_val) * 100 if current_val != 0 else 0
                            
                            st.metric(
                                f"**{ratio}**",
                                f"{predicted_val:.1%}",
                                f"{change:+.1f}%",
                                help=f"Confidence: {pred['confidence']} | Current: {current_val:.1%}"
                            )
                
                with pred_col2:
                    st.markdown("#### ⚖️ **Financial Health Predictions**")
                    for ratio in ['Current Ratio', 'Debt-to-Equity']:
                        if ratio in predictions:
                            pred = predictions[ratio]
                            current_val = pred['current_value']
                            predicted_val = pred['predicted_value']
                            change = ((predicted_val - current_val) / current_val) * 100 if current_val != 0 else 0
                            
                            st.metric(
                                f"**{ratio}**",
                                f"{predicted_val:.2f}",
                                f"{change:+.1f}%",
                                help=f"Confidence: {pred['confidence']} | Current: {current_val:.2f}"
                            )
                
                with pred_col3:
                    st.markdown("#### 📊 **Prediction Summary**")
                    
                    # Calculate overall trend
                    positive_trends = sum(1 for p in predictions.values() if p['historical_trend'] > 0)
                    total_trends = len(predictions)
                    
                    if positive_trends / total_trends > 0.6:
                        st.success("📈 **Overall Trend: Positive**")
                        trend_message = "Most financial metrics showing improvement"
                    elif positive_trends / total_trends > 0.4:
                        st.info("📊 **Overall Trend: Mixed**") 
                        trend_message = "Mixed performance across metrics"
                    else:
                        st.warning("📉 **Overall Trend: Declining**")
                        trend_message = "Most metrics showing decline"
                    
                    st.write(trend_message)
                    
                    # Investment outlook based on predictions
                    if 'ROE' in predictions:
                        predicted_roe = predictions['ROE']['predicted_value']
                        if predicted_roe > 0.15:
                            st.success("💰 **Outlook: Strong Buy**")
                        elif predicted_roe > 0.10:
                            st.info("📈 **Outlook: Buy**")
                        else:
                            st.warning("⚖️ **Outlook: Hold**")
            else:
                st.warning("⚠️ Insufficient data for reliable predictions")

# Continue with other pages...
elif page == "Financial Health Check":
    st.header("🏥 Financial Health Assessment")
    st.markdown("*Comprehensive health analysis using real financial data*")
    
    available_companies = sorted(df['Company'].unique())
    company = st.selectbox("Select Company for Assessment:", available_companies)
    
    # Get latest available data
    company_data = df[df['Company'] == company]
    annual_data = company_data[company_data['Period_Type'] == 'Annual'].sort_values('Year')
    
    if annual_data.empty:
        st.error(f"❌ No annual data available for {company}")
    else:
        latest_data = annual_data.iloc[-1]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader(f"📊 {company} - Latest Financial Health")
            st.write(f"**Assessment Period:** {int(latest_data['Year'])} Annual")
            
            # Health scoring
            recommendation, color, explanation = generate_investment_recommendation(latest_data)
            
            # Extract score from explanation
            try:
                score = int(explanation.split("Score: ")[1].split("/")[0])
            except:
                score = 50  # Default score if parsing fails
            
            st.metric("Health Score", f"{score}/100")
            st.progress(score / 100)
            
            # Health status
            if score >= 80:
                st.success("🌟 **EXCELLENT** - Outstanding financial health!")
            elif score >= 65:
                st.success("👍 **GOOD** - Strong financial position")
            elif score >= 50:
                st.info("⚖️ **AVERAGE** - Moderate financial health")
            elif score >= 35:
                st.warning("⚠️ **BELOW AVERAGE** - Areas for improvement")
            else:
                st.error("🚨 **POOR** - Significant concerns")
        
        with col2:
            st.subheader("📈 Health Breakdown")
            
            # Show individual metrics that contribute to the score
            metrics_data = {
                "ROE": latest_data.get('ROE', 0),
                "ROA": latest_data.get('ROA', 0), 
                "Net Profit Margin": latest_data.get('Net Profit Margin', 0),
                "Current Ratio": latest_data.get('Current Ratio', 0),
                "Debt-to-Equity": latest_data.get('Debt-to-Equity', 0)
            }
            
            for metric, value in metrics_data.items():
                if pd.notna(value):
                    if metric in ['ROE', 'ROA', 'Net Profit Margin']:
                        st.metric(metric, f"{value:.1%}")
                    else:
                        st.metric(metric, f"{value:.2f}")

elif page == "Company Comparison":
    st.header("⚖️ Company Comparison Tool")
    st.markdown("*Compare financial performance across companies for specific periods*")
    
    # Get available data for comparison settings
    available_companies = sorted(df['Company'].unique())
    available_years = sorted(df['Year'].unique())
    
    # Comparison settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        comparison_type = st.selectbox(
            "Comparison Type:",
            ["General Financial Comparison", "Profitability Focus", "Growth Analysis", "Risk Assessment"]
        )
    
    with col2:
        comparison_year = st.selectbox("Year:", available_years)
    
    with col3:
        # Check what period types are available for the selected year
        year_data = df[df['Year'] == comparison_year]
        available_periods = []
        
        if not year_data[year_data['Period_Type'] == 'Annual'].empty:
            available_periods.append("Annual")
        
        quarterly_data = year_data[year_data['Period_Type'] == 'Quarterly']
        for q in sorted(quarterly_data['Quarter'].unique()):
            if q > 0:
                available_periods.append(f"Q{int(q)}")
        
        if available_periods:
            comparison_period = st.selectbox("Period:", available_periods)
        else:
            st.error(f"No data available for {comparison_year}")
            st.stop()
    
    # Filter data based on selection
    if comparison_period == "Annual":
        comparison_data = df[(df['Year'] == comparison_year) & (df['Period_Type'] == 'Annual')]
    else:
        quarter_num = int(comparison_period[1])
        comparison_data = df[(df['Year'] == comparison_year) & (df['Quarter'] == quarter_num)]
    
    if not comparison_data.empty:
        # Display comparison table
        st.subheader(f"📊 {comparison_type} - {comparison_year} {comparison_period}")
        
        # Select metrics based on comparison type
        if comparison_type == "General Financial Comparison":
            metrics = ['ROE', 'ROA', 'Net Profit Margin', 'Current Ratio', 'Debt-to-Equity']
        elif comparison_type == "Profitability Focus":
            metrics = ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin']
        elif comparison_type == "Growth Analysis":
            metrics = ['ROE', 'ROA', 'Net Profit Margin']
        else:  # Risk Assessment
            metrics = ['Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
        
        # Create comparison dataframe with available metrics
        available_metrics = [m for m in metrics if m in comparison_data.columns]
        if available_metrics:
            display_data = comparison_data[['Company'] + available_metrics].copy()
            
            # Format percentages for display
            for metric in available_metrics:
                if metric in ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin', 'Debt-to-Assets']:
                    display_data[metric] = display_data[metric].apply(lambda x: f"{x:.1%}" if pd.notna(x) else "N/A")
                else:
                    display_data[metric] = display_data[metric].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
            
            # Display with highlighting
            st.dataframe(display_data, use_container_width=True)
            
            # Create comparison charts if we have enough metrics
            if len(available_metrics) >= 2:
                col1, col2 = st.columns(2)
                
                with col1:
                    # First metric chart
                    metric1 = available_metrics[0]
                    chart_data = comparison_data[comparison_data[metric1].notna()]
                    
                    if not chart_data.empty:
                        values1 = chart_data[metric1].values
                        companies = chart_data['Company'].values
                        
                        fig1 = px.bar(
                            x=companies, 
                            y=values1,
                            title=f"{metric1} Comparison - {comparison_year} {comparison_period}",
                            labels={'x': 'Company', 'y': metric1},
                            color=values1,
                            color_continuous_scale='viridis'
                        )
                        fig1.update_layout(showlegend=False)
                        st.plotly_chart(fig1, use_container_width=True)
                
                with col2:
                    # Second metric chart
                    metric2 = available_metrics[1]
                    chart_data = comparison_data[comparison_data[metric2].notna()]
                    
                    if not chart_data.empty:
                        values2 = chart_data[metric2].values
                        companies = chart_data['Company'].values
                        
                        fig2 = px.bar(
                            x=companies, 
                            y=values2,
                            title=f"{metric2} Comparison - {comparison_year} {comparison_period}",
                            labels={'x': 'Company', 'y': metric2},
                            color=values2,
                            color_continuous_scale='plasma'
                        )
                        fig2.update_layout(showlegend=False)
                        st.plotly_chart(fig2, use_container_width=True)
            
            # Performance leaders
            st.subheader("🏆 Performance Leaders")
            
            # Create columns for each available metric
            if available_metrics:
                leader_cols = st.columns(min(len(available_metrics), 5))  # Max 5 columns
                
                for i, metric in enumerate(available_metrics[:5]):  # Show max 5 metrics
                    with leader_cols[i]:
                        metric_data = comparison_data[comparison_data[metric].notna()]
                        
                        if not metric_data.empty:
                            # For debt ratios, lower is better
                            if metric in ['Debt-to-Equity', 'Debt-to-Assets']:
                                best_idx = metric_data[metric].idxmin()
                                icon = "🥇 Lowest"
                            else:
                                best_idx = metric_data[metric].idxmax()
                                icon = "🥇 Best"
                            
                            best_company = metric_data.loc[best_idx, 'Company']
                            best_value = metric_data.loc[best_idx, metric]
                            
                            if metric in ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin', 'Debt-to-Assets']:
                                value_str = f"{best_value:.1%}"
                            else:
                                value_str = f"{best_value:.2f}"
                            
                            st.metric(f"{icon} {metric}", best_company, value_str)
            
            # Comparative analysis insights
            st.subheader("📋 Comparative Analysis Insights")
            
            # Find overall best performer based on available data
            performance_scores = {}
            
            for company in comparison_data['Company']:
                company_row = comparison_data[comparison_data['Company'] == company].iloc[0]
                _, _, explanation = generate_investment_recommendation(company_row)
                
                try:
                    score = int(explanation.split("Score: ")[1].split("/")[0])
                except:
                    score = 50  # Default score if parsing fails
                
                performance_scores[company] = score
            
            if performance_scores:
                best_performer = max(performance_scores.keys(), key=lambda k: performance_scores[k])
                worst_performer = min(performance_scores.keys(), key=lambda k: performance_scores[k])
                
                col_insight1, col_insight2 = st.columns(2)
                
                with col_insight1:
                    st.success(f"🌟 **Top Performer:** {best_performer}")
                    st.write(f"Overall Score: {performance_scores[best_performer]}/100")
                    
                with col_insight2:
                    st.info(f"📈 **Growth Opportunity:** {worst_performer}")
                    st.write(f"Overall Score: {performance_scores[worst_performer]}/100")
        
        else:
            st.warning(f"⚠️ No comparable metrics available for {comparison_type}")
    
    else:
        st.error(f"❌ No data available for {comparison_year} {comparison_period}")

# Footer
st.markdown("---")
st.markdown("🤖 **Financial AI Assistant** | Saudi Food Sector Analysis | Powered by Real Financial Data")
st.markdown("*Using actual CSV data from your trained models for accurate financial analysis*")

# Display data info
if not df.empty:
    st.markdown(f"**Data Coverage:** {df['Year'].min()}-{df['Year'].max()} | **Companies:** {', '.join(sorted(df['Company'].unique()))} | **Records:** {len(df)}")

if models_loaded:
    st.markdown("**Status:** ✅ AI Models loaded | 📊 Real CSV data integrated")
else:
    st.markdown("**Status:** 📊 Real CSV data analysis | ⚙️ Enhanced predictions available")
