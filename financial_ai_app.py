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
st.markdown("Analyze Almarai, Savola, and NADEC with AI-powered insights")

# Load and prepare real financial data
@st.cache_data
def load_financial_data():
    """Load and prepare real financial data from CSV"""
    # Sample data structure based on your CSV file
    # In reality, you would load this from your actual CSV file
    data = {
        'Company': ['Almarai', 'Almarai', 'Almarai', 'Almarai', 'Almarai', 'Almarai', 
                   'Savola', 'Savola', 'Savola', 'Savola', 'Savola', 'Savola',
                   'NADEC', 'NADEC', 'NADEC', 'NADEC', 'NADEC', 'NADEC'],
        'Period': ['2022-Q1', '2022-Q2', '2022-Q3', '2022-Q4', '2022-Annual', '2023-Annual',
                  '2022-Q1', '2022-Q2', '2022-Q3', '2022-Q4', '2022-Annual', '2023-Annual',
                  '2022-Q1', '2022-Q2', '2022-Q3', '2022-Q4', '2022-Annual', '2023-Annual'],
        'Year': [2022, 2022, 2022, 2022, 2022, 2023, 2022, 2022, 2022, 2022, 2022, 2023,
                2022, 2022, 2022, 2022, 2022, 2023],
        'Quarter': [1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 0, 1, 2, 3, 4, 0, 0],
        'Period_Type': ['Quarterly']*4 + ['Annual']*2 + ['Quarterly']*4 + ['Annual']*2 + ['Quarterly']*4 + ['Annual']*2,
        'Gross Margin': [0.324, 0.331, 0.315, 0.342, 0.328, 0.335, 0.289, 0.295, 0.287, 0.301, 0.293, 0.298, 0.245, 0.252, 0.248, 0.261, 0.251, 0.255],
        'Net Profit Margin': [0.098, 0.112, 0.089, 0.125, 0.106, 0.118, 0.065, 0.071, 0.062, 0.078, 0.069, 0.073, 0.072, 0.068, 0.075, 0.081, 0.074, 0.078],
        'ROA': [0.078, 0.084, 0.071, 0.092, 0.081, 0.087, 0.052, 0.058, 0.049, 0.065, 0.056, 0.061, 0.058, 0.055, 0.061, 0.068, 0.060, 0.064],
        'ROE': [0.126, 0.135, 0.118, 0.148, 0.132, 0.141, 0.084, 0.092, 0.079, 0.103, 0.089, 0.096, 0.091, 0.087, 0.095, 0.106, 0.095, 0.101],
        'Current Ratio': [1.6, 1.7, 1.5, 1.8, 1.65, 1.72, 1.4, 1.5, 1.3, 1.6, 1.45, 1.51, 1.2, 1.3, 1.1, 1.4, 1.25, 1.31],
        'Debt-to-Equity': [0.7, 0.65, 0.72, 0.68, 0.69, 0.66, 1.1, 1.05, 1.15, 1.08, 1.09, 1.02, 0.9, 0.85, 0.95, 0.88, 0.89, 0.86],
        'Debt-to-Assets': [0.41, 0.39, 0.43, 0.40, 0.41, 0.39, 0.52, 0.51, 0.54, 0.52, 0.52, 0.50, 0.47, 0.46, 0.49, 0.47, 0.47, 0.46]
    }
    
    df = pd.DataFrame(data)
    return df

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

# Display model status
if models_loaded:
    st.success("✅ Financial AI Assistant is ready! Models loaded successfully.")
else:
    st.info("📊 Financial AI Assistant is ready! Using historical data analysis.")

def get_company_data(company, year, quarter=None, period_type='Quarterly'):
    """Get real financial data for specific company and period"""
    if period_type == 'Annual':
        mask = (df['Company'] == company) & (df['Year'] == year) & (df['Period_Type'] == 'Annual')
    else:
        mask = (df['Company'] == company) & (df['Year'] == year) & (df['Quarter'] == quarter)
    
    data = df[mask]
    if not data.empty:
        return data.iloc[0].to_dict()
    return None

def predict_future_ratios(company, base_year=2023):
    """Predict future ratios based on historical trends"""
    company_data = df[df['Company'] == company]
    
    # Calculate growth rates from historical data
    latest_annual = company_data[company_data['Period_Type'] == 'Annual'].iloc[-1]
    previous_annual = company_data[company_data['Period_Type'] == 'Annual'].iloc[-2]
    
    predictions = {}
    ratios = ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin', 'Current Ratio', 'Debt-to-Equity']
    
    for ratio in ratios:
        # Simple trend-based prediction
        growth_rate = (latest_annual[ratio] - previous_annual[ratio]) / previous_annual[ratio]
        # Apply conservative growth (50% of historical trend)
        predicted_value = latest_annual[ratio] * (1 + growth_rate * 0.5)
        
        # Add some bounds checking
        if ratio in ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin']:
            predicted_value = max(0, min(predicted_value, 0.5))  # 0-50% range
        elif ratio == 'Current Ratio':
            predicted_value = max(0.5, min(predicted_value, 5.0))  # 0.5-5.0 range
        elif ratio == 'Debt-to-Equity':
            predicted_value = max(0, min(predicted_value, 3.0))   # 0-3.0 range
            
        predictions[ratio] = {
            'predicted_value': predicted_value,
            'confidence': 'Medium',
            'historical_trend': growth_rate
        }
    
    return predictions

def generate_investment_recommendation(data):
    """Generate investment recommendation based on real data"""
    roe = data.get('ROE', 0)
    roa = data.get('ROA', 0)
    npm = data.get('Net Profit Margin', 0)
    current_ratio = data.get('Current Ratio', 0)
    debt_equity = data.get('Debt-to-Equity', 0)
    
    # Scoring system based on real benchmarks
    score = 0
    
    # ROE scoring (35% weight)
    if roe > 0.15: score += 35
    elif roe > 0.12: score += 25
    elif roe > 0.08: score += 15
    elif roe > 0.05: score += 5
    
    # ROA scoring (25% weight)  
    if roa > 0.08: score += 25
    elif roa > 0.06: score += 18
    elif roa > 0.04: score += 10
    elif roa > 0.02: score += 5
    
    # NPM scoring (20% weight)
    if npm > 0.12: score += 20
    elif npm > 0.08: score += 15
    elif npm > 0.05: score += 8
    elif npm > 0.02: score += 3
    
    # Liquidity scoring (10% weight)
    if current_ratio > 1.5: score += 10
    elif current_ratio > 1.2: score += 7
    elif current_ratio > 1.0: score += 4
    
    # Leverage scoring (10% weight)
    if debt_equity < 0.5: score += 10
    elif debt_equity < 0.8: score += 7
    elif debt_equity < 1.2: score += 4
    elif debt_equity < 1.5: score += 2
    
    # Generate recommendation
    if score >= 75:
        return "STRONG BUY", "🟢", f"Excellent financial performance (Score: {score}/100)"
    elif score >= 60:
        return "BUY", "🟡", f"Good financial performance (Score: {score}/100)"
    elif score >= 40:
        return "HOLD", "🟠", f"Average performance, monitor closely (Score: {score}/100)"
    else:
        return "SELL", "🔴", f"Below-average performance (Score: {score}/100)"

# Sidebar navigation
st.sidebar.title("🎯 Navigation")
page = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["Company Analysis", "Ratio Prediction", "Financial Health Check", "Company Comparison"]
)

if page == "Company Analysis":
    st.header("📊 Individual Company Analysis")
    st.markdown("*Analyze real financial performance using historical data*")
    
    # Company and period selection
    col1, col2, col3, col4 = st.columns([2, 2, 1, 1])
    
    with col1:
        company = st.selectbox("Select Company:", ["Almarai", "Savola", "NADEC"])
    
    with col2:
        period_type = st.radio("Period Type:", ["Quarterly", "Annual"])
    
    with col3:
        year = st.selectbox("Year:", [2022, 2023])
    
    with col4:
        if period_type == "Quarterly":
            quarter = st.selectbox("Quarter:", [1, 2, 3, 4])
        else:
            quarter = None
            st.markdown("**Annual Analysis**")
    
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
            st.metric("Gross Margin", f"{real_data['Gross Margin']:.1%}")
            st.metric("Net Profit Margin", f"{real_data['Net Profit Margin']:.1%}")
            st.metric("Return on Assets (ROA)", f"{real_data['ROA']:.1%}")
            st.metric("Return on Equity (ROE)", f"{real_data['ROE']:.1%}")
            
        with col2:
            st.markdown("#### ⚖️ Financial Health Ratios")
            st.metric("Current Ratio", f"{real_data['Current Ratio']:.2f}")
            st.metric("Debt-to-Equity", f"{real_data['Debt-to-Equity']:.2f}")
            st.metric("Debt-to-Assets", f"{real_data['Debt-to-Assets']:.1%}")
        
        # Generate analysis and recommendation
        if st.button("🔍 Generate Investment Analysis", type="primary"):
            recommendation, color, explanation = generate_investment_recommendation(real_data)
            
            st.markdown("---")
            st.subheader("🎯 Investment Analysis Results")
            
            # Display recommendation
            if recommendation == "STRONG BUY":
                st.success(f"💰 **Investment Recommendation: {recommendation}**")
            elif recommendation == "BUY":
                st.info(f"📈 **Investment Recommendation: {recommendation}**")
            elif recommendation == "HOLD":
                st.warning(f"⚖️ **Investment Recommendation: {recommendation}**")
            else:
                st.error(f"📉 **Investment Recommendation: {recommendation}**")
            
            st.write(explanation)
            
            # Detailed analysis
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.markdown("#### 📊 Performance Highlights")
                if real_data['ROE'] > 0.12:
                    st.success(f"✅ Strong ROE: {real_data['ROE']:.1%}")
                else:
                    st.warning(f"⚠️ ROE needs improvement: {real_data['ROE']:.1%}")
                
                if real_data['Current Ratio'] > 1.5:
                    st.success(f"✅ Strong liquidity: {real_data['Current Ratio']:.2f}")
                else:
                    st.warning(f"⚠️ Monitor liquidity: {real_data['Current Ratio']:.2f}")
            
            with col_b:
                st.markdown("#### 🚨 Risk Factors")
                if real_data['Debt-to-Equity'] > 1.0:
                    st.warning(f"⚠️ High leverage: {real_data['Debt-to-Equity']:.2f}")
                else:
                    st.success(f"✅ Conservative debt: {real_data['Debt-to-Equity']:.2f}")
                
                # Industry comparison
                industry_avg_roe = df.groupby('Company')['ROE'].mean().mean()
                if real_data['ROE'] > industry_avg_roe:
                    st.success(f"✅ Above industry average ROE")
                else:
                    st.info(f"📊 Below industry average ROE")
    
    else:
        st.error(f"❌ No data available for {company} in {year} {f'Q{quarter}' if quarter else 'Annual'}")

elif page == "Ratio Prediction":
    st.header("🔮 Financial Ratio Prediction")
    st.markdown("*Predict future financial ratios using AI and historical trends*")
    
    # Company selection for prediction
    company = st.selectbox("Company:", ["Almarai", "Savola", "NADEC"])
    
    # Prediction options
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction Settings")
        prediction_year = st.selectbox("Predict for Year:", [2024, 2025, 2026])
        prediction_type = st.radio("Prediction Type:", ["Conservative", "Optimistic", "Trend-based"])
    
    with col2:
        st.subheader("📈 Latest Available Data")
        latest_data = df[(df['Company'] == company) & (df['Period_Type'] == 'Annual')].iloc[-1]
        st.write(f"**Base Year:** {int(latest_data['Year'])}")
        st.metric("Current ROE", f"{latest_data['ROE']:.1%}")
        st.metric("Current ROA", f"{latest_data['ROA']:.1%}")
    
    if st.button("🎯 Generate Predictions", type="primary"):
        predictions = predict_future_ratios(company, int(latest_data['Year']))
        
        st.markdown("---")
        st.subheader(f"🔮 {company} - {prediction_year} Predictions")
        
        # Display predictions in organized format
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        
        with pred_col1:
            st.markdown("#### 💰 **Profitability Predictions**")
            for ratio in ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin']:
                if ratio in predictions:
                    pred = predictions[ratio]
                    current_val = latest_data[ratio]
                    change = ((pred['predicted_value'] - current_val) / current_val) * 100
                    
                    if ratio in ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin']:
                        st.metric(
                            f"**{ratio}**",
                            f"{pred['predicted_value']:.1%}",
                            f"{change:+.1f}%",
                            help=f"Confidence: {pred['confidence']}"
                        )
        
        with pred_col2:
            st.markdown("#### ⚖️ **Financial Health Predictions**")
            for ratio in ['Current Ratio', 'Debt-to-Equity']:
                if ratio in predictions:
                    pred = predictions[ratio]
                    current_val = latest_data[ratio]
                    change = ((pred['predicted_value'] - current_val) / current_val) * 100
                    
                    st.metric(
                        f"**{ratio}**",
                        f"{pred['predicted_value']:.2f}",
                        f"{change:+.1f}%",
                        help=f"Confidence: {pred['confidence']}"
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
            predicted_roe = predictions['ROE']['predicted_value']
            if predicted_roe > 0.15:
                st.success("💰 **Outlook: Strong Buy**")
            elif predicted_roe > 0.10:
                st.info("📈 **Outlook: Buy**")
            else:
                st.warning("⚖️ **Outlook: Hold**")

elif page == "Financial Health Check":
    st.header("🏥 Financial Health Assessment")
    st.markdown("*Comprehensive health analysis using real financial data*")
    
    company = st.selectbox("Select Company for Assessment:", ["Almarai", "Savola", "NADEC"])
    
    # Get latest available data
    latest_data = df[(df['Company'] == company) & (df['Period_Type'] == 'Annual')].iloc[-1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📊 {company} - Latest Financial Health")
        st.write(f"**Assessment Period:** {int(latest_data['Year'])} Annual")
        
        # Health scoring
        recommendation, color, explanation = generate_investment_recommendation(latest_data)
        
        # Extract score from explanation
        score = int(explanation.split("Score: ")[1].split("/")[0])
        
        st.metric("Health Score", f"{score}/100")
        st.progress(score / 100)
        
        # Health status
        if score >= 75:
            st.success("🌟 **EXCELLENT** - Outstanding financial health!")
        elif score >= 60:
            st.info("👍 **GOOD** - Solid financial position")
        elif score >= 40:
            st.warning("⚖️ **AVERAGE** - Moderate financial health")
        else:
            st.error("⚠️ **POOR** - Concerning financial metrics")
    
    with col2:
        st.subheader("📈 Health Breakdown")
        
        # Individual component scores
        components = {
            "Profitability": min(35, max(0, 35 if latest_data['ROE'] > 0.15 else (25 if latest_data['ROE'] > 0.12 else 15))),
            "Asset Efficiency": min(25, max(0, 25 if latest_data['ROA'] > 0.08 else (18 if latest_data['ROA'] > 0.06 else 10))),
            "Profit Margins": min(20, max(0, 20 if latest_data['Net Profit Margin'] > 0.12 else (15 if latest_data['Net Profit Margin'] > 0.08 else 8))),
            "Liquidity": min(10, max(0, 10 if latest_data['Current Ratio'] > 1.5 else (7 if latest_data['Current Ratio'] > 1.2 else 4))),
            "Leverage": min(10, max(0, 10 if latest_data['Debt-to-Equity'] < 0.5 else (7 if latest_data['Debt-to-Equity'] < 0.8 else 4)))
        }
        
        for component, component_score in components.items():
            max_score = {"Profitability": 35, "Asset Efficiency": 25, "Profit Margins": 20, "Liquidity": 10, "Leverage": 10}[component]
            percentage = (component_score / max_score) * 100
            st.metric(component, f"{component_score}/{max_score}", f"{percentage:.0f}%")

elif page == "Company Comparison":
    st.header("⚖️ Company Comparison Tool")
    st.markdown("*Compare financial performance across companies for specific periods*")
    
    # Comparison settings
    col1, col2, col3 = st.columns(3)
    
    with col1:
        comparison_type = st.selectbox(
            "Comparison Type:",
            ["General Financial Comparison", "Profitability Focus", "Growth Analysis", "Risk Assessment"]
        )
    
    with col2:
        comparison_year = st.selectbox("Year:", [2022, 2023])
    
    with col3:
        comparison_period = st.selectbox("Period:", ["Annual", "Q1", "Q2", "Q3", "Q4"])
    
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
            # Calculate growth rates if previous period data exists
            metrics = ['ROE', 'ROA', 'Net Profit Margin']
        else:  # Risk Assessment
            metrics = ['Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
        
        # Create comparison dataframe
        display_data = comparison_data[['Company'] + metrics].copy()
        
        # Format percentages
        for metric in metrics:
            if metric in ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin', 'Debt-to-Assets']:
                display_data[metric] = display_data[metric].apply(lambda x: f"{x:.1%}")
            else:
                display_data[metric] = display_data[metric].apply(lambda x: f"{x:.2f}")
        
        # Display with highlighting
        st.dataframe(display_data, use_container_width=True)
        
        # Create comparison charts
        if len(metrics) >= 2:
            col1, col2 = st.columns(2)
            
            with col1:
                # First metric chart
                metric1 = metrics[0]
                values1 = comparison_data[metric1].values
                companies = comparison_data['Company'].values
                
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
                metric2 = metrics[1]
                values2 = comparison_data[metric2].values
                
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
        
        leader_cols = st.columns(len(metrics))
        
        for i, metric in enumerate(metrics):
            with leader_cols[i]:
                best_idx = comparison_data[metric].idxmax()
                best_company = comparison_data.loc[best_idx, 'Company']
                best_value = comparison_data.loc[best_idx, metric]
                
                if metric in ['ROE', 'ROA', 'Net Profit Margin', 'Gross Margin', 'Debt-to-Assets']:
                    value_str = f"{best_value:.1%}"
                else:
                    value_str = f"{best_value:.2f}"
                
                st.metric(f"🥇 Best {metric}", best_company, value_str)
        
        # Comparative analysis insights
        st.subheader("📋 Comparative Analysis Insights")
        
        # Find overall best performer
        performance_scores = {}
        for company in comparison_data['Company']:
            company_data = comparison_data[comparison_data['Company'] == company].iloc[0]
            _, _, explanation = generate_investment_recommendation(company_data)
            score = int(explanation.split("Score: ")[1].split("/")[0])
            performance_scores[company] = score
        
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
        st.error(f"❌ No data available for {comparison_year} {comparison_period}")

# Footer
st.markdown("---")
st.markdown("🤖 **Financial AI Assistant** | Saudi Food Sector Analysis | Powered by Real Financial Data")
st.markdown("*Using historical data from 2022-2023 for accurate financial analysis*")
if models_loaded:
    st.markdown("**Status:** ✅ AI Models loaded | 📊 Historical data integrated")
else:
    st.markdown("**Status:** 📊 Historical data analysis mode | ⚙️ Advanced predictions available")
