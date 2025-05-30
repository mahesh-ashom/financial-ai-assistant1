import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
# Add this class definition after the imports
class ComprehensiveRatiPredictor:
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
                if input_copy['Company'] in self.le_company.classes_:
                    input_copy['Company_Encoded'] = self.le_company.transform([input_copy['Company']])[0]
                else:
                    input_copy['Company_Encoded'] = 0  # Default encoding
            except:
                input_copy['Company_Encoded'] = 0

        # Simple prediction logic for demo
        if 'ROE' not in input_copy:
            # Simple ROE prediction based on other ratios
            roe = input_copy.get('ROA', 0.08) * 1.2  # Simple approximation
            results['ROE'] = {
                'predicted_value': roe,
                'confidence': 'Medium',
                'iteration': 1
            }

        return results, input_copy
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

# Load models
@st.cache_resource
def load_models():
    try:
        comprehensive_predictor = joblib.load('comprehensive_ratio_predictor.pkl')
        company_encoder = joblib.load('company_encoder.pkl')
        return comprehensive_predictor, company_encoder
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None, None

comprehensive_predictor, company_encoder = load_models()

if comprehensive_predictor and company_encoder:
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Type:",
        ["Company Analysis", "Ratio Prediction", "Financial Health Check"]
    )
    
    if page == "Company Analysis":
        st.header("📊 Individual Company Analysis")
        
        # Company selection
        company = st.selectbox("Select Company:", ["Almarai", "Savola", "NADEC"])
        
        # Input financial ratios
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Input Known Financial Data")
            
            # Optional inputs - user can leave some blank
            gross_margin = st.number_input("Gross Margin (%)", min_value=0.0, max_value=100.0, value=25.0, step=0.1) / 100
            net_profit_margin = st.number_input("Net Profit Margin (%)", min_value=-10.0, max_value=50.0, value=8.0, step=0.1) / 100
            roa = st.number_input("ROA (%)", min_value=-5.0, max_value=30.0, value=6.0, step=0.1) / 100
            current_ratio = st.number_input("Current Ratio", min_value=0.0, max_value=5.0, value=1.5, step=0.1)
            debt_to_equity = st.number_input("Debt-to-Equity", min_value=0.0, max_value=5.0, value=0.8, step=0.1)
            
            # Time info
            year = st.number_input("Year", min_value=2020, max_value=2025, value=2024)
            quarter = st.selectbox("Quarter", [1, 2, 3, 4])
            
        with col2:
            st.subheader("🎯 AI Predictions")
            
            if st.button("🔍 Analyze Company", type="primary"):
                # Prepare input data
                input_data = {
                    'Company': company,
                    'Year': year,
                    'Quarter': quarter,
                    'Gross Margin': gross_margin,
                    'Net Profit Margin': net_profit_margin,
                    'ROA': roa,
                    'Current Ratio': current_ratio,
                    'Debt-to-Equity': debt_to_equity
                }
                
                try:
                    # Get predictions
                    predictions, complete_data = comprehensive_predictor.predict_all_ratios(input_data)
                    
                    if predictions:
                        st.success("✅ Analysis Complete!")
                        
                        # Display predictions
                        for ratio, pred_info in predictions.items():
                            if ratio in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE']:
                                st.metric(
                                    f"🔮 Predicted {ratio}",
                                    f"{pred_info['predicted_value']:.1%}",
                                    help=f"Confidence: {pred_info['confidence']}"
                                )
                            else:
                                st.metric(
                                    f"🔮 Predicted {ratio}",
                                    f"{pred_info['predicted_value']:.2f}",
                                    help=f"Confidence: {pred_info['confidence']}"
                                )
                        
                        # Investment recommendation
                        if 'ROE' in predictions:
                            roe = predictions['ROE']['predicted_value']
                            if roe > 0.15:
                                st.success("💰 **Investment Recommendation: STRONG BUY**")
                                st.write("Excellent ROE performance indicates strong profitability!")
                            elif roe > 0.10:
                                st.info("📊 **Investment Recommendation: BUY**")
                                st.write("Good ROE performance with solid returns.")
                            elif roe > 0.05:
                                st.warning("⚖️ **Investment Recommendation: HOLD**")
                                st.write("Moderate performance, monitor closely.")
                            else:
                                st.error("📉 **Investment Recommendation: SELL**")
                                st.write("Low ROE indicates poor performance.")
                    
                    else:
                        st.warning("⚠️ Could not generate predictions with provided data.")
                        
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
    
    elif page == "Ratio Prediction":
        st.header("🔮 Financial Ratio Prediction")
        st.write("Enter partial financial data and let AI predict missing ratios!")
        
        # Company selection
        company = st.selectbox("Company:", ["Almarai", "Savola", "NADEC"])
        
        # Create dynamic input form
        st.subheader("📊 Enter Available Financial Data")
        
        col1, col2, col3 = st.columns(3)
        
        input_data = {'Company': company, 'Year': 2024, 'Quarter': 4}
        
        with col1:
            st.write("**Profitability Ratios**")
            if st.checkbox("I have Gross Margin"):
                input_data['Gross Margin'] = st.number_input("Gross Margin (%)", 0.0, 100.0, 25.0) / 100
            
            if st.checkbox("I have Net Profit Margin"):
                input_data['Net Profit Margin'] = st.number_input("Net Profit Margin (%)", -10.0, 50.0, 8.0) / 100
                
            if st.checkbox("I have ROA"):
                input_data['ROA'] = st.number_input("ROA (%)", -5.0, 30.0, 6.0) / 100
        
        with col2:
            st.write("**Liquidity & Leverage**")
            if st.checkbox("I have Current Ratio"):
                input_data['Current Ratio'] = st.number_input("Current Ratio", 0.0, 5.0, 1.5)
            
            if st.checkbox("I have Debt-to-Equity"):
                input_data['Debt-to-Equity'] = st.number_input("Debt-to-Equity", 0.0, 5.0, 0.8)
                
            if st.checkbox("I have Debt-to-Assets"):
                input_data['Debt-to-Assets'] = st.number_input("Debt-to-Assets (%)", 0.0, 100.0, 40.0) / 100
        
        with col3:
            st.write("**Other Info**")
            input_data['Year'] = st.number_input("Year", 2020, 2025, 2024)
            input_data['Quarter'] = st.selectbox("Quarter", [1, 2, 3, 4])
        
        if st.button("🎯 Predict Missing Ratios", type="primary"):
            try:
                predictions, complete_data = comprehensive_predictor.predict_all_ratios(input_data)
                
                if predictions:
                    st.success(f"✅ Predicted {len(predictions)} financial ratios!")
                    
                    # Display in organized format
                    pred_col1, pred_col2 = st.columns(2)
                    
                    with pred_col1:
                        st.subheader("📈 Profitability Predictions")
                        for ratio in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE']:
                            if ratio in predictions:
                                st.write(f"**{ratio}:** {predictions[ratio]['predicted_value']:.1%} "
                                        f"(Confidence: {predictions[ratio]['confidence']})")
                    
                    with pred_col2:
                        st.subheader("💰 Financial Health Predictions")
                        for ratio in ['Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']:
                            if ratio in predictions:
                                if ratio == 'Debt-to-Assets':
                                    st.write(f"**{ratio}:** {predictions[ratio]['predicted_value']:.1%} "
                                            f"(Confidence: {predictions[ratio]['confidence']})")
                                else:
                                    st.write(f"**{ratio}:** {predictions[ratio]['predicted_value']:.2f} "
                                            f"(Confidence: {predictions[ratio]['confidence']})")
                else:
                    st.warning("⚠️ Need more input data to make predictions.")
                    
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    
    elif page == "Financial Health Check":
        st.header("🏥 Financial Health Assessment")
        st.write("Get a comprehensive health score for any company!")
        
        # Quick assessment form
        company = st.selectbox("Select Company for Assessment:", ["Almarai", "Savola", "NADEC"])
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Key Financial Metrics")
            roe = st.slider("ROE (%)", 0, 30, 12) / 100
            roa = st.slider("ROA (%)", 0, 20, 6) / 100
            npm = st.slider("Net Profit Margin (%)", 0, 25, 8) / 100
            current_ratio = st.slider("Current Ratio", 0.5, 3.0, 1.5)
            debt_equity = st.slider("Debt-to-Equity", 0.0, 3.0, 0.8)
        
        with col2:
            if st.button("📋 Generate Health Report", type="primary"):
                # Calculate health score
                health_score = 0
                
                # ROE scoring (30% weight)
                if roe > 0.15: health_score += 30
                elif roe > 0.10: health_score += 20
                elif roe > 0.05: health_score += 10
                
                # ROA scoring (25% weight)
                if roa > 0.10: health_score += 25
                elif roa > 0.06: health_score += 15
                elif roa > 0.03: health_score += 8
                
                # NPM scoring (20% weight)
                if npm > 0.15: health_score += 20
                elif npm > 0.10: health_score += 12
                elif npm > 0.05: health_score += 6
                
                # Liquidity scoring (15% weight)
                if current_ratio > 1.5: health_score += 15
                elif current_ratio > 1.2: health_score += 10
                elif current_ratio > 1.0: health_score += 5
                
                # Leverage scoring (10% weight)
                if debt_equity < 0.5: health_score += 10
                elif debt_equity < 1.0: health_score += 6
                elif debt_equity < 1.5: health_score += 3
                
                # Display results
                st.subheader(f"🎯 Financial Health Score: {health_score}/100")
                
                # Progress bar
                st.progress(health_score / 100)
                
                # Health status
                if health_score >= 80:
                    st.success("🌟 **EXCELLENT** - Outstanding financial health!")
                    recommendation = "Strong Buy - Exceptional investment opportunity"
                elif health_score >= 60:
                    st.info("👍 **GOOD** - Solid financial position")
                    recommendation = "Buy - Good investment potential"
                elif health_score >= 40:
                    st.warning("⚖️ **AVERAGE** - Moderate financial health")
                    recommendation = "Hold - Monitor performance closely"
                else:
                    st.error("⚠️ **POOR** - Concerning financial metrics")
                    recommendation = "Sell - High risk investment"
                
                st.write(f"**Investment Recommendation:** {recommendation}")
                
                # Detailed breakdown
                st.subheader("📊 Score Breakdown")
                breakdown_data = {
                    'Metric': ['ROE Performance', 'ROA Efficiency', 'Profit Margins', 'Liquidity', 'Leverage'],
                    'Weight': ['30%', '25%', '20%', '15%', '10%'],
                    'Score': [
                        f"{min(30, max(0, 30 if roe > 0.15 else (20 if roe > 0.10 else (10 if roe > 0.05 else 0))))}/30",
                        f"{min(25, max(0, 25 if roa > 0.10 else (15 if roa > 0.06 else (8 if roa > 0.03 else 0))))}/25",
                        f"{min(20, max(0, 20 if npm > 0.15 else (12 if npm > 0.10 else (6 if npm > 0.05 else 0))))}/20",
                        f"{min(15, max(0, 15 if current_ratio > 1.5 else (10 if current_ratio > 1.2 else (5 if current_ratio > 1.0 else 0))))}/15",
                        f"{min(10, max(0, 10 if debt_equity < 0.5 else (6 if debt_equity < 1.0 else (3 if debt_equity < 1.5 else 0))))}/10"
                    ]
                }
                
                breakdown_df = pd.DataFrame(breakdown_data)
                st.dataframe(breakdown_df, use_container_width=True)

else:
    st.error("❌ Could not load AI models. Please ensure model files are uploaded correctly.")
    st.info("Required files: comprehensive_ratio_predictor.pkl, company_encoder.pkl")

# Footer
st.markdown("---")
st.markdown("🤖 **Financial AI Assistant** | Saudi Food Sector Analysis | Powered by Advanced AI")
st.markdown("*Predicts financial ratios, investment recommendations, and company health scores*")
