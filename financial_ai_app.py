# CREATE WEB APP CODE
print("🌐 Creating Web App Code for Deployment")
print("="*50)

# This code will create a Streamlit app file
streamlit_app_code = '''
import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load your trained models (you'll upload these files)
@st.cache_resource
def load_models():
    # You'll need to upload these .pkl files to Streamlit
    roe_model = joblib.load('roe_prediction_model.pkl')
    invest_model = joblib.load('investment_model.pkl')
    status_model = joblib.load('company_status_model.pkl')
    le_invest = joblib.load('investment_encoder.pkl')
    le_status = joblib.load('status_encoder.pkl')
    
    return roe_model, invest_model, status_model, le_invest, le_status

# Streamlit App
st.title("🤖 Financial AI Assistant")
st.markdown("Ask me about company financial analysis!")

# Load models
try:
    roe_model, invest_model, status_model, le_invest, le_status = load_models()
    
    # Sidebar for input
    st.sidebar.header("Company Analysis")
    
    # Input fields
    company_name = st.sidebar.text_input("Company Name", "Your Company")
    gross_margin = st.sidebar.slider("Gross Margin", 0.0, 1.0, 0.3, 0.01)
    net_profit_margin = st.sidebar.slider("Net Profit Margin", 0.0, 1.0, 0.1, 0.01)
    roa = st.sidebar.slider("ROA", 0.0, 1.0, 0.08, 0.01)
    current_ratio = st.sidebar.slider("Current Ratio", 0.0, 5.0, 1.5, 0.1)
    debt_to_equity = st.sidebar.slider("Debt-to-Equity", 0.0, 5.0, 0.8, 0.1)
    debt_to_assets = st.sidebar.slider("Debt-to-Assets", 0.0, 1.0, 0.45, 0.01)
    
    if st.sidebar.button("Analyze Company"):
        # Prepare input data
        input_data = pd.DataFrame({
            'Gross Margin': [gross_margin],
            'Net Profit Margin': [net_profit_margin], 
            'ROA': [roa],
            'ROE': [0],  # Will be predicted
            'Current Ratio': [current_ratio],
            'Debt-to-Equity': [debt_to_equity],
            'Debt-to-Assets': [debt_to_assets],
            'Year': [2024],
            'Quarter': [4],
            'Company_Encoded': [1]
        })
        
        # Make predictions
        roe_features = ['Gross Margin', 'Net Profit Margin', 'ROA', 'Current Ratio', 
                       'Debt-to-Equity', 'Debt-to-Assets', 'Year', 'Quarter', 'Company_Encoded']
        
        # Predict ROE
        predicted_roe = roe_model.predict(input_data[roe_features])[0]
        input_data['ROE'] = predicted_roe
        
        # Predict Investment
        invest_features = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 
                          'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets', 
                          'Year', 'Quarter', 'Company_Encoded']
        
        invest_pred = invest_model.predict(input_data[invest_features])[0]
        invest_recommendation = le_invest.inverse_transform([invest_pred])[0]
        invest_confidence = max(invest_model.predict_proba(input_data[invest_features])[0])
        
        # Predict Status
        status_pred = status_model.predict(input_data[invest_features])[0]
        company_status = le_status.inverse_transform([status_pred])[0]
        
        # Display results
        st.success(f"Analysis Complete for {company_name}!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Predicted ROE", f"{predicted_roe:.1%}")
            
        with col2:
            st.metric("Investment Recommendation", invest_recommendation)
            
        with col3:
            st.metric("Company Status", company_status)
            
        # Confidence
        st.info(f"AI Confidence: {invest_confidence:.0%}")
        
        # Business interpretation
        if invest_recommendation == "Buy":
            st.success("💰 Strong investment opportunity!")
        elif invest_recommendation == "Hold":
            st.warning("⚪ Maintain current position")
        else:
            st.error("❌ High risk - consider avoiding")
            
except FileNotFoundError:
    st.error("Please upload model files first!")
    st.info("Upload: roe_prediction_model.pkl, investment_model.pkl, company_status_model.pkl, investment_encoder.pkl, status_encoder.pkl")

# Chat interface
st.header("💬 Ask Questions")
question = st.text_input("Ask me about financial analysis:", "What should I invest in?")

if question:
    if "best investment" in question.lower():
        st.write("🤖 Based on my analysis, look for companies with:")
        st.write("• ROE > 15%")
        st.write("• Current Ratio > 1.5") 
        st.write("• Debt-to-Equity < 1.0")
    elif "risk" in question.lower():
        st.write("🤖 I identify risk based on:")
        st.write("• Low profitability ratios")
        st.write("• Poor liquidity (Current Ratio < 1.0)")
        st.write("• High debt levels")
    else:
        st.write("🤖 I can help you analyze any company! Just enter the financial ratios on the left.")
'''

# Save the Streamlit app code
with open('financial_ai_app.py', 'w') as f:
    f.write(streamlit_app_code)

print("✅ Web app code created: 'financial_ai_app.py'")
print("📁 Next steps:")
print("1. Save your models (run the save code from earlier)")
print("2. Upload to Streamlit Cloud")
print("3. Share the link with your boss!")
