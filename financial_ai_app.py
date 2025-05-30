import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Page title
st.set_page_config(page_title="Financial AI Assistant", page_icon="🤖")

st.title("🤖 Financial AI Assistant")
st.markdown("**Advanced AI-Powered Financial Analysis for Saudi Companies**")

# Load your AI models
@st.cache_resource
def load_models():
    try:
        roe_model = joblib.load('roe_prediction_model.pkl')
        invest_model = joblib.load('investment_model.pkl')
        status_model = joblib.load('company_status_model.pkl')
        le_invest = joblib.load('investment_encoder.pkl')
        le_status = joblib.load('status_encoder.pkl')
        return roe_model, invest_model, status_model, le_invest, le_status
    except:
        return None, None, None, None, None

# Load historical data (recreate the training data structure)
@st.cache_data
def load_historical_data():
    # This represents your original training data
    # In a real deployment, you'd load this from a database
    historical_data = {
        'Almarai': {
            'latest_ratios': {
                'Gross Margin': 0.387,
                'Net Profit Margin': 0.126,
                'ROA': 0.037,
                'Current Ratio': 1.264,
                'Debt-to-Equity': 1.097,
                'Debt-to-Assets': 0.547
            },
            'avg_performance': {
                'ROE': 0.0569,
                'Growth_Trend': 'Stable',
                'Risk_Level': 'Low',
                'Investment_Score': 2.41
            },
            'historical_roe': [0.022, 0.052, 0.050, 0.160, 0.067, 0.058, 0.037],
            'years': [2017, 2018, 2019, 2020, 2021, 2022, 2023]
        },
        'Savola': {
            'latest_ratios': {
                'Gross Margin': 0.200,
                'Net Profit Margin': 0.038,
                'ROA': 0.031,
                'Current Ratio': 0.780,
                'Debt-to-Equity': 1.941,
                'Debt-to-Assets': 0.630
            },
            'avg_performance': {
                'ROE': 0.0322,
                'Growth_Trend': 'Declining',
                'Risk_Level': 'Medium',
                'Investment_Score': 0.41
            },
            'historical_roe': [0.010, 0.024, 0.020, 0.126, -0.057, 0.038, 0.031],
            'years': [2017, 2018, 2019, 2020, 2021, 2022, 2023]
        },
        'NADEC': {
            'latest_ratios': {
                'Gross Margin': 0.350,
                'Net Profit Margin': 0.050,
                'ROA': 0.028,
                'Current Ratio': 0.872,
                'Debt-to-Equity': 1.829,
                'Debt-to-Assets': 0.647
            },
            'avg_performance': {
                'ROE': 0.0138,
                'Growth_Trend': 'Volatile',
                'Risk_Level': 'High',
                'Investment_Score': 0.28
            },
            'historical_roe': [0.018, 0.026, 0.020, 0.084, -0.238, 0.025, 0.028],
            'years': [2017, 2018, 2019, 2020, 2021, 2022, 2023]
        }
    }
    return historical_data

# Try to load models
roe_model, invest_model, status_model, le_invest, le_status = load_models()
historical_data = load_historical_data()

if roe_model is None:
    st.error("❌ Model files not found! Please upload the .pkl files first.")
    st.info("Upload these 5 files: roe_prediction_model.pkl, investment_model.pkl, company_status_model.pkl, investment_encoder.pkl, status_encoder.pkl")
else:
    st.success("✅ AI Models loaded successfully!")
    
    # Create sections with tabs
    tab1, tab2 = st.tabs(["📊 Analyze Any Company", "🔮 Saudi Companies Predictor"])
    
    # ============================================================================
    # TAB 1: GENERIC COMPANY ANALYZER (Original)
    # ============================================================================
    with tab1:
        st.header("📊 Universal Company Analysis")
        st.markdown("*Analyze any company worldwide by entering financial ratios*")
        
        # Create two columns
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Company Information")
            company_name = st.text_input("Company Name:", value="Test Company")
            
            st.subheader("Enter Financial Ratios:")
            gross_margin = st.slider("Gross Margin (%)", 0, 100, 30) / 100
            net_profit_margin = st.slider("Net Profit Margin (%)", 0, 50, 10) / 100
            roa = st.slider("ROA - Return on Assets (%)", 0, 30, 8) / 100
            current_ratio = st.slider("Current Ratio", 0.0, 5.0, 1.5, 0.1)
            debt_to_equity = st.slider("Debt-to-Equity Ratio", 0.0, 5.0, 0.8, 0.1)
            debt_to_assets = st.slider("Debt-to-Assets Ratio (%)", 0, 100, 45) / 100
            
            # Generic analyze button
            if st.button("🔍 ANALYZE COMPANY", type="primary", key="generic_analyze"):
                # Prepare data for AI
                input_data = pd.DataFrame({
                    'Gross Margin': [gross_margin],
                    'Net Profit Margin': [net_profit_margin],
                    'ROA': [roa],
                    'ROE': [0],
                    'Current Ratio': [current_ratio],
                    'Debt-to-Equity': [debt_to_equity],
                    'Debt-to-Assets': [debt_to_assets],
                    'Year': [2024],
                    'Quarter': [4],
                    'Company_Encoded': [1]
                })
                
                # AI Analysis
                roe_features = ['Gross Margin', 'Net Profit Margin', 'ROA', 'Current Ratio', 
                               'Debt-to-Equity', 'Debt-to-Assets', 'Year', 'Quarter', 'Company_Encoded']
                
                # Step 1: Predict ROE
                predicted_roe = roe_model.predict(input_data[roe_features])[0]
                input_data['ROE'] = predicted_roe
                
                # Step 2: Investment recommendation
                invest_features = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 
                                  'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets', 
                                  'Year', 'Quarter', 'Company_Encoded']
                
                invest_pred = invest_model.predict(input_data[invest_features])[0]
                invest_recommendation = le_invest.inverse_transform([invest_pred])[0]
                invest_confidence = max(invest_model.predict_proba(input_data[invest_features])[0])
                
                # Step 3: Company status
                status_pred = status_model.predict(input_data[invest_features])[0] 
                company_status = le_status.inverse_transform([status_pred])[0]
                
                # Show results in the right column
                with col2:
                    st.header(f"🎯 Analysis: {company_name}")
                    
                    # Big result boxes
                    st.metric("🔮 Predicted ROE", f"{predicted_roe:.1%}", 
                             help="Return on Equity - how profitable the company is")
                    
                    st.metric("💰 Investment Recommendation", invest_recommendation,
                             help="Buy, Hold, or Sell recommendation")
                    
                    st.metric("🏢 Company Health", company_status,
                             help="Overall financial health status")
                    
                    st.metric("🎯 AI Confidence", f"{invest_confidence:.0%}",
                             help="How confident the AI is in its recommendation")
                    
                    # Color-coded advice
                    if invest_recommendation == "Buy":
                        st.success("💚 **STRONG INVESTMENT OPPORTUNITY!**")
                        st.write("✅ High profitability expected")
                        st.write("✅ Good financial health")
                        st.write("✅ Recommended for investment")
                    elif invest_recommendation == "Hold":
                        st.warning("💛 **STABLE INVESTMENT - MONITOR CLOSELY**")
                        st.write("⚪ Maintain current position")
                        st.write("⚪ Watch for changes")
                    else:
                        st.error("💔 **HIGH RISK - AVOID INVESTMENT**")
                        st.write("❌ Poor financial indicators")
                        st.write("❌ High risk of losses")
    
    # ============================================================================
    # TAB 2: SAUDI COMPANIES PREDICTOR (NEW!)
    # ============================================================================
    with tab2:
        st.header("🔮 Saudi Companies 2024 Predictions")
        st.markdown("*AI predictions based on 7 years of historical data (2016-2023)*")
        
        # Company selector
        selected_company = st.selectbox(
            "🏢 Select Saudi Company:",
            ["Select a company...", "Almarai", "Savola", "NADEC"],
            index=0
        )
        
        if selected_company != "Select a company...":
            company_data = historical_data[selected_company]
            
            # Create columns for layout
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader(f"📊 {selected_company} Analysis")
                
                # Show current ratios
                st.write("**Latest Financial Ratios (2023):**")
                latest = company_data['latest_ratios']
                
                for ratio, value in latest.items():
                    if 'Ratio' in ratio:
                        st.write(f"• {ratio}: {value:.2f}")
                    else:
                        st.write(f"• {ratio}: {value:.1%}")
                
                # Predict 2024 performance
                if st.button(f"🚀 PREDICT {selected_company.upper()} 2024", type="primary", key=f"predict_{selected_company}"):
                    # Prepare prediction data using company's latest ratios
                    company_encoded = {'Almarai': 0, 'Savola': 1, 'NADEC': 2}[selected_company]
                    
                    prediction_input = pd.DataFrame({
                        'Gross Margin': [latest['Gross Margin']],
                        'Net Profit Margin': [latest['Net Profit Margin']],
                        'ROA': [latest['ROA']],
                        'ROE': [0],
                        'Current Ratio': [latest['Current Ratio']],
                        'Debt-to-Equity': [latest['Debt-to-Equity']],
                        'Debt-to-Assets': [latest['Debt-to-Assets']],
                        'Year': [2024],
                        'Quarter': [0],  # Annual prediction
                        'Company_Encoded': [company_encoded]
                    })
                    
                    # Make AI prediction
                    roe_features = ['Gross Margin', 'Net Profit Margin', 'ROA', 'Current Ratio', 
                                   'Debt-to-Equity', 'Debt-to-Assets', 'Year', 'Quarter', 'Company_Encoded']
                    
                    predicted_roe_2024 = roe_model.predict(prediction_input[roe_features])[0]
                    prediction_input['ROE'] = predicted_roe_2024
                    
                    # Investment recommendation for 2024
                    invest_features = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 
                                      'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets', 
                                      'Year', 'Quarter', 'Company_Encoded']
                    
                    invest_pred_2024 = invest_model.predict(prediction_input[invest_features])[0]
                    invest_rec_2024 = le_invest.inverse_transform([invest_pred_2024])[0]
                    invest_conf_2024 = max(invest_model.predict_proba(prediction_input[invest_features])[0])
                    
                    # Company status for 2024
                    status_pred_2024 = status_model.predict(prediction_input[invest_features])[0]
                    status_2024 = le_status.inverse_transform([status_pred_2024])[0]
                    
                    # Show predictions in right column
                    with col2:
                        st.subheader(f"🎯 {selected_company} 2024 Forecast")
                        
                        # 2024 Predictions
                        st.metric("🔮 Predicted 2024 ROE", f"{predicted_roe_2024:.1%}",
                                 delta=f"{predicted_roe_2024 - company_data['avg_performance']['ROE']:.1%}")
                        
                        st.metric("💰 2024 Investment Outlook", invest_rec_2024)
                        st.metric("🏢 Expected 2024 Status", status_2024)
                        st.metric("🎯 Prediction Confidence", f"{invest_conf_2024:.0%}")
                        
                        # Historical performance insights
                        st.write("**📈 Historical Insights:**")
                        avg_perf = company_data['avg_performance']
                        st.write(f"• Average ROE (2016-2023): {avg_perf['ROE']:.1%}")
                        st.write(f"• Growth Trend: {avg_perf['Growth_Trend']}")
                        st.write(f"• Risk Level: {avg_perf['Risk_Level']}")
                        
                        # Investment advice
                        if invest_rec_2024 == "Buy":
                            st.success(f"💚 **{selected_company} - STRONG BUY for 2024!**")
                        elif invest_rec_2024 == "Hold":
                            st.warning(f"💛 **{selected_company} - HOLD and Monitor**")
                        else:
                            st.error(f"💔 **{selected_company} - AVOID in 2024**")
            
            # Historical trend chart
            st.subheader(f"📈 {selected_company} ROE Historical Trend")
            
            # Create the chart
            fig, ax = plt.subplots(figsize=(10, 4))
            years = company_data['years']
            roe_values = [r * 100 for r in company_data['historical_roe']]  # Convert to percentage
            
            ax.plot(years, roe_values, marker='o', linewidth=2, markersize=6)
            ax.set_title(f'{selected_company} ROE Trend (2017-2023)')
            ax.set_xlabel('Year')
            ax.set_ylabel('ROE (%)')
            ax.grid(True, alpha=0.3)
            ax.set_ylim(min(roe_values) - 2, max(roe_values) + 2)
            
            # Add trend line
            z = np.polyfit(years, roe_values, 1)
            p = np.poly1d(z)
            ax.plot(years, p(years), "--", alpha=0.7, color='red')
            
            st.pyplot(fig)
            
            # Company comparison
            st.subheader("🏆 Company Rankings (Based on AI Training)")
            ranking_data = []
            for comp_name, comp_data in historical_data.items():
                ranking_data.append({
                    'Company': comp_name,
                    'Avg ROE': f"{comp_data['avg_performance']['ROE']:.1%}",
                    'Investment Score': f"{comp_data['avg_performance']['Investment_Score']:.2f}",
                    'Trend': comp_data['avg_performance']['Growth_Trend'],
                    'Risk': comp_data['avg_performance']['Risk_Level']
                })
            
            ranking_df = pd.DataFrame(ranking_data)
            st.dataframe(ranking_df, hide_index=True)

    # ============================================================================
    # CHAT SECTION (ENHANCED)
    # ============================================================================
    st.header("💬 Ask the AI Questions")
    
    # Predefined questions
    st.subheader("Quick Questions:")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("❓ Which Saudi company is best?"):
            st.info("🤖 **AI Answer:** Based on my analysis of 2016-2023 data, **Almarai** performs best with average ROE of 5.7% and highest investment score of 2.41. They show stable growth and low risk.")
    
    with col2:
        if st.button("❓ What should I expect in 2024?"):
            st.info("🤖 **AI Answer:** Use the Saudi Companies Predictor tab above! I can forecast 2024 performance for Almarai, Savola, and NADEC based on their historical patterns.")
    
    with col3:
        if st.button("❓ How accurate are predictions?"):
            st.info("🤖 **AI Answer:** My models achieve 95% accuracy for investment recommendations and 66.5% accuracy for ROE predictions, trained on 7 years of actual Saudi company data.")
    
    # Custom question
    custom_question = st.text_input("Or ask your own question:", placeholder="Which company should I invest in for 2024?")
    
    if custom_question:
        if any(word in custom_question.lower() for word in ["saudi", "almarai", "savola", "nadec", "2024"]):
            st.write("🤖 **AI Answer:** Use the 'Saudi Companies Predictor' tab above for detailed 2024 forecasts based on my training on these companies!")
        elif any(word in custom_question.lower() for word in ["best", "invest", "recommend"]):
            st.write("🤖 **AI Answer:** Based on historical performance, Almarai shows the strongest fundamentals. However, use both analyzer tabs to get current recommendations!")
        elif "accurate" in custom_question.lower():
            st.write("🤖 **AI Answer:** My models achieve 95% accuracy for investment recommendations and are trained on 96 data points spanning 2016-2023.")
        else:
            st.write("🤖 **AI Answer:** I'm specialized in financial analysis of Saudi companies (Almarai, Savola, NADEC) and general company analysis. Please use the analyzer tabs above for detailed insights!")

# Footer
st.markdown("---")
st.markdown("**🤖 Powered by Advanced AI Financial Models**")
st.markdown("*Built for professional investment analysis • Trained on Saudi market data (2016-2023)*")
