import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import os

# Page title - MUST BE FIRST
st.set_page_config(page_title="Financial AI Assistant", page_icon="🤖")

st.title("🤖 Financial AI Assistant")
st.markdown("**Advanced AI-Powered Financial Analysis & Portfolio Optimization**")

# DEBUG: Show files
st.write("🔍 **Files found:**")
files = [f for f in os.listdir('.') if f.endswith('.csv')]
for file in files:
    st.write(f"• {file}")
st.write("---")

# Load CSV data function
@st.cache_data
def load_real_csv_data():
    """Load and process the actual CSV file"""
    try:
        # Use the exact filename we found
        filename = 'Savola Almarai NADEC Financial Ratios CSV.csv.csv'
        st.write(f"🔍 Attempting to load: {filename}")
        
        df = pd.read_csv(filename)
        st.write(f"✅ Raw data loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Show first few rows for debugging
        st.write("📊 **First 3 rows of data:**")
        st.dataframe(df.head(3))
        
        # Clean column names (remove extra spaces)
        df.columns = df.columns.str.strip()
        
        # Remove empty columns
        df = df.dropna(axis=1, how='all')
        
        # Filter out empty rows
        df = df.dropna(subset=['Company', 'Year'])
        
        st.write(f"✅ Cleaned data: {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        st.error(f"❌ Error loading CSV: {e}")
        return None

# Calculate company statistics from real data
@st.cache_data
def calculate_company_stats(df):
    """Calculate real statistics from the CSV data"""
    companies = df['Company'].unique()
    st.write(f"📊 **Companies found:** {list(companies)}")
    
    company_stats = {}
    
    for company in companies:
        company_data = df[df['Company'] == company].copy()
        st.write(f"📈 **{company}**: {len(company_data)} data points")
        
        # Calculate historical averages
        avg_roe = company_data['ROE'].mean()
        avg_roa = company_data['ROA'].mean() 
        avg_gross_margin = company_data['Gross Margin'].mean()
        avg_net_margin = company_data['Net Profit Margin'].mean()
        
        # Handle the 'Current Ratio' column (note the spaces)
        current_ratio_col = 'Current Ratio'
        if 'Current Ratio' not in company_data.columns:
            # Try alternative column names
            for col in company_data.columns:
                if 'current' in col.lower() and 'ratio' in col.lower():
                    current_ratio_col = col
                    break
        
        avg_current_ratio = company_data[current_ratio_col].mean()
        avg_debt_to_equity = company_data['Debt-to-Equity'].mean()
        avg_debt_to_assets = company_data['Debt-to-Assets'].mean()
        
        # Calculate volatility (standard deviation)
        roe_volatility = company_data['ROE'].std()
        
        # Expected return (based on historical average)
        expected_return = avg_roe * 1.1
        
        # Get latest data (most recent entry)
        latest_data = company_data.iloc[-1]
        
        # Historical ROE values and years for trends
        historical_roe = company_data.groupby('Year')['ROE'].mean().values
        years = sorted(company_data['Year'].unique())
        
        # Determine growth trend
        if len(historical_roe) >= 3:
            recent_trend = np.polyfit(range(len(historical_roe[-3:])), historical_roe[-3:], 1)[0]
            if recent_trend > 0.005:
                growth_trend = 'Improving'
            elif recent_trend < -0.005:
                growth_trend = 'Declining'
            else:
                growth_trend = 'Stable'
        else:
            growth_trend = 'Stable'
        
        # Risk level based on volatility
        if roe_volatility < 0.03:
            risk_level = 'Low'
        elif roe_volatility < 0.06:
            risk_level = 'Medium'
        else:
            risk_level = 'High'
        
        company_stats[company] = {
            'latest_ratios': {
                'Gross Margin': latest_data['Gross Margin'],
                'Net Profit Margin': latest_data['Net Profit Margin'],
                'ROA': latest_data['ROA'],
                'Current Ratio': latest_data[current_ratio_col],
                'Debt-to-Equity': latest_data['Debt-to-Equity'],
                'Debt-to-Assets': latest_data['Debt-to-Assets']
            },
            'avg_performance': {
                'ROE': avg_roe,
                'Growth_Trend': growth_trend,
                'Risk_Level': risk_level,
                'Expected_Return': expected_return,
                'Volatility': roe_volatility,
                'Sharpe_Ratio': (expected_return - 0.025) / roe_volatility if roe_volatility > 0 else 0
            },
            'historical_roe': historical_roe.tolist(),
            'years': years,
            'data_points': len(company_data)
        }
    
    return company_stats

# Portfolio optimization functions
def calculate_portfolio_return(weights, expected_returns):
    """Calculate expected portfolio return"""
    return np.sum(weights * expected_returns)

def calculate_portfolio_volatility(weights, cov_matrix):
    """Calculate portfolio volatility (risk)"""
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

def check_target_feasibility(company_stats, target_return):
    """Check if the target return is achievable with available companies"""
    companies = list(company_stats.keys())
    
    # Find maximum possible return (best single company)
    max_single_return = max(company_stats[comp]['avg_performance']['Expected_Return'] 
                           for comp in companies)
    
    # Calculate realistic maximum (accounting for diversification)
    expected_returns = [company_stats[comp]['avg_performance']['Expected_Return'] 
                       for comp in companies]
    sorted_returns = sorted(expected_returns, reverse=True)
    
    if len(sorted_returns) >= 2:
        realistic_max = sorted_returns[0] * 0.8 + sorted_returns[1] * 0.2
    else:
        realistic_max = sorted_returns[0]
    
    return {
        'is_feasible': target_return <= realistic_max,
        'target_return': target_return,
        'max_single_return': max_single_return,
        'realistic_max': realistic_max,
        'gap': target_return - realistic_max if target_return > realistic_max else 0,
        'best_company': max(companies, key=lambda x: company_stats[x]['avg_performance']['Expected_Return'])
    }

# Load real data
csv_data = load_real_csv_data()
if csv_data is not None:
    company_stats = calculate_company_stats(csv_data)
    st.success("✅ Real CSV data loaded successfully!")
    st.info(f"📊 Loaded {len(csv_data)} data points from {len(company_stats)} companies ({csv_data['Year'].min()}-{csv_data['Year'].max()})")
    
    # Show what the AI found in your data
    st.subheader("🔍 What Your Real Data Shows:")
    for company, stats in company_stats.items():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric(f"{company} Avg ROE", f"{stats['avg_performance']['ROE']:.1%}")
        with col2:
            st.metric(f"{company} Expected Return", f"{stats['avg_performance']['Expected_Return']:.1%}")
        with col3:
            st.metric(f"{company} Risk Level", stats['avg_performance']['Risk_Level'])
    
else:
    st.error("❌ Failed to load CSV data!")
    company_stats = {}

if company_stats:
    # Create sections with tabs
    tab1, tab2, tab3 = st.tabs(["📊 Real Data Analysis", "🔮 2024 Predictions", "🤖 Honest Portfolio Builder"])
    
    # ============================================================================
    # TAB 1: REAL DATA ANALYSIS
    # ============================================================================
    with tab1:
        st.header("📊 Real Data from Your CSV File")
        st.markdown(f"*Analysis based on actual data from {csv_data['Year'].min()}-{csv_data['Year'].max()}*")
        
        # Show real statistics
        for company, stats in company_stats.items():
            with st.expander(f"📊 {company} Detailed Analysis"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Latest Financial Ratios (2023):**")
                    for ratio, value in stats['latest_ratios'].items():
                        if 'Ratio' in ratio:
                            st.write(f"• {ratio}: {value:.2f}")
                        else:
                            st.write(f"• {ratio}: {value:.1%}")
                
                with col2:
                    st.write("**Historical Performance:**")
                    st.write(f"• Average ROE: {stats['avg_performance']['ROE']:.1%}")
                    st.write(f"• Expected Return: {stats['avg_performance']['Expected_Return']:.1%}")
                    st.write(f"• Volatility: {stats['avg_performance']['Volatility']:.1%}")
                    st.write(f"• Risk Level: {stats['avg_performance']['Risk_Level']}")
                    st.write(f"• Growth Trend: {stats['avg_performance']['Growth_Trend']}")
                    st.write(f"• Data Points: {stats['data_points']}")
        
        # Historical trends chart
        st.subheader("📈 Historical ROE Trends (Real Data)")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for company, stats in company_stats.items():
            ax.plot(stats['years'], [r * 100 for r in stats['historical_roe']], 
                   marker='o', linewidth=2, label=company)
        
        ax.set_xlabel('Year')
        ax.set_ylabel('ROE (%)')
        ax.set_title('Historical ROE Trends from Your CSV Data')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        
        # Company comparison table
        st.subheader("🏆 Company Comparison (Real Data)")
        comparison_data = []
        for company, stats in company_stats.items():
            comparison_data.append({
                'Company': company,
                'Historical Avg ROE': f"{stats['avg_performance']['ROE']:.1%}",
                'Expected Return': f"{stats['avg_performance']['Expected_Return']:.1%}",
                'Volatility': f"{stats['avg_performance']['Volatility']:.1%}",
                'Risk Level': stats['avg_performance']['Risk_Level'],
                'Growth Trend': stats['avg_performance']['Growth_Trend'],
                'Data Points': stats['data_points']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, hide_index=True)
    
    # ============================================================================
    # TAB 2: 2024 PREDICTIONS
    # ============================================================================
    with tab2:
        st.header("🔮 2024 Predictions Based on Your Real Data")
        st.markdown("*Forecasts using actual historical patterns from your CSV*")
        
        selected_company = st.selectbox(
            "🏢 Select Company for 2024 Prediction:",
            ["Select a company..."] + list(company_stats.keys()),
            index=0
        )
        
        if selected_company != "Select a company...":
            stats = company_stats[selected_company]
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader(f"📊 {selected_company} Current Status")
                
                # Show latest ratios from CSV
                st.write("**Latest Financial Ratios (from your CSV):**")
                for ratio, value in stats['latest_ratios'].items():
                    if 'Ratio' in ratio:
                        st.write(f"• {ratio}: {value:.2f}")
                    else:
                        st.write(f"• {ratio}: {value:.1%}")
                
                st.write("**Historical Performance:**")
                st.write(f"• Average ROE (2016-2023): {stats['avg_performance']['ROE']:.1%}")
                st.write(f"• Volatility: {stats['avg_performance']['Volatility']:.1%}")
                st.write(f"• Growth Trend: {stats['avg_performance']['Growth_Trend']}")
            
            with col2:
                st.subheader(f"🔮 2024 Forecast")
                
                # Simple trend-based prediction
                historical_roe = stats['historical_roe']
                if len(historical_roe) >= 3:
                    # Linear trend extrapolation
                    years_for_trend = list(range(len(historical_roe)))
                    trend_coef = np.polyfit(years_for_trend, historical_roe, 1)[0]
                    predicted_2024_roe = historical_roe[-1] + trend_coef
                else:
                    predicted_2024_roe = stats['avg_performance']['Expected_Return']
                
                # Ensure reasonable bounds
                predicted_2024_roe = max(0, min(predicted_2024_roe, stats['avg_performance']['ROE'] * 2))
                
                st.metric("🔮 Predicted 2024 ROE", f"{predicted_2024_roe:.1%}",
                         delta=f"{predicted_2024_roe - stats['avg_performance']['ROE']:.1%} vs Historical Avg")
                
                # Investment recommendation based on real data
                if predicted_2024_roe > stats['avg_performance']['ROE'] * 1.1:
                    recommendation = "Buy"
                    st.success("💚 **RECOMMENDED - IMPROVING TREND**")
                elif predicted_2024_roe > stats['avg_performance']['ROE'] * 0.9:
                    recommendation = "Hold"
                    st.warning("💛 **HOLD - STABLE PERFORMANCE**")
                else:
                    recommendation = "Sell"
                    st.error("💔 **CAUTION - DECLINING TREND**")
                
                st.metric("💰 CSV-Based Recommendation", recommendation)
                st.metric("📊 Expected Return Range", f"{stats['avg_performance']['Expected_Return']:.1%}")
                st.metric("⚖️ Risk Assessment", stats['avg_performance']['Risk_Level'])
    
    # ============================================================================
    # TAB 3: HONEST PORTFOLIO BUILDER (WITH REAL DATA)
    # ============================================================================
    with tab3:
        st.header("🤖 Honest Portfolio Builder (Real Data)")
        st.markdown("*Portfolio optimization using your actual CSV data*")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("🎯 Your Investment Goals")
            
            investment_amount = st.number_input("💰 Investment Amount (SAR)", 
                                              min_value=10000, max_value=10000000, 
                                              value=100000, step=10000)
            
            target_return = st.slider("🎯 Target Annual Return (%)", 
                                    min_value=1, max_value=15, value=5) / 100
            
            max_possible_return = max(stats['avg_performance']['Expected_Return'] for stats in company_stats.values())
            st.info(f"💡 **Based on your real data**: Maximum realistic return is ~{max_possible_return:.1%}")
            
            if st.button("🚀 BUILD REALISTIC PORTFOLIO", type="primary"):
                # Check feasibility with real data
                feasibility = check_target_feasibility(company_stats, target_return)
                
                with col2:
                    st.subheader("🎯 Real Data Analysis")
                    
                    if feasibility['is_feasible']:
                        st.success("✅ **TARGET IS ACHIEVABLE WITH YOUR DATA**")
                        
                        # Show simple allocation based on performance
                        companies = list(company_stats.keys())
                        expected_returns = [company_stats[comp]['avg_performance']['Expected_Return'] 
                                          for comp in companies]
                        
                        # Simple allocation: weight by performance
                        total_performance = sum(expected_returns)
                        weights = [ret / total_performance for ret in expected_returns]
                        
                        portfolio_return = sum(w * r for w, r in zip(weights, expected_returns))
                        
                        # Show results
                        st.metric("🎯 Expected Return", f"{portfolio_return:.1%}",
                                 delta=f"{portfolio_return - target_return:.1%} vs Target")
                        
                        # Allocation
                        st.subheader("💰 Recommended Allocation")
                        for i, company in enumerate(companies):
                            weight = weights[i]
                            amount = investment_amount * weight
                            expected_return_comp = company_stats[company]['avg_performance']['Expected_Return']
                            st.write(f"**{company}**: {weight:.1%} ({amount:,.0f} SAR) - Expected: {expected_return_comp:.1%}")
                        
                        st.success("✅ **PORTFOLIO RECOMMENDED BASED ON YOUR REAL DATA**")
                    
                    else:
                        # Show honest analysis
                        st.error("❌ **TARGET NOT ACHIEVABLE WITH YOUR DATA**")
                        st.write(f"🎯 **Your Target**: {target_return:.1%}")
                        st.write(f"📊 **Maximum Possible**: {feasibility['realistic_max']:.1%}")
                        st.write(f"📈 **Gap**: {feasibility['gap']:.1%}")
                        
                        # Show what's actually possible
                        st.subheader("📊 What Your Real Data Shows")
                        for company, stats in company_stats.items():
                            st.write(f"• **{company}**: {stats['avg_performance']['Expected_Return']:.1%} expected return")
                        
                        # Suggest realistic target
                        realistic_target = feasibility['realistic_max'] * 0.9
                        st.info(f"💡 **Suggested Target**: {realistic_target:.1%} (achievable with your data)")
                        
                        st.error("🚫 **NO PORTFOLIO RECOMMENDED - TARGET TOO HIGH FOR AVAILABLE COMPANIES**")

    # ============================================================================
    # CHAT SECTION (UPDATED FOR REAL DATA)
    # ============================================================================
    st.header("💬 Ask About Your Real Data")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("❓ What does my real data show?"):
            best_company = max(company_stats.keys(), key=lambda x: company_stats[x]['avg_performance']['Expected_Return'])
            best_return = company_stats[best_company]['avg_performance']['Expected_Return']
            st.info(f"🤖 **Real Data Shows:** {best_company} is your best performer with {best_return:.1%} expected return. Your CSV contains {len(csv_data)} real data points from 2016-2023.")
    
    with col2:
        if st.button("❓ Can I get 10% returns?"):
            max_possible = max(stats['avg_performance']['Expected_Return'] for stats in company_stats.values())
            if max_possible >= 0.10:
                st.info("🤖 **Maybe!** Your data suggests it might be possible with the right mix.")
            else:
                st.info(f"🤖 **NO** - Your real data shows maximum ~{max_possible:.1%}. For 10%+ returns, you need different companies or asset classes not in your current dataset.")
    
    with col3:
        if st.button("❓ Which company is actually best?"):
            best_company = max(company_stats.keys(), key=lambda x: company_stats[x]['avg_performance']['Expected_Return'])
            best_return = company_stats[best_company]['avg_performance']['Expected_Return']
            st.info(f"🤖 **Based on YOUR data:** {best_company} with {best_return:.1%} expected return and {company_stats[best_company]['avg_performance']['Risk_Level']} risk level.")

# Footer
st.markdown("---")
st.markdown("**🤖 Powered by YOUR Real Data • CSV-Based Analysis • Truth From Your Numbers**")
if csv_data is not None:
    st.markdown(f"*Analysis based on {len(csv_data)} actual data points • No fake estimates • Your data speaks truth*")
else:
    st.markdown("*Waiting for CSV data to be loaded...*")
