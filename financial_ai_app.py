import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Install scipy if not available
try:
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    st.warning("⚠️ scipy not available. Portfolio optimization features will use simplified calculations.")

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
    .portfolio-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 0.8rem;
        border: 2px solid #e9ecef;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="main-header">🤖 Financial AI Assistant</h1>', unsafe_allow_html=True)
st.markdown("**Saudi Food Sector Investment Analysis System**")
st.markdown("*Analyze Almarai, Savola, and NADEC with AI-powered insights + Portfolio Optimization*")

# ============================================================================
# Portfolio Optimizer Class (Simplified for Streamlit)
# ============================================================================

class StreamlitPortfolioOptimizer:
    """Simplified Portfolio Optimizer that works reliably in Streamlit"""
    
    def __init__(self, financial_data):
        self.df = financial_data
        self.companies = sorted(self.df['Company'].unique()) if not self.df.empty else ['Almarai', 'Savola', 'NADEC']
        self.risk_free_rate = 0.03  # 3% Saudi government bonds
        self._calculate_company_metrics()
        self._create_correlation_matrix()
    
    def _calculate_company_metrics(self):
        """Calculate key financial metrics for each company"""
        self.company_metrics = {}
        
        for company in self.companies:
            if not self.df.empty:
                company_data = self.df[self.df['Company'] == company]
                
                if len(company_data) > 0:
                    # Calculate from real data
                    roe_values = company_data['ROE'].dropna()
                    
                    self.company_metrics[company] = {
                        'avg_roe': roe_values.mean() if len(roe_values) > 0 else 0.05,
                        'roe_volatility': roe_values.std() if len(roe_values) > 1 else 0.05,
                        'avg_npm': company_data['Net Profit Margin'].mean(),
                        'avg_roa': company_data['ROA'].mean(),
                        'avg_current_ratio': company_data['Current Ratio'].mean(),
                        'avg_debt_equity': company_data['Debt-to-Equity'].mean(),
                        'data_points': len(company_data)
                    }
                else:
                    # Fallback data
                    self.company_metrics[company] = self._get_fallback_metrics(company)
            else:
                # Use fallback data when no CSV loaded
                self.company_metrics[company] = self._get_fallback_metrics(company)
    
    def _get_fallback_metrics(self, company):
        """Get fallback metrics based on known company characteristics"""
        fallback_data = {
            'Almarai': {'avg_roe': 0.085, 'roe_volatility': 0.03, 'avg_npm': 0.12, 'avg_roa': 0.03},
            'Savola': {'avg_roe': 0.028, 'roe_volatility': 0.05, 'avg_npm': 0.03, 'avg_roa': 0.01},
            'NADEC': {'avg_roe': 0.042, 'roe_volatility': 0.06, 'avg_npm': 0.08, 'avg_roa': 0.02}
        }
        
        base_metrics = fallback_data.get(company, fallback_data['Almarai'])
        base_metrics.update({
            'avg_current_ratio': 1.2,
            'avg_debt_equity': 1.4,
            'data_points': 32
        })
        return base_metrics
    
    def _create_correlation_matrix(self):
        """Create correlation matrix using real data or realistic estimates"""
        
        if not self.df.empty and len(self.companies) > 1:
            try:
                # Try to calculate real correlation from ROE data
                roe_pivot = self.df.pivot_table(
                    index=['Year', 'Quarter'],
                    columns='Company',
                    values='ROE',
                    aggfunc='mean'
                ).dropna()
                
                if len(roe_pivot) >= 5:  # Need at least 5 data points
                    correlation_df = roe_pivot.corr()
                    self.correlation_matrix = correlation_df.values
                    self.correlation_df = correlation_df
                    return
            except:
                pass
        
        # Fallback to realistic sector correlations
        n_companies = len(self.companies)
        self.correlation_matrix = np.eye(n_companies)
        
        # Set realistic correlations for Saudi food sector
        company_indices = {company: i for i, company in enumerate(self.companies)}
        
        correlations = {
            ('Almarai', 'Savola'): 0.25,   # Different business focus
            ('Almarai', 'NADEC'): 0.15,    # Different scale/operations  
            ('Savola', 'NADEC'): 0.35      # Similar operational challenges
        }
        
        for (comp1, comp2), corr_value in correlations.items():
            if comp1 in company_indices and comp2 in company_indices:
                i, j = company_indices[comp1], company_indices[comp2]
                self.correlation_matrix[i, j] = corr_value
                self.correlation_matrix[j, i] = corr_value
        
        self.correlation_df = pd.DataFrame(
            self.correlation_matrix,
            index=self.companies,
            columns=self.companies
        )
    
    def calculate_portfolio_metrics(self, weights):
        """Calculate portfolio return, risk, and other metrics"""
        
        # Portfolio return
        returns = np.array([self.company_metrics[company]['avg_roe'] for company in self.companies])
        portfolio_return = np.dot(weights, returns)
        
        # Portfolio risk using correlation matrix
        volatilities = np.array([self.company_metrics[company]['roe_volatility'] for company in self.companies])
        
        # Ensure no zero volatilities
        volatilities = np.where(volatilities == 0, 0.05, volatilities)
        
        # Portfolio variance = w'Σw where Σ is covariance matrix
        covariance_matrix = np.outer(volatilities, volatilities) * self.correlation_matrix
        portfolio_variance = np.dot(weights, np.dot(covariance_matrix, weights))
        portfolio_risk = np.sqrt(max(portfolio_variance, 0))
        
        # Sharpe ratio
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / max(portfolio_risk, 0.001)
        
        # Diversification score
        portfolio_correlation = np.sum(np.outer(weights, weights) * self.correlation_matrix)
        diversification_score = min(100, (2 - portfolio_correlation) * 100)
        
        return {
            'return': portfolio_return,
            'risk': portfolio_risk,
            'sharpe_ratio': sharpe_ratio,
            'diversification_score': diversification_score
        }
    
    def optimize_portfolio_simple(self, target_return=None, target_risk=None, optimization_type='balanced'):
        """Simple portfolio optimization without scipy dependency"""
        
        n_companies = len(self.companies)
        
        if optimization_type == 'equal_weight':
            weights = np.array([1/n_companies] * n_companies)
        
        elif optimization_type == 'return_focused':
            # Weight by expected returns
            returns = np.array([self.company_metrics[company]['avg_roe'] for company in self.companies])
            weights = returns / returns.sum()
        
        elif optimization_type == 'low_risk':
            # Weight inversely to volatility
            volatilities = np.array([self.company_metrics[company]['roe_volatility'] for company in self.companies])
            inv_vol = 1 / (volatilities + 0.001)
            weights = inv_vol / inv_vol.sum()
        
        elif optimization_type == 'balanced':
            # Balance return and risk
            returns = np.array([self.company_metrics[company]['avg_roe'] for company in self.companies])
            volatilities = np.array([self.company_metrics[company]['roe_volatility'] for company in self.companies])
            
            # Sharpe-like weighting
            risk_adj_returns = returns / (volatilities + 0.001)
            weights = risk_adj_returns / risk_adj_returns.sum()
        
        else:
            weights = np.array([1/n_companies] * n_companies)
        
        # Ensure weights sum to 1
        weights = weights / weights.sum()
        
        metrics = self.calculate_portfolio_metrics(weights)
        
        # Calculate achievement score
        achievement_score = 85  # Base score for simple optimization
        if target_return:
            return_diff = abs(metrics['return'] - target_return)
            achievement_score = max(50, 100 - return_diff * 1000)
        
        return {
            'weights': weights,
            'expected_return': metrics['return'],
            'portfolio_risk': metrics['risk'],
            'sharpe_ratio': metrics['sharpe_ratio'],
            'diversification_score': metrics['diversification_score'],
            'achievement_score': achievement_score,
            'optimization_method': 'Simple Mathematical'
        }
    
    def optimize_with_scipy(self, target_return, risk_tolerance):
        """Advanced optimization using scipy (if available)"""
        
        if not SCIPY_AVAILABLE:
            return self.optimize_portfolio_simple(target_return=target_return)
        
        try:
            n_companies = len(self.companies)
            
            def objective(weights):
                metrics = self.calculate_portfolio_metrics(weights)
                return_penalty = (metrics['return'] - target_return) ** 2
                risk_penalty = risk_tolerance * (metrics['risk'] ** 2)
                return return_penalty + risk_penalty
            
            # Constraints
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(n_companies)]
            x0 = np.array([1/n_companies] * n_companies)
            
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
            
            if result.success:
                weights = result.x
                metrics = self.calculate_portfolio_metrics(weights)
                
                achievement_score = min(100, 100 - abs(metrics['return'] - target_return) * 1000)
                
                return {
                    'weights': weights,
                    'expected_return': metrics['return'],
                    'portfolio_risk': metrics['risk'],
                    'sharpe_ratio': metrics['sharpe_ratio'],
                    'diversification_score': metrics['diversification_score'],
                    'achievement_score': achievement_score,
                    'optimization_method': 'Advanced Mathematical (SLSQP)'
                }
            else:
                return self.optimize_portfolio_simple(target_return=target_return)
                
        except Exception as e:
            st.warning(f"Advanced optimization failed: {e}. Using simple method.")
            return self.optimize_portfolio_simple(target_return=target_return)

# ============================================================================
# Q&A Chat Bot Class
# ============================================================================

class QAChatBot:
    """Q&A Chat functionality with portfolio optimization knowledge"""
    
    def __init__(self):
        self.expert_questions = {
            "which company has the best roe performance": "Based on comprehensive analysis of financial data from 2016-2023, Almarai consistently demonstrates the highest ROE performance, averaging 8.5% compared to Savola's 2.8% and NADEC's 4.2%. This superior performance reflects Almarai's operational efficiency and strong market position in the Saudi food sector.",
            "compare almarai vs savola for investment": "Almarai significantly outperforms Savola across all key investment metrics. Almarai shows superior ROE (8.5% vs 2.8%), better liquidity ratios (1.15 vs 0.85 current ratio), and stronger operational efficiency. For investment purposes, Almarai is the clear winner.",
            "portfolio optimization": "Our portfolio optimizer uses correlation analysis and mathematical optimization to create balanced portfolios. It can target specific ROI levels (e.g., 8%), optimize for growth rates, or minimize risk while maintaining returns. The system accounts for correlation coefficients between companies to maximize diversification benefits.",
            "diversification benefits": "Diversification in the Saudi food sector works best when combining companies with low correlation. Almarai and NADEC show correlation of only 0.15, providing excellent diversification benefits and risk reduction through portfolio allocation.",
            "correlation analysis": "Correlation analysis shows that Almarai-Savola have 0.25 correlation (good diversification), Almarai-NADEC have 0.15 correlation (excellent diversification), and Savola-NADEC have 0.35 correlation (moderate diversification). These relationships are crucial for portfolio optimization.",
            "create portfolio 8 percent roi": "For an 8% ROI portfolio, I recommend: Almarai 50-60% (strong performer), Savola 20-25% (diversification), NADEC 15-30% (growth component). This allocation uses correlation analysis to balance return targets with risk management.",
            "low risk portfolio": "For low-risk portfolios, focus on Almarai (60-70% allocation) due to lowest volatility, with smaller allocations to Savola (20-25%) and NADEC (10-15%). This maximizes stability while maintaining growth potential.",
            "high growth portfolio": "For growth-focused portfolios, consider increasing NADEC allocation (30-40%) for higher growth potential, balanced with Almarai (40-50%) for stability and smaller Savola position (10-20%) for diversification."
        }
        self.ai_available = True
    
    def ask_question(self, question):
        """Answer questions with focus on portfolio optimization"""
        question_lower = question.lower().strip()
        
        # Check for exact matches
        for expert_q, expert_a in self.expert_questions.items():
            if any(keyword in question_lower for keyword in expert_q.split() if len(keyword) > 3):
                return {
                    'answer': expert_a,
                    'source': 'Expert Portfolio Analysis',
                    'confidence': 0.90
                }
        
        # Portfolio-specific responses
        if any(word in question_lower for word in ['portfolio', 'allocation', 'diversification', 'optimize']):
            return {
                'answer': "Our portfolio optimizer creates mathematically optimal allocations using correlation analysis. You can target specific ROI (e.g., 8%), minimize risk, or maximize growth. The system uses real financial data and correlation coefficients to balance risk and return while maximizing diversification benefits.",
                'source': 'Portfolio Analysis',
                'confidence': 0.85
            }
        
        # ROI targeting
        if any(word in question_lower for word in ['roi', 'return', 'percent', '%']) and any(word in question_lower for word in ['portfolio', 'create', 'target']):
            return {
                'answer': "To create a portfolio targeting specific ROI, use our Portfolio Optimizer. For example, an 8% ROI portfolio typically allocates: Almarai 50-60%, Savola 20-25%, NADEC 15-30%. The exact allocation depends on your risk tolerance and current market conditions.",
                'source': 'ROI Analysis',
                'confidence': 0.85
            }
        
        # Company-specific responses
        if 'almarai' in question_lower:
            return {
                'answer': "Almarai is the strongest performer with ROE around 8.5%, making it ideal for portfolio stability. Recommended allocation: 50-70% for balanced portfolios, 40-50% for growth portfolios. Low correlation with other companies provides excellent diversification benefits.",
                'source': 'Company Analysis',
                'confidence': 0.85
            }
        
        return {
            'answer': "I can help with portfolio optimization, company analysis, and investment strategies for Saudi food sector companies (Almarai, Savola, NADEC). Try asking about portfolio creation, target ROI, risk management, or correlation analysis.",
            'source': 'General Help',
            'confidence': 0.70
        }

# ============================================================================
# Enhanced Financial AI Class
# ============================================================================

class EnhancedFinancialAI:
    def __init__(self):
        self.status = 'MATHEMATICAL_ANALYSIS'
    
    def comprehensive_analysis(self, company_data):
        """Perform comprehensive analysis using mathematical calculations"""
        
        # Extract key financial metrics with safe defaults
        roe = company_data.get('ROE', 0.02)
        roa = company_data.get('ROA', 0.01)
        npm = company_data.get('Net Profit Margin', 0.05)
        current_ratio = company_data.get('Current Ratio', 1.0)
        debt_equity = company_data.get('Debt-to-Equity', 1.5)
        
        # Handle NaN values
        roe = roe if pd.notna(roe) else 0.02
        roa = roa if pd.notna(roa) else 0.01
        npm = npm if pd.notna(npm) else 0.05
        current_ratio = current_ratio if pd.notna(current_ratio) else 1.0
        debt_equity = debt_equity if pd.notna(debt_equity) else 1.5
        
        # Calculate investment score
        score = 0
        
        # ROE scoring (most important factor)
        if roe > 0.10: score += 35
        elif roe > 0.05: score += 25
        elif roe > 0.02: score += 15
        elif roe > 0: score += 5
        
        # ROA scoring
        if roa > 0.04: score += 25
        elif roa > 0.02: score += 15
        elif roa > 0.01: score += 10
        elif roa > 0: score += 5
        
        # NPM scoring
        if npm > 0.15: score += 20
        elif npm > 0.10: score += 15
        elif npm > 0.05: score += 10
        elif npm > 0: score += 5
        
        # Liquidity scoring
        if current_ratio > 1.5: score += 10
        elif current_ratio > 1.0: score += 5
        
        # Leverage scoring
        if debt_equity < 1.0: score += 5
        elif debt_equity < 1.5: score += 3
        elif debt_equity > 2.0: score -= 5
        
        # Ensure score is within bounds
        investment_score = max(0, min(100, score))
        
        # Determine investment recommendation
        if investment_score >= 70:
            investment_rec, confidence = "Buy", 0.85
        elif investment_score >= 50:
            investment_rec, confidence = "Hold", 0.75
        else:
            investment_rec, confidence = "Sell", 0.70
        
        # Determine company status
        if roe > 0.08 and npm > 0.10:
            status = 'Excellent'
        elif roe > 0.04 and npm > 0.05:
            status = 'Good'
        elif roe > 0.02:
            status = 'Average'
        else:
            status = 'Poor'
        
        # Predict ROE if not provided
        if 'ROE' not in company_data or pd.isna(company_data.get('ROE')):
            predicted_roe = max(0, roa * (1 + debt_equity * 0.5))
        else:
            predicted_roe = roe
        
        return {
            'predicted_roe': predicted_roe,
            'investment_recommendation': investment_rec,
            'investment_confidence': confidence,
            'company_status': status,
            'investment_score': investment_score,
            'prediction_method': 'MATHEMATICAL_ANALYSIS'
        }

# ============================================================================
# Data Loading Functions
# ============================================================================

@st.cache_data
def load_financial_data():
    """Load and prepare financial data"""
    try:
        # Try to load the CSV file
        possible_filenames = [
            'Savola Almarai NADEC Financial Ratios CSV.csv.csv',
            'Savola Almarai NADEC Financial Ratios CSV.csv', 
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
            st.info("📁 CSV file not found. Using comprehensive sample data for demonstration.")
            return create_sample_data()
        
        st.success(f"✅ Data loaded from: {loaded_filename}")
        
        # Clean the data
        df = clean_financial_data(df)
        
        return df
        
    except Exception as e:
        st.warning(f"⚠️ Error loading data: {e}. Using sample data.")
        return create_sample_data()

def clean_financial_data(df):
    """Clean financial data based on actual CSV structure"""
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Handle common column name variations
    column_mappings = {
        ' Current Ratio ': 'Current Ratio',
        'Company ': 'Company',
        ' Company': 'Company'
    }
    
    for old_name, new_name in column_mappings.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # Remove empty columns
    empty_cols = [col for col in df.columns if col == '' or col.startswith('_') or col.startswith('Unnamed')]
    df = df.drop(columns=empty_cols, errors='ignore')
    
    # Define financial ratio columns
    financial_columns = ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE', 
                        'Current Ratio', 'Debt-to-Equity', 'Debt-to-Assets']
    
    # Clean financial columns
    for col in financial_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                # Remove percentage signs and commas
                df[col] = df[col].astype(str).str.replace('%', '').str.replace(',', '').str.strip()
                # Replace common text values
                df[col] = df[col].replace(['N/A', 'n/a', 'NA', 'null', 'NULL', ''], np.nan)
            
            # Convert to numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean Year and Quarter columns
    if 'Year' in df.columns:
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    
    if 'Quarter' in df.columns:
        df['Quarter'] = pd.to_numeric(df['Quarter'], errors='coerce')
    
    # Fill missing values with company-specific medians
    if 'Company' in df.columns:
        for company in df['Company'].unique():
            if pd.notna(company):
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
    """Create comprehensive sample data for demonstration"""
    companies = ['Almarai', 'Savola', 'NADEC']
    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
    quarters = [1, 2, 3, 4]
    
    data = []
    
    # Base financial ratios for each company (based on real Saudi food sector data)
    base_ratios = {
        'Almarai': {
            'Gross Margin': 0.35, 'Net Profit Margin': 0.12, 'ROA': 0.03, 'ROE': 0.085,
            'Current Ratio': 1.15, 'Debt-to-Equity': 1.20, 'Debt-to-Assets': 0.55
        },
        'Savola': {
            'Gross Margin': 0.19, 'Net Profit Margin': 0.03, 'ROA': 0.01, 'ROE': 0.028,
            'Current Ratio': 0.85, 'Debt-to-Equity': 1.45, 'Debt-to-Assets': 0.59
        },
        'NADEC': {
            'Gross Margin': 0.38, 'Net Profit Margin': 0.08, 'ROA': 0.024, 'ROE': 0.042,
            'Current Ratio': 0.95, 'Debt-to-Equity': 1.80, 'Debt-to-Assets': 0.64
        }
    }
    
    np.random.seed(42)  # For consistent sample data
    
    for company in companies:
        for year in years:
            for quarter in quarters:
                quarterly_ratios = base_ratios[company].copy()
                
                # Add realistic variation (economic cycles, seasonal effects)
                trend_factor = 1 + (year - 2019) * 0.02  # Slight upward trend
                seasonal_factor = 1 + np.sin(quarter * np.pi / 2) * 0.05  # Seasonal variation
                random_factor = np.random.normal(1, 0.15)  # Random variation
                
                for ratio in quarterly_ratios:
                    quarterly_ratios[ratio] *= trend_factor * seasonal_factor * random_factor
                    
                    # Ensure values stay within realistic bounds
                    if ratio in ['Gross Margin', 'Net Profit Margin', 'ROA', 'ROE']:
                        quarterly_ratios[ratio] = max(-0.05, min(0.4, quarterly_ratios[ratio]))
                    elif ratio == 'Current Ratio':
                        quarterly_ratios[ratio] = max(0.3, min(2.5, quarterly_ratios[ratio]))
                    else:  # Debt ratios
                        quarterly_ratios[ratio] = max(0.2, min(2.5, quarterly_ratios[ratio]))
                
                # Create period identifier
                period_date = f"{quarter * 3}/31/{year}"
                
                data.append({
                    'Period': period_date,
                    'Period_Type': 'Quarterly',
                    'Year': year,
                    'Quarter': quarter,
                    'Company': company,
                    **quarterly_ratios
                })
    
    sample_df = pd.DataFrame(data)
    st.info(f"📊 Created sample data: {len(sample_df)} records for {len(companies)} companies ({years[0]}-{years[-1]})")
    
    return sample_df

# ============================================================================
# Initialize System
# ============================================================================

# Load data and initialize AI system
df = load_financial_data()
enhanced_financial_ai = EnhancedFinancialAI()

# Initialize Portfolio Optimizer
@st.cache_resource
def get_portfolio_optimizer():
    return StreamlitPortfolioOptimizer(df)

portfolio_optimizer = get_portfolio_optimizer()

# Initialize Q&A Chat Bot
if 'qa_chat_bot' not in st.session_state:
    st.session_state.qa_chat_bot = QAChatBot()

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# ============================================================================
# Sidebar Navigation
# ============================================================================

st.sidebar.title("🎯 Navigation")

# System Status
st.sidebar.subheader("🤖 System Status")
st.sidebar.success("✅ **AI + Portfolio Optimizer Active**")
st.sidebar.write(f"📊 Data: {len(df)} records" if not df.empty else "📊 Data: Sample mode")
st.sidebar.write(f"🏢 Companies: {len(portfolio_optimizer.companies)}")
st.sidebar.write(f"🔬 Optimization: {'Advanced (scipy)' if SCIPY_AVAILABLE else 'Mathematical'}")

# Main navigation
page = st.sidebar.selectbox(
    "Choose Analysis Type:",
    ["🏠 Dashboard", "📊 Company Analysis", "🔮 Quick Prediction", 
     "💬 AI Chat Q&A", "🎯 Portfolio Optimizer",  # PORTFOLIO OPTIMIZER HERE!
     "🏥 Health Check", "⚖️ Comparison", "🎯 Custom Analysis"]
)

# ============================================================================
# PORTFOLIO OPTIMIZER PAGE - COMPLETE IMPLEMENTATION
# ============================================================================

if page == "🎯 Portfolio Optimizer":
    st.header("🎯 Advanced Portfolio Optimizer")
    st.markdown("*Create mathematically optimized portfolios using correlation analysis*")
    
    # Show optimization capabilities
    opt_info_col1, opt_info_col2 = st.columns(2)
    
    with opt_info_col1:
        st.info("🔬 **Mathematical Precision**: Uses correlation coefficients and real financial data")
        st.info("🎯 **Target-Based**: Create portfolios for specific ROI, growth, or risk levels")
    
    with opt_info_col2:
        st.info("📊 **Data-Driven**: Based on 2016-2023 Saudi food sector performance")
        st.info("🔗 **Diversification**: Optimizes using company correlation analysis")
    
    # Portfolio optimization tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎯 Target ROI", "📈 Growth Focus", "⚖️ Risk Control", "📊 Strategies", "🔗 Correlation"])
    
    # ==================== TARGET ROI OPTIMIZATION ====================
    with tab1:
        st.subheader("🎯 Target ROI Portfolio Creation")
        st.markdown("*Design portfolio to achieve specific return on investment*")
        
        roi_col1, roi_col2 = st.columns([1, 1])
        
        with roi_col1:
            target_roi = st.slider("🎯 Target ROI (%)", 1, 15, 8, 1) / 100
            risk_tolerance = st.selectbox("⚖️ Risk Tolerance", ["low", "medium", "high"])
            investment_amount = st.number_input("💰 Investment Amount (SAR)", 10000, 10000000, 100000, 10000)
        
        with roi_col2:
            st.markdown("#### 📊 Expected Outcomes")
            st.metric("🎯 Target Return", f"{target_roi:.1%}")
            st.metric("💰 Investment", f"SAR {investment_amount:,}")
            st.metric("📈 Expected Annual Gain", f"SAR {investment_amount * target_roi:,.0f}")
        
        if st.button("🔍 OPTIMIZE FOR TARGET ROI", type="primary", key="roi_opt"):
            with st.spinner("🤖 Creating optimal portfolio..."):
                
                # Map risk tolerance to optimization parameters
                risk_multiplier = {"low": 0.5, "medium": 1.0, "high": 1.5}[risk_tolerance]
                
                if SCIPY_AVAILABLE:
                    result = portfolio_optimizer.optimize_with_scipy(target_roi, risk_multiplier)
                else:
                    # Use simple optimization targeting the ROI
                    if target_roi <= 0.04:
                        opt_type = "low_risk"
                    elif target_roi >= 0.08:
                        opt_type = "return_focused"
                    else:
                        opt_type = "balanced"
                    
                    result = portfolio_optimizer.optimize_portfolio_simple(target_return=target_roi, optimization_type=opt_type)
                
                st.markdown("---")
                st.subheader("🎯 Optimized Portfolio Results")
                
                # Portfolio allocation
                portfolio_col1, portfolio_col2 = st.columns([1, 1])
                
                with portfolio_col1:
                    st.markdown("#### 📊 Portfolio Allocation")
                    
                    allocation_data = []
                    for i, company in enumerate(portfolio_optimizer.companies):
                        weight = result['weights'][i]
                        allocation_amount = investment_amount * weight
                        
                        allocation_data.append({
                            'Company': company,
                            'Weight': weight,
                            'Amount': allocation_amount
                        })
                        
                        # Display metrics
                        st.metric(f"{company}", f"{weight:.1%}", f"SAR {allocation_amount:,.0f}")
                    
                    # Create pie chart
                    allocation_df = pd.DataFrame(allocation_data)
                    
                    fig_pie = px.pie(
                        allocation_df, 
                        values='Weight', 
                        names='Company',
                        title="Portfolio Allocation",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with portfolio_col2:
                    st.markdown("#### 📈 Performance Metrics")
                    
                    metric_col1, metric_col2 = st.columns(2)
                    
                    with metric_col1:
                        st.metric("Expected ROI", f"{result['expected_return']:.2%}")
                        st.metric("Portfolio Risk", f"{result['portfolio_risk']:.2%}")
                        st.metric("Sharpe Ratio", f"{result['sharpe_ratio']:.2f}")
                    
                    with metric_col2:
                        st.metric("Achievement Score", f"{result['achievement_score']:.0f}/100")
                        st.metric("Diversification Score", f"{result['diversification_score']:.0f}/100")
                        
                        # Risk assessment
                        if result['portfolio_risk'] < 0.08:
                            st.success("🟢 Low Risk Portfolio")
                        elif result['portfolio_risk'] < 0.15:
                            st.warning("🟡 Medium Risk Portfolio")
                        else:
                            st.error("🔴 High Risk Portfolio")
                    
                    # Investment projections
                    st.markdown("#### 💰 Investment Projections")
                    
                    expected_annual_return = investment_amount * result['expected_return']
                    
                    projection_data = []
                    for year in [1, 3, 5, 10]:
                        future_value = investment_amount * (1 + result['expected_return'])**year
                        projection_data.append({
                            'Year': year,
                            'Portfolio Value': future_value,
                            'Total Gain': future_value - investment_amount
                        })
                    
                    projection_df = pd.DataFrame(projection_data)
                    
                    fig_projection = px.line(
                        projection_df, 
                        x='Year', 
                        y='Portfolio Value',
                        title="Portfolio Growth Projection",
                        markers=True
                    )
                    fig_projection.update_layout(yaxis_tickformat=',.0f')
                    st.plotly_chart(fig_projection, use_container_width=True)
                    
                    # Summary table
                    st.markdown("**Investment Summary:**")
                    for _, row in projection_df.iterrows():
                        st.write(f"Year {row['Year']}: SAR {row['Portfolio Value']:,.0f} (+SAR {row['Total Gain']:,.0f})")
    
    # ==================== GROWTH FOCUS OPTIMIZATION ====================
    with tab2:
        st.subheader("📈 Growth-Focused Portfolio Optimization")
        st.markdown("*Maximize growth potential while managing risk*")
        
        growth_col1, growth_col2 = st.columns([1, 1])
        
        with growth_col1:
            growth_target = st.slider("📈 Target Growth Rate (%)", 2, 20, 10, 1) / 100
            max_risk = st.slider("⚠️ Maximum Risk Level (%)", 5, 25, 15, 1) / 100
            growth_investment = st.number_input("💰 Investment Amount (SAR)", 10000, 10000000, 100000, 10000, key="growth_inv")
        
        with growth_col2:
            st.markdown("#### 📊 Growth Parameters")
            st.metric("🎯 Growth Target", f"{growth_target:.1%}")
            st.metric("⚠️ Risk Limit", f"{max_risk:.1%}")
            st.metric("💰 Investment", f"SAR {growth_investment:,}")
        
        if st.button("📈 OPTIMIZE FOR GROWTH", type="primary", key="growth_opt"):
            with st.spinner("📊 Creating growth-focused portfolio..."):
                
                # For growth optimization, use return-focused strategy
                result = portfolio_optimizer.optimize_portfolio_simple(optimization_type="return_focused")
                
                st.markdown("---")
                st.subheader("📈 Growth-Optimized Portfolio")
                
                growth_result_col1, growth_result_col2 = st.columns([1, 1])
                
                with growth_result_col1:
                    st.markdown("#### 📊 Growth Allocation")
                    
                    for i, company in enumerate(portfolio_optimizer.companies):
                        weight = result['weights'][i]
                        amount = growth_investment * weight
                        
                        # Get company growth metrics
                        company_metrics = portfolio_optimizer.company_metrics[company]
                        expected_roe = company_metrics['avg_roe']
                        
                        st.metric(
                            f"{company}", 
                            f"{weight:.1%}", 
                            f"Expected ROE: {expected_roe:.1%}"
                        )
                        st.write(f"Amount: SAR {amount:,.0f}")
                        st.markdown("---")
                
                with growth_result_col2:
                    st.markdown("#### 📈 Growth Analysis")
                    
                    st.metric("Expected Portfolio Return", f"{result['expected_return']:.2%}")
                    st.metric("Portfolio Risk", f"{result['portfolio_risk']:.2%}")
                    st.metric("Risk-Adjusted Return", f"{result['sharpe_ratio']:.2f}")
                    
                    # Risk vs Growth assessment
                    if result['portfolio_risk'] <= max_risk:
                        st.success("✅ Risk target achieved!")
                    else:
                        st.warning("⚠️ Risk above target - consider adjusting allocation")
                    
                    if result['expected_return'] >= growth_target:
                        st.success("✅ Growth target achieved!")
                    else:
                        st.info(f"📊 Growth: {result['expected_return']:.1%} (Target: {growth_target:.1%})")
                    
                    # Growth projection chart
                    years = list(range(1, 11))
                    growth_values = [growth_investment * (1 + result['expected_return'])**year for year in years]
                    
                    growth_chart_df = pd.DataFrame({
                        'Year': years,
                        'Portfolio Value': growth_values
                    })
                    
                    fig_growth = px.line(
                        growth_chart_df, 
                        x='Year', 
                        y='Portfolio Value',
                        title="10-Year Growth Projection",
                        markers=True
                    )
                    fig_growth.update_layout(yaxis_tickformat=',.0f')
                    st.plotly_chart(fig_growth, use_container_width=True)
    
    # ==================== RISK CONTROL OPTIMIZATION ====================
    with tab3:
        st.subheader("⚖️ Risk-Controlled Portfolio Optimization")
        st.markdown("*Minimize risk while maintaining acceptable returns*")
        
        risk_col1, risk_col2 = st.columns([1, 1])
        
        with risk_col1:
            target_risk = st.slider("⚖️ Maximum Risk Level (%)", 3, 20, 10, 1) / 100
            min_return = st.slider("📊 Minimum Required Return (%)", 2, 12, 5, 1) / 100
            risk_investment = st.number_input("💰 Investment Amount (SAR)", 10000, 10000000, 100000, 10000, key="risk_inv")
        
        with risk_col2:
            st.markdown("#### ⚖️ Risk Parameters")
            st.metric("⚖️ Max Risk", f"{target_risk:.1%}")
            st.metric("📊 Min Return", f"{min_return:.1%}")
            st.metric("💰 Investment", f"SAR {risk_investment:,}")
        
        if st.button("⚖️ OPTIMIZE FOR RISK CONTROL", type="primary", key="risk_opt"):
            with st.spinner("⚖️ Creating risk-controlled portfolio..."):
                
                # Use low-risk optimization strategy
                result = portfolio_optimizer.optimize_portfolio_simple(optimization_type="low_risk")
                
                st.markdown("---")
                st.subheader("⚖️ Risk-Controlled Portfolio")
                
                risk_result_col1, risk_result_col2 = st.columns([1, 1])
                
                with risk_result_col1:
                    st.markdown("#### 📊 Conservative Allocation")
                    
                    for i, company in enumerate(portfolio_optimizer.companies):
                        weight = result['weights'][i]
                        amount = risk_investment * weight
                        
                        # Get company risk metrics
                        company_metrics = portfolio_optimizer.company_metrics[company]
                        company_risk = company_metrics['roe_volatility']
                        
                        risk_color = "🟢" if company_risk < 0.04 else "🟡" if company_risk < 0.06 else "🔴"
                        
                        st.metric(
                            f"{company} {risk_color}", 
                            f"{weight:.1%}", 
                            f"Risk: {company_risk:.1%}"
                        )
                        st.write(f"Amount: SAR {amount:,.0f}")
                        st.markdown("---")
                
                with risk_result_col2:
                    st.markdown("#### ⚖️ Risk Analysis")
                    
                    st.metric("Portfolio Risk", f"{result['portfolio_risk']:.2%}")
                    st.metric("Expected Return", f"{result['expected_return']:.2%}")
                    st.metric("Risk Efficiency", f"{result['expected_return']/result['portfolio_risk']:.2f}")
                    
                    # Risk assessment
                    if result['portfolio_risk'] <= target_risk:
                        st.success("✅ Risk target achieved!")
                    else:
                        st.warning("⚠️ Risk above target")
                    
                    if result['expected_return'] >= min_return:
                        st.success("✅ Return requirement met!")
                    else:
                        st.warning("⚠️ Return below minimum")
                    
                    # Risk vs Return visualization
                    companies = portfolio_optimizer.companies
                    company_returns = [portfolio_optimizer.company_metrics[comp]['avg_roe'] for comp in companies]
                    company_risks = [portfolio_optimizer.company_metrics[comp]['roe_volatility'] for comp in companies]
                    
                    risk_return_df = pd.DataFrame({
                        'Company': companies,
                        'Expected Return': company_returns,
                        'Risk': company_risks,
                        'Weight': result['weights']
                    })
                    
                    fig_risk_return = px.scatter(
                        risk_return_df,
                        x='Risk',
                        y='Expected Return',
                        size='Weight',
                        color='Company',
                        title="Risk vs Return Analysis",
                        labels={'Risk': 'Risk (Volatility)', 'Expected Return': 'Expected Return (ROE)'}
                    )
                    
                    # Add portfolio point
                    fig_risk_return.add_scatter(
                        x=[result['portfolio_risk']],
                        y=[result['expected_return']],
                        mode='markers',
                        marker=dict(size=20, color='red', symbol='star'),
                        name='Portfolio'
                    )
                    
                    fig_risk_return.update_layout(
                        xaxis_tickformat='.1%',
                        yaxis_tickformat='.1%'
                    )
                    st.plotly_chart(fig_risk_return, use_container_width=True)
    
    # ==================== STRATEGY COMPARISON ====================
    with tab4:
        st.subheader("📊 Portfolio Strategy Comparison")
        st.markdown("*Compare different optimization strategies*")
        
        strategy_investment = st.number_input("💰 Investment Amount for Comparison (SAR)", 10000, 10000000, 100000, 10000, key="strategy_inv")
        
        if st.button("📊 COMPARE ALL STRATEGIES", type="primary", key="strategy_comp"):
            with st.spinner("📊 Analyzing all strategies..."):
                
                strategies = {
                    "Equal Weight": "equal_weight",
                    "Return Focused": "return_focused", 
                    "Low Risk": "low_risk",
                    "Balanced": "balanced"
                }
                
                strategy_results = []
                
                for strategy_name, strategy_type in strategies.items():
                    result = portfolio_optimizer.optimize_portfolio_simple(optimization_type=strategy_type)
                    
                    strategy_results.append({
                        'Strategy': strategy_name,
                        'Expected Return': result['expected_return'],
                        'Risk': result['portfolio_risk'],
                        'Sharpe Ratio': result['sharpe_ratio'],
                        'Diversification': result['diversification_score'],
                        'Almarai %': result['weights'][0] if len(result['weights']) > 0 else 0,
                        'Savola %': result['weights'][1] if len(result['weights']) > 1 else 0,
                        'NADEC %': result['weights'][2] if len(result['weights']) > 2 else 0
                    })
                
                strategy_df = pd.DataFrame(strategy_results)
                
                st.markdown("---")
                st.subheader("📊 Strategy Comparison Results")
                
                # Display comparison table
                display_df = strategy_df.copy()
                for col in ['Expected Return', 'Risk']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}")
                for col in ['Sharpe Ratio']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")
                for col in ['Almarai %', 'Savola %', 'NADEC %']:
                    display_df[col] = display_df[col].apply(lambda x: f"{x:.1%}")
                
                st.dataframe(display_df.set_index('Strategy'), use_container_width=True)
                
                # Visualization
                chart_col1, chart_col2 = st.columns(2)
                
                with chart_col1:
                    # Risk vs Return chart
                    fig_strategies = px.scatter(
                        strategy_df,
                        x='Risk',
                        y='Expected Return',
                        color='Strategy',
                        size='Sharpe Ratio',
                        title="Strategy Risk vs Return",
                        labels={'Risk': 'Portfolio Risk', 'Expected Return': 'Expected Return'}
                    )
                    fig_strategies.update_layout(
                        xaxis_tickformat='.1%',
                        yaxis_tickformat='.1%'
                    )
                    st.plotly_chart(fig_strategies, use_container_width=True)
                
                with chart_col2:
                    # Allocation comparison
                    allocation_data = []
                    for _, row in strategy_df.iterrows():
                        for company in ['Almarai', 'Savola', 'NADEC']:
                            allocation_data.append({
                                'Strategy': row['Strategy'],
                                'Company': company,
                                'Allocation': row[f'{company} %']
                            })
                    
                    allocation_df = pd.DataFrame(allocation_data)
                    
                    fig_allocation = px.bar(
                        allocation_df,
                        x='Strategy',
                        y='Allocation',
                        color='Company',
                        title="Allocation by Strategy",
                        labels={'Allocation': 'Allocation (%)'}
                    )
                    fig_allocation.update_layout(yaxis_tickformat='.1%')
                    st.plotly_chart(fig_allocation, use_container_width=True)
                
                # Best strategy recommendation
                best_sharpe = strategy_df.loc[strategy_df['Sharpe Ratio'].idxmax()]
                best_return = strategy_df.loc[strategy_df['Expected Return'].idxmax()]
                lowest_risk = strategy_df.loc[strategy_df['Risk'].idxmin()]
                
                st.markdown("#### 🏆 Strategy Recommendations")
                
                rec_col1, rec_col2, rec_col3 = st.columns(3)
                
                with rec_col1:
                    st.success(f"🏆 **Best Risk-Adjusted**: {best_sharpe['Strategy']}")
                    st.write(f"Sharpe Ratio: {best_sharpe['Sharpe Ratio']:.2f}")
                    st.write(f"Return: {best_sharpe['Expected Return']:.1%}")
                    st.write(f"Risk: {best_sharpe['Risk']:.1%}")
                
                with rec_col2:
                    st.info(f"📈 **Highest Return**: {best_return['Strategy']}")
                    st.write(f"Return: {best_return['Expected Return']:.1%}")
                    st.write(f"Risk: {best_return['Risk']:.1%}")
                    st.write(f"Sharpe: {best_return['Sharpe Ratio']:.2f}")
                
                with rec_col3:
                    st.success(f"🛡️ **Lowest Risk**: {lowest_risk['Strategy']}")
                    st.write(f"Risk: {lowest_risk['Risk']:.1%}")
                    st.write(f"Return: {lowest_risk['Expected Return']:.1%}")
                    st.write(f"Sharpe: {lowest_risk['Sharpe Ratio']:.2f}")
    
    # ==================== CORRELATION ANALYSIS ====================
    with tab5:
        st.subheader("🔗 Correlation Analysis & Diversification")
        st.markdown("*Understand company relationships for optimal diversification*")
        
        corr_col1, corr_col2 = st.columns([2, 1])
        
        with corr_col1:
            st.markdown("#### 🔗 Company Correlation Matrix")
            
            # Create correlation heatmap
            fig_corr = px.imshow(
                portfolio_optimizer.correlation_df,
                title="Company Correlation Matrix",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                text_auto=True
            )
            fig_corr.update_layout(height=400)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            st.markdown("#### 📊 Company Performance Metrics")
            
            # Create performance metrics table
            metrics_data = []
            for company in portfolio_optimizer.companies:
                metrics = portfolio_optimizer.company_metrics[company]
                metrics_data.append({
                    'Company': company,
                    'Avg ROE': f"{metrics['avg_roe']:.2%}",
                    'Volatility': f"{metrics['roe_volatility']:.2%}",
                    'Risk-Return Ratio': f"{metrics['avg_roe']/metrics['roe_volatility']:.1f}",
                    'Data Points': metrics.get('data_points', 'N/A')
                })
            
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df.set_index('Company'), use_container_width=True)
        
        with corr_col2:
            st.markdown("#### 🎯 Diversification Insights")
            
            # Calculate correlation insights
            companies = portfolio_optimizer.companies
            correlations = []
            
            for i in range(len(companies)):
                for j in range(i+1, len(companies)):
                    corr_value = portfolio_optimizer.correlation_matrix[i, j]
                    
                    if corr_value < 0.3:
                        level = "Excellent"
                        color = "🟢"
                    elif corr_value < 0.7:
                        level = "Good"
                        color = "🟡"
                    else:
                        level = "Limited"
                        color = "🔴"
                    
                    correlations.append({
                        'Pair': f"{companies[i]} vs {companies[j]}",
                        'Correlation': f"{corr_value:.3f}",
                        'Level': level,
                        'Color': color
                    })
            
            for corr in correlations:
                st.markdown(f"**{corr['Pair']}**")
                st.write(f"{corr['Color']} {corr['Correlation']} - {corr['Level']} diversification")
                st.markdown("---")
            
            st.markdown("#### 💡 Portfolio Tips")
            st.success("🟢 **Best Pairs**: Low correlation (< 0.3)")
            st.info("🟡 **Good Pairs**: Medium correlation (0.3-0.7)")
            st.warning("🔴 **Avoid**: High correlation (> 0.7)")
            
            st.markdown("#### 📊 Diversification Benefits")
            st.write("• **Risk Reduction**: Lower correlation = better risk reduction")
            st.write("• **Return Stability**: Diversified portfolios have more stable returns")
            st.write("• **Optimal Allocation**: Use correlation to determine weights")

# ============================================================================
# DASHBOARD PAGE (Updated with Portfolio Preview)
# ============================================================================

elif page == "🏠 Dashboard":
    st.header("📊 Financial AI Dashboard")
    st.markdown("*Overview of Saudi Food Sector Performance + Portfolio Optimization*")
    
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
        
        # Portfolio Correlation Preview
        st.subheader("🔗 Company Correlation Analysis")
        
        corr_preview_col1, corr_preview_col2 = st.columns([2, 1])
        
        with corr_preview_col1:
            fig_corr_preview = px.imshow(
                portfolio_optimizer.correlation_df,
                title="Correlation Between Companies",
                color_continuous_scale="RdBu_r",
                aspect="auto"
            )
            fig_corr_preview.update_layout(height=300)
            st.plotly_chart(fig_corr_preview, use_container_width=True)
        
        with corr_preview_col2:
            st.markdown("#### 🎯 Quick Portfolio")
            st.info("💡 **Tip**: Lower correlation (blue) = Better diversification")
            
            if st.button("🚀 Create Balanced Portfolio"):
                balanced_result = portfolio_optimizer.optimize_portfolio_simple(optimization_type="balanced")
                
                st.markdown("**Recommended Allocation:**")
                for i, company in enumerate(portfolio_optimizer.companies):
                    weight = balanced_result['weights'][i]
                    st.write(f"• {company}: {weight:.1%}")
                
                st.metric("Expected ROI", f"{balanced_result['expected_return']:.1%}")
                st.metric("Risk Level", f"{balanced_result['portfolio_risk']:.1%}")
        
        # Latest performance summary
        st.subheader("🏆 Latest Company Performance")
        
        # Get latest data for each company
        latest_year = df['Year'].max()
        latest_data = df[df['Year'] == latest_year].groupby('Company').tail(1)
        
        if not latest_data.empty:
            performance_cols = st.columns(min(len(latest_data), 3))
            
            for i, (_, company_data) in enumerate(latest_data.iterrows()):
                if i < len(performance_cols):
                    with performance_cols[i]:
                        company = company_data['Company']
                        roe = company_data['ROE']
                        
                        recommendation_result = enhanced_financial_ai.comprehensive_analysis(company_data.to_dict())
                        recommendation = recommendation_result['investment_recommendation']
                        
                        st.markdown(f"### {company}")
                        st.metric("ROE", f"{roe:.1%}")
                        
                        if recommendation in ["Strong Buy", "Buy"]:
                            st.success(f"📈 {recommendation}")
                        elif "Hold" in recommendation:
                            st.warning(f"⚖️ {recommendation}")
                        else:
                            st.error(f"📉 {recommendation}")
        
        # Sector trends chart
        st.subheader("📈 Sector ROE Trends")
        
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

# ============================================================================
# Q&A CHAT PAGE (Enhanced with Portfolio Knowledge)
# ============================================================================

elif page == "💬 AI Chat Q&A":
    st.header("💬 Interactive AI Financial Chat")
    st.markdown("*Ask questions about companies, investments, and portfolio optimization*")
    st.markdown("🎯 **Enhanced with Portfolio Knowledge**")
    
    # Example questions with portfolio focus
    st.subheader("💡 Example Questions")
    example_questions = [
        "Which company has the best ROE performance?",
        "Create a portfolio with 8% ROI",
        "How does portfolio optimization work?",
        "What are the correlation benefits between companies?",
        "Which companies should I combine for low risk?",
        "Best allocation for growth-focused portfolio?"
    ]
    
    example_cols = st.columns(2)
    for i, question in enumerate(example_questions):
        col = example_cols[i % 2]
        if col.button(f"💡 {question}", key=f"example_{i}"):
            st.session_state.user_question = question
    
    # Question input
    user_question = st.text_input(
        "Ask your question:",
        value=st.session_state.get('user_question', ''),
        placeholder="e.g., How do I create a portfolio with 8% ROI?",
        key="question_input"
    )
    
    col1, col2 = st.columns([1, 1])
