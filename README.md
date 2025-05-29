# Financial AI Assistant - Saudi Food Sector

## 📜 Description

This project is a comprehensive Financial AI System designed for the analysis of companies in the Saudi food sector, specifically Savola, Almarai, and NADEC. It leverages historical financial data (2016-2023) to train AI models that provide insights into company performance, investment recommendations, and optimized portfolio strategies. The system culminates in an interactive Streamlit web application for easy access to these AI-powered financial tools.

## ✨ Key Features

* **Individual Company Analysis:**
    * Predicts Return on Equity (ROE).
    * Generates AI-based investment recommendations (e.g., Strong Buy, Buy, Hold, Sell).
    * Classifies company status (e.g., Excellent, Good, Average, Poor).
* **Portfolio Optimization:**
    * Offers multiple portfolio strategies (Conservative, Balanced, Growth, Aggressive/Risk-Adjusted Return) based on Modern Portfolio Theory and AI.
    * Calculates expected returns and risk levels for optimized portfolios.
* **Target-Based Portfolio Construction:**
    * Creates portfolios aiming for specific user-defined Return on Investment (ROI) targets.
* **Investment Calculator:**
    * Calculates potential investment returns and fund allocations for various investment amounts and portfolio strategies.
* **Interactive Web Application:**
    * A user-friendly Streamlit application provides access to all analysis and tools.

## 📊 Data Source

* **Dataset:** `Savola Almarai NADEC Financial Ratios CSV.csv`.
* **Companies:** Savola, Almarai, NADEC (3 Saudi food sector companies).
* **Records:** 96 financial data points covering the period from 2016 to 2023.
* **Data Quality:** The script processes the data to be 100% clean with no missing values for modeling.

## 🤖 AI Models Developed

The system trains and utilizes several AI models:

1.  **ROE Prediction Model:**
    * Algorithm: XGBoost Regressor.
    * Purpose: Predicts future Return on Equity.
    * Reported R² Score: 66.5%.
2.  **Investment Recommendation Model:**
    * Algorithm: Random Forest Classifier.
    * Purpose: Provides investment advice.
    * Reported Accuracy: 95%.
3.  **Company Status Classification Model:**
    * Algorithm: XGBoost Classifier.
    * Purpose: Assesses the overall financial health/status of a company.
    * Reported Accuracy: 100%.
4.  **Encoders:** LabelEncoders are used for categorical features like company name, investment recommendations, and company status.
5.  **Portfolio Optimizers:** Custom classes for `AdvancedPortfolioOptimizer` and `TargetBasedPortfolioOptimizer` implement portfolio construction logic.

*The .pkl files in this repository (e.g., `company_encoder.pkl`, `comprehensive_ratio_predictor.pkl`) are components or outputs of this system, such as saved encoders or predictive models.*

## 🛠️ Technologies & Libraries Used

* **Core:** Python [implied throughout]
* **Data Handling & Analysis:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost, LightGBM
* **Web Application:** Streamlit
* **Visualization:** Matplotlib, Seaborn, Plotly
* **Model Persistence:** Joblib

## 🚀 Getting Started

### Prerequisites

* Python (e.g., 3.8+)
* pip

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repository-url>
    cd <your-repository-name>
    ```
2.  Install the required packages:
    ```bash
    pip install pandas
