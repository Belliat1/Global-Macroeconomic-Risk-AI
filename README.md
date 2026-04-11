# 🌍 Phase 1: Global Geopolitical Risk & Macroeconomic EDA

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Pandas](https://img.shields.io/badge/Pandas-Data_Engineering-150458.svg)
![Plotly](https://img.shields.io/badge/Plotly-Interactive_Charts-3F4F75.svg)

## 📌 Business Context
Following the simulated geopolitical escalation in February 2026 (US-Iran conflict) and the subsequent restrictions in the Strait of Hormuz, global energy supply chains experienced a severe disruption affecting ~20% of the world's daily oil supply. 

This repository contains **Phase 1** of a broader Crisis Management System: a comprehensive Exploratory Data Analysis (EDA) and Data Engineering pipeline designed to quantify the cascading macroeconomic effects of crude oil price surges across 14 key global economies.

## 🏗️ Notebook Highlights (`01_eda_petrol_risk_analysis.ipynb`)
This notebook demonstrates professional data manipulation and visualization workflows:
* **Robust Data Ingestion:** Standardizing schemas (snake_case) and performing relational joins between micro-level retail pricing datasets and macro-level GDP impact datasets.
* **Feature Engineering:** Cleaning and consolidating raw data into a "Gold Layer" master dataset (`processed_macro_risk.csv`) ready for downstream Machine Learning and LLM integration.
* **High-Impact Visualization:** Building an interactive **Macroeconomic Risk Matrix** using `Plotly` to map the correlation between Retail Fuel Price Surges and Projected GDP Impact, classifying countries by vulnerability thresholds.

## 📊 Key Executive Insights
Based on the Quadrant Analysis, the global macroeconomic response is categorized into three systemic profiles:
1. **Severe Stagflation Risk (The Vulnerable):** Countries like **Pakistan** and **Iran** are in the critical zone (>20% retail price surge correlating with severe GDP contraction approaching -2.0% or worse).
2. **Strategic Resilience (The Giants):** **India** and **China** demonstrate massive population exposure but surprisingly low retail price surges, indicating aggressive government market intervention to shield consumers.
3. **The Windfall Economies (Net Exporters):** **Saudi Arabia** and the **UAE** act as a natural macroeconomic hedge, showing positive projected GDP growth despite the global supply shock.

## ⚙️ How to Run
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed (`pandas`, `numpy`, `plotly`).
3. Run the `01_eda_petrol_risk_analysis.ipynb` notebook to generate the interactive Plotly matrix and export the clean data layer for ML modeling.

---
*Built as a foundation for predictive AI modeling.*
