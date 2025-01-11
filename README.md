# Cashflow Forecast Project

## Overview
The **Cashflow Forecast** project is a Python-based tool designed to process financial data, apply predictive modeling techniques, and visualize results. It integrates ARIMA, Random Forest, and LSTM models to generate forecasts and analyze cash flow trends over time. Potential use cases include helping organizations optimize working capital, evaluate investment opportunities, and predict liquidity needs. For example, this tool aligns with advanced financial modeling approaches similar to those explored in the context of my Master's thesis on using data-driven techniques for forecasting and strategic decision-making in dynamic environments.

## Features
- **Data Processing**: Clean and prepare raw financial data.
- **Modeling**:
  - ARIMA for time-series analysis.
  - Random Forest for ensemble learning.
  - LSTM for sequence modeling.
- **Visualization**: Create graphs and output tables for better insights.

---

## Requirements
Ensure you have the following installed:

- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - matplotlib
  - statsmodels
  - scikit-learn
  - tensorflow
  - pmdarima

To install dependencies, run:
```bash
pip install -r requirements.txt
```

---

## Setup Guide

### Uploading Your Data
- Users should upload their raw financial data files into the `data` directory located in the root of the project.
- Data should be in CSV format for compatibility with the provided scripts.
- **Expected Data Schema**:
  - `Date` (format: YYYY-MM-DD): The date of the transaction.
  - `Transaction_ID` (string): A unique identifier for each transaction.
  - `Amount` (numeric): The transaction amount.
  - `Category` (string): The category or classification of the transaction (e.g., Sales, Expenses).
  - `Description` (string): A brief description of the transaction (optional).

Ensure the data is clean and free of missing or invalid entries to avoid processing errors. Refer to `data_processing/data_extraction.py` for detailed format requirements.
- Users should upload their raw financial data files into the `data` directory located in the root of the project.
- Data should be in CSV format for compatibility with the provided scripts.
- Ensure that the data includes necessary columns like date, transaction details, and amounts to allow preprocessing and modeling. Refer to `data_processing/data_extraction.py` for detailed format requirements.

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd CashflowForecast
   ```

2. **Prepare Your Data**:
   - Place your raw financial data in the `data` directory.
   - Ensure the data follows the expected structure (refer to the `data_processing/data_extraction.py` for specifics).

3. **Run Initializer**:
   Execute the initializer script to process data and run models:
   ```bash
   python initializer.py
   ```

---

## Workflow

### 1. Data Preparation
- **Input**: Raw data files in the `data` directory.
- **Process**: Run `data_processing/data_processing.py` to:
  - Anonymize sensitive IDs.
  - Clean and structure data for modeling.

### 2. Model Execution
- ARIMA:
  - Code: `models/arima_model.py`
  - Use for time-series forecasting with seasonal or trend components.
- Random Forest:
  - Code: `models/random_forest_model.py`
  - Ensemble-based learning for cash flow prediction.
- LSTM:
  - Code: `models/lstm_model.py`
  - Deep learning for sequence-based forecasting.

### 3. Visualization
- Graphs are automatically generated and stored in the `graphs` directory.
- Modify `graphs_creation.py` to customize plots.

### 4. Outputs
- Forecast tables are saved in `output_tables`.
- Errors and issues logged in `all_errors.xlsx`.

---

## File and Folder Structure

- **data**: Raw input files.
- **data_processing**:
  - `data_extraction.py`: Load and extract data.
  - `data_processing.py`: Process and prepare data.
- **models**:
  - `arima_model.py`: Implements ARIMA modeling.
  - `random_forest_model.py`: Implements Random Forest modeling.
  - `lstm_model.py`: Implements LSTM modeling.
- **graphs**: Stores generated visualizations.
- **initializer.py**: Runs the entire workflow.
- **test.py**: For testing functionalities.
- **all_errors.xlsx**: Logs errors encountered.

---

## Customization
To adapt this project to your data:
1. Modify the `data_processing.py` script to align with your data structure.
2. Update model parameters in the respective model scripts:
   - ARIMA: Seasonal parameters, p/d/q values.
   - Random Forest: Hyperparameters.
   - LSTM: Sequence length and architecture.
3. Adjust `graphs_creation.py` to customize visual outputs.

---

## Contribution
Contributions are welcome! Please submit pull requests or report issues via GitHub.

---

