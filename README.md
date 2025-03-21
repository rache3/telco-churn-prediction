# Telco Customer Churn Prediction

## Project Overview

This project aims to predict customer churn for a telecommunications company using machine learning. The dataset contains customer demographics, account information, and service usage details. The goal is to develop a predictive model to help the company identify customers at risk of churning and take proactive measures.

## Dataset Information

- **Source**: Kaggle - Telco Customer Churn dataset
- **Features**: Customer demographics, service subscription details, monthly charges, tenure, contract type, payment method, and more.
- **Target Variable**: `Churn` (Binary: 1 = Churned, 0 = Not Churned)

## Methodology

1. **Data Preprocessing**
   - Handled missing values
   - Converted categorical variables to numerical using One-Hot Encoding
   - Scaled numerical features
   - **Feature Engineering:**
     - Created a new feature `SeniorGroup` by categorizing `SeniorCitizen` into "Older Generation" and "Younger Generation"
     - Transformed `Churn` into "Churned" and "Not Churned" for better interpretability
2. **Exploratory Data Analysis (EDA)**
   - Analyzed feature distributions and correlations
   - Visualized churn rates across different customer segments
3. **Model Selection and Training**
   - Used `XGBoost` classifier for prediction
   - Performed hyperparameter tuning with `RandomizedSearchCV`
4. **Model Evaluation**
   - Metrics: Accuracy, ROC-AUC Score, Precision, Recall, F1-Score
   - Feature Importance analysis
5. **Testing on Unseen Data**
   - Evaluated model performance on the test set

## Results
The model was evaluated using accuracy, ROC-AUC, and a confusion matrix. Key insights show high recall but a high false positive rate, suggesting room for optimization.
- **Training Accuracy**: 0.7963
- **Test Accuracy**: 0.7906
- **ROC-AUC Score**: 0.6679
- **Key Features** influencing churn: Tenure, Monthly Charges, Contract Type, Payment Method

## How to Run the Code

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Script

```bash
python src/train_model.py
```

### 3. Make Predictions

```python
from src.predict import predict_churn
predictions = predict_churn(new_data)
```

## Repository Structure

```
├── data/              # (Optional) Sample dataset
├── notebooks/         # Jupyter notebooks for EDA & modeling
├── src/               # Python scripts for preprocessing & model training
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate.py
│   ├── predict.py
├── models/            # Trained models
├── README.md          # Project documentation
├── requirements.txt   # Dependencies
├── .gitignore         # Files to exclude from Git
```

## Future Improvements

- Optimize threshold tuning to balance precision and recall
- Experiment with deep learning models
- Deploy the model as an API for real-time predictions

## Author

Rachel A.

## License

MIT License

