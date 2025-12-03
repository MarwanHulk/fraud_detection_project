                        Machine learning P2
NAME -> Marwan Mohamed
ID -> 10001890
# 🏥 Healthcare Provider Fraud Detection  

    PROJECT OVERVIEW
This project builds an end-to-end machine learning system to detect potentially fraudulent healthcare providers using Medicare claim data. The system integrates inpatient, outpatient, and beneficiary datasets, engineers provider-level features, trains multiple models, and produces fraud probability scores for each provider in the test dataset.
The workflow includes: multi-table data integration, feature engineering, class imbalance handling, model training and evaluation, and final prediction generation.
   
   SUMMARY OF RESULTS
Best Model: (Random Forest or Gradient Boosting, depending on your final Notebook 02 results).
Reason for selection: Highest Precision-Recall AUC and consistent ROC-AUC on imbalanced data.

Typical validation performance (replace with your real results):
	•	ROC-AUC: around 0.80–0.85
	•	PR-AUC: around 0.40–0.50
	•	Confusion matrix: moderate precision and moderate–high recall depending on threshold

Key insights:
	•	Fraudulent providers tend to have unusually high reimbursement amounts, high claim volumes, and many unique beneficiaries.
	•	Beneficiaries with many chronic conditions correlate with suspicious activity.
	•	Lower probability thresholds increase recall and reduce missed fraud cases.
Final output is a CSV file (provider_fraud_predictions.csv) containing fraud probabilities for all test providers.

REPRODUCTION INSTRUCTIONS
	1.	Install Dependencies
Use a Python virtual environment. Install required packages:
pip install pandas numpy scikit-learn matplotlib seaborn xgboost joblib reportlab python-pptx
	2.	Run Notebook 1: Feature Engineering
File: notebooks/01_data_exploration_and_feature_engineering.ipynb
This notebook loads raw training and test datasets, cleans them, merges inpatient, outpatient, and beneficiary data, and produces provider-level features.
Outputs saved:

	•	train_provider_features.csv
	•	test_provider_features.csv

	3.	Run Notebook 2: Modeling
File: notebooks/02_modeling.ipynb
This notebook handles missing values, imputes medians, scales data, splits into train/validation, trains multiple models (Logistic Regression, Random Forest, Gradient Boosting), evaluates them, and selects the best model based on PR-AUC.
    Saved outputs:

	•	best_model.pkl
	•	X_val.npy
	•	y_val.npy
	•	val_indices.npy

	4.	Run Notebook 3: Evaluation
File: notebooks/03_evaluation.ipynb
This notebook loads the saved model and validation data, generates the ROC curve, Precision-Recall curve, confusion matrix, performs threshold tuning, feature importance calculations, and false positive/false negative analysis.
	5.	Final Predictions
The best model is retrained on the full dataset and used to generate provider_fraud_predictions.csv, containing fraud probability for each test provider.