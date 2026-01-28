# CodeAlpha_Credit-Scoring-Model
📊 Credit Scoring Model using Machine Learning
📌 Project Overview

This project focuses on building a Credit Scoring Model to predict a customer’s credit score category using machine learning techniques. Multiple classification algorithms are trained and evaluated to compare performance, with Random Forest achieving the best accuracy of 77%.

The project follows a standard data science workflow including data preprocessing, feature scaling, model training, and evaluation.

🧠 Problem Statement

Financial institutions need reliable models to assess customer creditworthiness. This project aims to:

Analyze credit-related data

Train classification models

Evaluate their predictive performance

🗂 Dataset

File: credit_score.csv

Target Variable: Credit_Score

Preprocessing Steps:

Removed unnecessary column (Unnamed: 0)

Checked missing values

Visualized missing data using a heatmap

Applied feature scaling using StandardScaler

⚙️ Technologies & Libraries Used

Python

NumPy

Pandas

Matplotlib

Seaborn

Scikit-learn

🔄 Workflow

Import required libraries

Load dataset

Data cleaning and exploration

Missing value analysis

Feature selection (X) and target (y)

Train-test split (80/20)

Feature scaling

Model training

Model evaluation

🤖 Models Implemented

The following machine learning models were trained and evaluated:

1️⃣ Logistic Regression

Used as a baseline model

Evaluated using classification report and accuracy

2️⃣ Random Forest Classifier ⭐

Ensemble learning method

Best performing model

Accuracy: 77%

3️⃣ Decision Tree Classifier

Tree-based model

Performance compared with other models

📈 Model Performance
Model	Accuracy

Logistic Regression	Evaluated

Decision Tree	Evaluated

Random Forest	77% ✅

🚀 How to Run the Project

1️⃣ Install dependencies

pip install numpy pandas matplotlib seaborn scikit-learn

2️⃣ Run the notebook

jupyter notebook Credit_Scoring_Model.ipynb

📁 Project Structure (Suggested)
Credit-Scoring-Model/


├── data/
   └── credit_score.csv


├── notebooks/
   └── Credit_Scoring_Model.ipynb


├── src/
   └── model_training.py


├── README.md
├── requirements.txt
└── Credit_Scoring_Model.txt

🎯 Conclusion

The Random Forest model outperformed other classifiers with an accuracy of 77%, making it the most suitable model for credit score prediction in this project. Further improvements can be achieved through hyperparameter tuning and feature engineering.

👤 Author

Mudasir Iqbal
📌 Machine Learning & Data Science Enthusiast
