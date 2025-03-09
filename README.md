# Loan Approval Prediction

## Overview
This project applies machine learning techniques to predict loan approval based on applicant details. The dataset consists of various features like income, loan amount, credit history, and personal details, and models are trained using SVM and Random Forest.

## Dataset
The dataset contains the following features:
- **Gender**: Male/Female
- **Married**: Yes/No
- **Dependents**: Number of dependents
- **Education**: Graduate/Not Graduate
- **Self_Employed**: Yes/No
- **ApplicantIncome**: Income of the applicant
- **CoapplicantIncome**: Income of the co-applicant
- **LoanAmount**: Loan amount applied for
- **Loan_Amount_Term**: Loan repayment term
- **Credit_History**: Credit history (1 = good, 0 = bad)
- **Property_Area**: Urban, Semi-Urban, Rural
- **Loan_Status**: Approved/Not Approved (Target variable)

## Data Preprocessing
1. **Handling Missing Values**:
   - Categorical values replaced with mode.
   - Numerical values replaced with median.
2. **Outlier Removal**:
   - IQR method applied to remove income outliers.
3. **Encoding Categorical Variables**:
   - Label encoding applied to categorical features.
   - One-hot encoding used for categorical variables in RandomForest model.
4. **Feature Scaling**:
   - StandardScaler applied to numerical features.

## Exploratory Data Analysis (EDA)
- Pie and bar charts for loan approval status.
- Income distribution analysis using histograms and box plots.
- Credit history vs. loan status visualization.
- Property area impact on loan approval.

## Machine Learning Models
### 1. Support Vector Classifier (SVC)
- Data split into training and test sets (80-20 ratio).
- Hyperparameter tuning using GridSearchCV.
- Model trained and tested.

### 2. Random Forest Classifier
- One-hot encoding applied to categorical features.
- Hyperparameter tuning using GridSearchCV.
- Model trained and tested.

## Performance Metrics
- **Accuracy Score**
- **Classification Report**
- **Confusion Matrix**

## Hyperparameter Tuning
- **SVC**: Tuned parameters include C, kernel, and gamma.
- **Random Forest**: Tuned n_estimators, max_depth, min_samples_split, min_samples_leaf, and max_features.

## GitHub Integration
- The project is version-controlled using GitHub.
- Commands included for configuring Git and pushing updates.

## How to Use
1. Clone the repository:
   ```bash
   git clone https://github.com/msr0b0tjennica/Loan-Approval-Prediction.git
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn plotly
   ```
3. Run the notebook to train and test the models.

## Future Improvements
- Implement deep learning models for better accuracy.
- Experiment with more feature engineering techniques.
- Deploy the model as a web application.

## Author
**Jennica Bhaskaran**
- GitHub: [msr0b0tjennica](https://github.com/msr0b0tjennica)
