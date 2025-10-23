# 🚗 Road Accident Risk Prediction

This project focuses on predicting road accident risks using machine learning techniques. The model takes into account various road conditions, environmental factors, and temporal features to estimate accident risk levels.

## 📊 Dataset Overview

- **Training Data**: 517,754 samples
- **Features**: 13 input features + 1 target variable
- **Target Variable**: `accident_risk` (continuous)
- **Evaluation Metric**: Root Mean Squared Error (RMSE)

### Features Description

#### Categorical Features
- `road_type`: Type of road
- `lighting`: Lighting conditions
- `weather`: Weather conditions
- `time_of_day`: Time period of the day

#### Binary Features
- `road_signs_present`: Presence of road signs
- `public_road`: Whether it's a public road
- `holiday`: Holiday period
- `school_season`: School season

#### Numeric Features
- `num_lanes`: Number of lanes
- `curvature`: Road curvature
- `speed_limit`: Speed limit
- `num_reported_accidents`: Number of previously reported accidents

## 🔧 Data Preprocessing (`preprocessing.ipynb`)

1. **Categorical Feature Encoding**
   - One-hot encoding for categorical variables
   - Alignment of train and test features
   
2. **Feature Engineering**
   - Created `road_capacity` feature: `speed_limit * num_lanes`
   - Exported encoded datasets for modeling

## 📈 Exploratory Data Analysis (`exp-data-analysis.ipynb`)

- Analyzed feature distributions
- Visualized relationships between features and accident risk
- Created box plots to understand categorical feature impacts
- Confirmed no missing values in the dataset

## 🤖 Model Development (`model.ipynb`)

### Models Evaluated
1. Linear Regression (baseline)
2. Decision Tree Regressor
3. Random Forest Regressor
4. XGBoost Regressor

### Model Tuning
- Performed hyperparameter optimization using RandomizedSearchCV
- Focused on tuning XGBoost as the best performing model
- Parameters tuned include:
  - Number of estimators
  - Max depth
  - Learning rate
  - Subsample ratio
  - Column sample ratio
  - Min child weight

### Feature Importance
- Analyzed top 10 most influential features using XGBoost's feature importance

## 📝 Results
The final model (tuned XGBoost) was used to generate predictions on the test set, which were saved in the Kaggle submission format.

## 🛠️ Technologies Used
- Python
- pandas, numpy for data manipulation
- scikit-learn for preprocessing and modeling
- XGBoost for advanced gradient boosting
- seaborn, matplotlib for visualization

## 📦 Project Structure
```
├── exp-data-analysis.ipynb    # Exploratory data analysis
├── preprocessing.ipynb        # Data preprocessing and feature engineering
├── model.ipynb               # Model development and evaluation
├── encoded_data/             # Processed datasets
│   ├── train_encoded.csv
│   └── test_encoded.csv
└── playground-series-data/   # Raw datasets
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
```
