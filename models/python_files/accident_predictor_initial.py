# -*- coding: utf-8 -*-
"""Accident_Predictor_initial.ipynb

**Install libraries**
"""

!pip install opendatasets -q

!pip install catboost

"""**Import libraries**"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pandas as pd
import opendatasets as od

"""Categorical feature preprocessing libraries"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

"""Random forest based feature selection libraries"""

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

"""Cross validation libraries"""

from sklearn.model_selection import(
    StratifiedKFold,
    cross_validate
)

"""SMOTE sampling libraries"""

from imblearn.over_sampling import SMOTE

"""Standared evaluation matrics libraries"""

from sklearn.metrics import make_scorer
from sklearn.metrics import(
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

"""ML models libraries"""

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

"""**Download dataset - US_Accidents_March23(https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents)**"""

od.download("https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents")

df = pd.read_csv("/content/us-accidents/US_Accidents_March23.csv",
                 nrows = 5000000
                 )

"""**Dataset analysis**"""

df.head()

df.info()

"""**Data preprocessing**

**Data cleaning**
"""

print(df.isnull().sum())

(df == 0).sum()

df_cleaned = df.dropna()

print("Cleaned Dataset Shape: ", df_cleaned.shape)

print("\nRemaining Null Values:")
print(df_cleaned.isnull().sum())

df_cleaned.to_csv('/content/cleaned_dataset.csv', index=False)

"""**Prepare external dataset**"""

features = [
    'ID',
    'Source',
    'Severity',
    'Start_Time',
    'End_Time',
    'Start_Lat',
    'Start_Lng',
    'Distance(mi)',
    'Street',
    'City',
    'County',
    'State',
    'Country',
    'Weather_Timestamp',
    'Temperature(F)',
    'Wind_Chill(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Direction',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Weather_Condition',
    'Bump',
    'Crossing',
    'Give_Way',
    'Junction',
    'No_Exit',
    'Railway',
    'Roundabout',
    'Station',
    'Stop',
    'Traffic_Calming',
    'Traffic_Signal',
    'Turning_Loop',
    'Sunrise_Sunset'
]

df_final = df_cleaned[features]

output_file = "accident_final.csv"

df_final.to_csv(output_file, index= False)

df_final.head()

df_final.info()

print(df_final.isnull().sum())

(df_final == 0).sum()

"""**Target values assigning**"""

df_final['Risk_Level'] = (
    df_final['Severity'] >= 3
).astype(int)

print(df_final['Risk_Level'].value_counts())

"""Dataset is imbalanced

**Feature extraction under data preprocessing part**
"""

df_final['Start_Time'] = pd.to_datetime(
    df_final['Start_Time'],
    format='mixed',
    errors='coerce'
)

df_final['Hour'] = df_final['Start_Time'].dt.hour

df_final['Day_of_Week'] = df_final['Start_Time'].dt.dayofweek

df_final['Is_Weekend'] = (
    df_final['Day_of_Week'] >= 5
).astype(int)

df_final['Is_Night'] = (
    (df_final['Hour'] >= 18) |
    (df_final['Hour'] <= 6)
).astype(int)

df_final['Weather_Condition'] = (
    df_final['Weather_Condition']
    .astype(str)
    .str.lower()
)

df_final['Rain_Flag'] = df_final['Weather_Condition'].str.contains('rain').astype(int)
df_final['Fog_Flag'] = df_final['Weather_Condition'].str.contains('fog').astype(int)
df_final['Snow_Flag'] = df_final['Weather_Condition'].str.contains('snow').astype(int)

df_final.info()

boolean_features = [
    'Bump',
    'Crossing',
    'Give_Way',
    'Junction',
    'No_Exit',
    'Railway',
    'Roundabout',
    'Station',
    'Stop',
    'Traffic_Calming',
    'Traffic_Signal',
    'Turning_Loop'
]

df_final[boolean_features] = (
    df_final[boolean_features]
    .astype(int)
)

print(df_final[boolean_features].head())

df_final.info()

"""**Manual feature selection**"""

final_features = [
    'Hour',
    'Day_of_Week',
    'Is_Weekend',
    'Is_Night',
    'Rain_Flag',
    'Fog_Flag',
    'Snow_Flag',
    'Start_Lat',
    'Start_Lng',
    'Distance(mi)',
    'Temperature(F)',
    'Wind_Chill(F)',
    'Humidity(%)',
    'Pressure(in)',
    'Visibility(mi)',
    'Wind_Speed(mph)',
    'Precipitation(in)',
    'Bump',
    'Crossing',
    'Give_Way',
    'Junction',
    'No_Exit',
    'Railway',
    'Roundabout',
    'Station',
    'Stop',
    'Traffic_Calming',
    'Traffic_Signal',
    'Turning_Loop'
]

X = df_final[final_features]
y = df_final['Risk_Level']

output1_file = "after_preprocessing.csv"

df_final.to_csv(output1_file, index= False)

print(X.isnull().sum())

print(X.dtypes)

"""**Random forest based feature selection**"""

model = RandomForestClassifier(
    n_estimators=25,
    random_state=42
)

model.fit(X, y)

"""**Check feature importance**"""

importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
})

importance = importance.sort_values(
    by='Importance',
    ascending=False
)

print(importance)

plt.figure(figsize=(10,6))

sns.barplot(
    x='Importance',
    y='Feature',
    data=importance
)

plt.title(
    'RoadVision AI Feature Importance'
)

plt.show()

"""**Cross Validation Setup**"""

cv = StratifiedKFold(
    n_splits = 5,
    shuffle = True,
    random_state = 42
)

"""**ML Models**

XGBoost, Random Forest, LightGBM, CatBoost, Losistic Regression, KNN, SVM, Neural Network
"""

models = {
    "XGBoost": XGBClassifier(
        n_estimators = 300,
        learning_rate = 0.05,
        max_depth = 8,
        subsample = 0.8,
        colsample_bytree = 0.8,
        random_state = 42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators = 300,
        max_depth = 20,
        random_state = 42,
        n_jobs = -1
    ),

    "LightGBM" : LGBMClassifier(
        iterations = 300,
        learning_rate = 0.05,
        max_depth = 8,
        random_state = 42
    ),

    "CatBoost": CatBoostClassifier(
        iterations = 300,
        learning_rate = 0.05,
        depth = 8,
        verbose = 0
    ),

    "Logistic Regression" : LogisticRegression(
        max_iter = 1000
    ),

    "KNN" : KNeighborsClassifier(
        n_neighbors = 7
    ),

    "SVM" : SVC(
        kernel = 'rbf'
    ),

    "Neural Network": MLPClassifier(
        hidden_layer_sizes = (128, 64),
        max_iter = 200,
        random_state = 42
    )


}

"""**Evalution Metrics**"""

scoring = {
    'accuracy': 'accuracy',
    'precision': 'precision_weighted',
    'recall': 'recall_weighted',
    'f1': 'f1_weighted'
}

"""**Cross Validation**"""

results = []

for name, model in models.items():

    print(f"\nTraining {name}")

    pipeline = ImbPipeline([

        ('smote',
         SMOTE(random_state=42)),

        ('model',
         model)

    ])

    scores = cross_validate(

        pipeline,
        X,
        y,

        cv=cv,

        scoring=scoring,

        n_jobs=-1
    )

    results.append({

        'Model': name,

        'Accuracy':
            scores[
                'test_accuracy'
            ].mean(),

        'Precision':
            scores[
                'test_precision'
            ].mean(),

        'Recall':
            scores[
                'test_recall'
            ].mean(),

        'F1 Score':
            scores[
                'test_f1'
            ].mean()
    })

"""**Compare Results**"""

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    by = 'F1 Score',
    ascending = False
)

print(results_df)