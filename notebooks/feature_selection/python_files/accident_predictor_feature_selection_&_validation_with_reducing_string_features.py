# -*- coding: utf-8 -*-
"""Accident_Predictor_Feature_Selection & Validation.ipynb
"""

!pip install opendatasets -q

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pandas as pd
import opendatasets as od

od.download("https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents")

df = pd.read_csv("/content/us-accidents/US_Accidents_March23.csv",
                 nrows = 5000000
                 )

df.head()

df.info()

print(df.isnull().sum())

(df == 0).sum()

df_cleaned = df.dropna()

print("Cleaned Dataset Shape: ", df_cleaned.shape)

print("\nRemaining Null Values:")
print(df_cleaned.isnull().sum())

df_cleaned.to_csv('/content/cleaned_dataset.csv', index=False)

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

df_final['Risk_Level'] = (
    df_final['Severity'] >= 3
).astype(int)

print(df_final['Risk_Level'].value_counts())

df_final.info()

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

import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

print(X.dtypes)

model = RandomForestClassifier(
    n_estimators=10,
    random_state=42
)

model.fit(X, y)

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