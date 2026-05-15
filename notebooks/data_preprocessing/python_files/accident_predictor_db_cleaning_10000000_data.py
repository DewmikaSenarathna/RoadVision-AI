# -*- coding: utf-8 -*-
"""Accident_Predictor_DB_Cleaning(2).ipynb
"""

!pip install numpy -q
!pip install pandas -q
!pip install matplotlib -q
!pip install tensorflow -q

!pip install opendatasets -q

# import libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pandas as pd
import opendatasets as od

od.download("https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents")

df = pd.read_csv(
    "/content/us-accidents/US_Accidents_March23.csv",
    nrows=10000000
)

df.head()

df.info()

print(df.isnull().sum())

(df == 0).sum()

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

df_selected = df[features]

output_file = "accidents_small.csv"

df_selected.to_csv(output_file, index=False)

df_cleaned = df.dropna()

print("Cleaned Dataset Shape:", df_cleaned.shape)

print("\nRemaining Null Values:")
print(df_cleaned.isnull().sum())

df_cleaned.to_csv('/content/cleaned_dataset.csv', index=False)