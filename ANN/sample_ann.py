import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# encode labels. e.g. "Female" is "1, 0" and "Male" is "0, 1"
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
