# IMPORTED LIBRARIES
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import *
from sklearn.preprocessing import LabelBinarizer


# Loading Data
data = pd.read_excel(r'D:\02 ARTIFICIAL INTELLIGENCE\artificial-intelligence\Abu Dhabi Weather 2024.xlsx')
# pd.set_option('display.max_rows', None)  # Remove comment to show all rows

# Calculating Sunshine Hours
data['SUNSHINE HOURS'] = (pd.to_datetime(data['SUNSET']) - pd.to_datetime(data['SUNRISE'])).dt.total_seconds() / 3600


# Linear Regression

import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([[1,1], [1,2], [2, 2], [2,3]])
y = np.dot(X, np.array([1, 2]) + 3)
reg = LinearRegression().fitx(X, y)
reg.score(X, y)

