#Importing required libraries

import pandas as pd
import numpy as np
import plotly.express as px

#Loading the data
data = pd.read_csv("../data/deliverytime.txt")
print(data.head())

#Getting to know the data
data.info()

print(data.isnull().sum())

#As we do not have any null values, we can move further in our task.