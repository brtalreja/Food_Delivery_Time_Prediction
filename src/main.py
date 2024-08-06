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

# As we do not have any null values, we can move further in our task.
# Because the dataset doesn't contain any feature that shows the difference between the restaurant and the delivery location.
# However, we have longitude and latitude of the restaurant and the delivery location.

#Finding the distance between the restaurant and delivery location based on their latitudes and longitudes by using haversine formula.

R = 6371 #Earth's radius in KM.

#Helper function to convert degrees to radians
def deg_to_rad(degrees):
    '''
        This function takes angle's degrees value as input and return the angle value in radians.
    '''
    return degrees * (np.pi/180)

#Function to calculate the distance
def distcalculate(lat1, lon1, lat2, lon2):
    '''
        This function takes the latitude and longitude in degrees as input and returns the distance between two points using haversine formula.
    '''
    d_lat = deg_to_rad(lat2 - lat1)
    d_lon = deg_to_rad(lon2 - lon1)
    a = np.sin(d_lat/2) ** 2 + np.cos(deg_to_rad(lat1)) * np.cos(deg_to_rad(lat2)) * np.sin(d_lon/2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

data['distance'] = np.nan

for i in range(len(data)):
    data.loc[i, 'distance'] = distcalculate(data.loc[i, 'Restaurant_latitude'],
                                            data.loc[i, 'Restaurant_longitude'],
                                            data.loc[i, 'Delivery_location_latitude'],
                                            data.loc[i, 'Delivery_location_longitude'])

print(data.head())