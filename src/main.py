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

#EDA

#Relation between distance and time taken to deliver the food.

figure = px.scatter(data_frame = data,
                    x = "distance",
                    y = "Time_taken(min)",
                    size = "Time_taken(min)",
                    trendline = "ols",
                    title = "Relationship between Distance and Time Taken",
                    labels = {"distance": "Distance", "Time_taken(min)": "Time Taken (mins)"})

figure.show()

figure.write_image("../output/relationship_between_distance_and_delivery_time.png")

# COMMENTS: 
# Observations:
# From the scatter plot, we can see two distinct clusters, i.e., first one around 5k distance and second around 20k distance.
# For distances around 5k, the time taken shows a variation between 10 minutes and 50 minutes.
# For distances around 20k, the time taken is again varied but appears to be slightly concentrated between 20 minutes to 40 minutes.
# The blue trend line seems horizontal which indicates that there is no strong correlation between distance and time taken in this data set.
# 
# Insights:
# As the horizontal trend line tells that the time taken does not increase linearly with the distance. The reasons for this could be due to various factors such as traffic conditions, varying speed limits, or differences in the time required for short vs. long distances.
# The distinct clusters might indicate that the distances are not continuous but rather fall into specific categories (e.g., different zones for delivery, specific routes, etc.). 
# The wide range of time taken for each distance suggests high variability, possibly due to external factors not captured in the data, such as waiting times, delivery handling times, or service efficiency.