#Importing required libraries

import pandas as pd
import numpy as np
import plotly.express as px
import math
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import keras_tuner as kt

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
# From the scatter plot, we can see three distinct clusters, i.e., first one around 0k distance, second around 5k distance and final around 20k distance.
# For distance around 0k, the time taken varies a lot, i.e. from 10 minutes to 50 minutes, in this cluster, the distance varies from 1 unit to 20 units.
# For distances around 5k, the time taken shows a variation between 10 minutes and 50 minutes.
# For distances around 20k, the time taken is again varied but appears to be slightly concentrated between 20 minutes to 40 minutes.
# The blue trend line seems horizontal which indicates that there is no strong correlation between distance and time taken in this data set.
# 
# Insights:
# beta1 = -2.14e-5, beta0 = 26.67, R_squared = 0.000006. As the horizontal trend line tells that the time taken does not increase linearly with the distance. The reasons for this could be due to various factors such as traffic conditions, varying speed limits, or differences in the time required for short vs. long distances.
# The distinct clusters might indicate that the distances are not continuous but rather fall into specific categories (e.g., different zones for delivery, specific routes, etc.). 
# The wide range of time taken for each distance suggests high variability, possibly due to external factors not captured in the data, such as waiting times, delivery handling times, or service efficiency.

#Relationship between the delivery time and age of the driver.

figure = px.scatter(data_frame = data,
                    x = "Delivery_person_Age",
                    y = "Time_taken(min)",
                    size = "Time_taken(min)",
                    color = "distance",
                    trendline = "ols",
                    title = "Relationship between Time taken (mins) and Driver's Age",
                    labels = {"Delivery_person_Age": "Delivery person's age", "Time_taken(min)": "Time Taken (mins)", "distance": "Distance"})

figure.show()

figure.write_image("../output/relationship_between_driver_age_and_time_taken.png")

# COMMENTS: 
# Observations:
# From the scatter plot, we can see various distinct clusters as per the age values, majority of the clusters can be seen from age = 20 to age = 39, the two extremes are age = 15 and age = 50.
# For most of the age ranges, we can see a variation between 10 minutes to 54 minutes for 3 to 20 units of distance.
# Drivers belonging to age of 15, 29, and 50 were seen to go for longer delivery routes. Mostly age = 50 were given tasks to deliver to large distances.
# The trend line shows a linear relationship between driver's age and the time taken for delivery. This indicates that young delivery partners take less time to deliver the food compared to the elder partners.
# 
# Insights:
# beta1 = 0.48, beta0 = 12.05, R_squared = 0.086. As the age of the driver increases, the time taken for delivery also tends to increase. Specifically, for each additional year of the driver's age, the delivery time increases by approximately 0.48 minutes on average.
# Drivers in the 15-29 age group generally have lower delivery times. The variability in delivery time within this group could be due to factors such as experience, familiarity with routes, and possibly higher physical stamina.
# Drivers in the 30-39 age group shows a wider range of delivery times, which could be attributed to a mix of factors including type of delivery vehicle, varying levels of experience, differences in driving habits, and potentially more cautious driving behavior compared to younger drivers.
# Drivers in their 50s, tend to have longer delivery times. This could be due to more cautious driving, possible physical limitations, or preference for routes that might be less congested but longer in distance.

#Relationship between delivery time and the ratings of the delivery partner.

figure = px.scatter(data_frame = data,
                    x = "Delivery_person_Ratings",
                    y = "Time_taken(min)",
                    size = "Time_taken(min)",
                    color = "distance",
                    trendline = "ols",
                    title = "Relationship between Time Taken (mins) and Driver's Ratings",
                    labels = {"Time_taken(min)":"Time Taken (mins)","distance":"Distance","Delivery_person_Ratings":"Delivery person's Ratings"})

figure.show()

figure.write_image("../output/relationship_between_time_taken_and_driver_ratings.png")

# COMMENTS:
# Observations:
# From the scatter plot, we can see various distinct clusters as per the ratings, majority of the clusters can be seen between rating 4 and 5, there are clusters for ratings 1, 2.5 to 4, and 6 as well.
# For most of the ratings, we can see a variation between 10 minutes to 54 minutes for 3 to 20 units of distance.
# Drivers with ratings 1 and 6 were seen to go for longer delivery routes. A lot of delivery partners who went for longer deliveries received a 6 rating.
# The trend line shows an inverse linear relationship between driver's ratings and the time taken for delivery. This indicates that delivery partners with higher ratings take less time to deliver the food compared to the partners with lower ratings.
# 
# Insights:
# beta1 = -9.48, beta0 = 70.21, R_squared = 0.11. As the rating of the driver increases, the time taken for delivery tends to decrease. Specifically, for each additional rating of the driver, the delivery time decreases by approximately 9.48 minutes on average.
# Drivers with low ratings like 1 have longer delivery times. This could be due to various factors such as lack of familiarity with routes, less efficiency in handling deliveries, or potentially lower motivation.
# Drivers in 2.5 to 4 rating categories might be in the process of improving their efficiency or might be facing occasional challenges that affect their delivery times.
# Drivers with high ratings 4 to 5, especially 6 consistently have shorter delivery times despite the longer distance for rating 6 people. Higher-rated drivers are more efficient, likely due to better familiarity with routes, higher motivation, and possibly more experience. This is a positive indicator for using driver ratings as a metric to optimize delivery performance.

#Affect of type of food and type of vehicle used for delivery on time taken for delivery.

figure = px.box(data,
                x = "Type_of_vehicle",
                y = "Time_taken(min)",
                color = "Type_of_order",
                title = "Affect of Food Type and Delivery Vehicle Type on Time Taken (mins)",
                labels = {"Type_of_vehicle": "Type of Vehicle", "Type_of_order": "Type of Food", "Time_taken(min)": "Time Taken (mins)"})

figure.show()

figure.write_image("../output/affect_of_food_type_and_delivery_vehicle_type_on_time_taken.png")

# COMMENTS:
# Electric scooters and scooters provide the most consistent delivery times with tighter IQRs across all food types. While bicycles and motorcycles show more variability in delivery times, which might be due to their limitations in distance and speed.
# Electric scooters appear to be the most efficient in terms of maintaining lower maximum delivery times and consistent IQRs. Though motorcycles can cover longer distances, the high variability in delivery times can be possibly seen due to traffic conditions and route complexities.
# The type of food being delivered does not significantly impact the median delivery times across different vehicle types. However, the variability in delivery times (IQR) can be influenced by the vehicle type, with motorized vehicles providing more consistency.
# For ensuring faster and more consistent delivery times, electric scooters and scooters should be preferred, especially for longer distances. Bicycles can be utilized for shorter and medium-distance deliveries, especially in areas where motorized vehicles might face traffic constraints.

#Delivery time prediction model

x = np.array(data[["Delivery_person_Age", "Delivery_person_Ratings", "distance"]])
y = np.array(data[['Time_taken(min)']])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.10, random_state=42)

#LSTM model

model = Sequential()
model.add(LSTM(128, return_sequences = True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(64, return_sequences = False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

#Training the model
model.compile(optimizer = "adam", loss = "mean_squared_error")
model.fit(x_train, y_train, batch_size = 1, epochs = 9)

#Testing the model

print("Food Delivery Time Predictor")
a = int(input("Age of Delivery Partner: "))
b = float(input("Ratings of Previous Deliveries: "))
c = int(input("Total Distance: "))

features = np.array([[a, b, c]])
print("Predicted delivery time in minutes = ", model.predict(features))

# Output for reference:
# Model: "sequential"
# ┌──────────────────────────────────────┬─────────────────────────────┬─────────────────┐
# │ Layer (type)                         │ Output Shape                │         Param # │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ lstm (LSTM)                          │ (None, 3, 128)              │          66,560 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ lstm_1 (LSTM)                        │ (None, 64)                  │          49,408 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense (Dense)                        │ (None, 25)                  │           1,625 │
# ├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
# │ dense_1 (Dense)                      │ (None, 1)                   │              26 │
# └──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
#  Total params: 117,619 (459.45 KB)
#  Trainable params: 117,619 (459.45 KB)
#  Non-trainable params: 0 (0.00 B)
# Epoch 1/9
# ←[1m41033/41033←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m101s←[0m 2ms/step - loss: 75.5305
# Epoch 2/9
# ←[1m41033/41033←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m99s←[0m 2ms/step - loss: 64.5294
# Epoch 3/9
# ←[1m41033/41033←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m80s←[0m 2ms/step - loss: 62.4145
# Epoch 4/9
# ←[1m41033/41033←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m73s←[0m 2ms/step - loss: 60.5329
# Epoch 5/9
# ←[1m41033/41033←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m91s←[0m 2ms/step - loss: 60.1548
# Epoch 6/9
# ←[1m41033/41033←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m98s←[0m 2ms/step - loss: 59.6412
# Epoch 7/9
# ←[1m41033/41033←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m94s←[0m 2ms/step - loss: 58.8463
# Epoch 8/9
# ←[1m41033/41033←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m94s←[0m 2ms/step - loss: 60.0760
# Epoch 9/9
# ←[1m41033/41033←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m89s←[0m 2ms/step - loss: 58.8106
# Food Delivery Time Predictor
# Age of Delivery Partner: 29
# Ratings of Previous Deliveries: 2.9
# Total Distance: 6
# ←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 210ms/step
# Predicted delivery time in minutes =  [[40.782604]]

# COMMENTS: 
# As we saw in our EDA, the delivery time for a 29 years old delivery partner is in the range of 3 to 54 minutes, despite the distance.
# Our predicted time is 40.78 minutes which lies in the aforementioned range.
# However, as we saw that the delivery partner's rating and delivery time are inversely proportional.
# A low rating of 2.9 suggests there is room for improvement and as the delivery partner's delivery time will reduce, the rating should increase.

#Testing the model on test data.
x_test_reshaped = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

print("Shape of x_test_reshaped:", x_test_reshaped.shape)

y_pred = model.predict(x_test_reshaped)


mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error (MSE):", mse)

mae = mean_absolute_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)

r2 = r2_score(y_test, y_pred)
print("R-squared (R²):", r2)

# 143/143 [==============================] - 1s 3ms/step
# Mean Squared Error (MSE): 55.89161352435876
# Mean Absolute Error (MAE): 5.815907666557713
# R-squared (R²): 0.35977780211349264

# COMMENTS:
# The model explains only 36% of the variance in the target variable and isn't capturing the underlying patterns in the data well.

#Hyperparameter Tuning.

#Tuning various parameters to change the model structure.
def build_model(hp):
    model = Sequential()

    model.add(LSTM(units=hp.Int('units_1', min_value=32, max_value=256, step=32), return_sequences=True, input_shape=(x_train.shape[1], 1)))

    if hp.Boolean('use_second_lstm'):
        model.add(LSTM(units=hp.Int('units_2', min_value=32, max_value=128, step=32), return_sequences=False))
    
    model.add(Dense(units=hp.Int('dense_units', min_value=16, max_value=64, step=16)))
    
    model.add(Dense(1))
    
    model.compile(optimizer=hp.Choice('optimizer', values=['adam', 'rmsprop']),
                  loss='mean_squared_error')
    return model

#Defining a keras tuner.
tuner = kt.Hyperband(build_model,
                     objective='val_loss',
                     max_epochs=10,
                     factor=3,
                     directory='tuner_dir',
                     project_name='delivery_time_prediction')


math.prod = np.prod

def prod(iterable):
    result = 1
    for n in iterable:
        result *= n
    return result

math.prod = prod

#Search the best validation loss model.
tuner.search(x_train, y_train, epochs=10, validation_split=0.1)

# OUTPUT:
# Trial 26 Complete [00h 01m 46s]
# val_loss: 55.90714645385742
# 
# Best val_loss So Far: 55.90714645385742
# Total elapsed time: 00h 18m 24s

#Build the best hypertuned model.
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
best_model = tuner.hypermodel.build(best_hps)
best_model.summary()

# Model: "sequential_1"
# _________________________________________________________________
#  Layer (type)                Output Shape              Param #   
# =================================================================
#  lstm_2 (LSTM)               (None, 3, 224)            202496    
#                                                                  
#  lstm_3 (LSTM)               (None, 96)                123264    
#                                                                  
#  dense_2 (Dense)             (None, 48)                4656      
#                                                                  
#  dense_3 (Dense)             (None, 1)                 49        
#                                                                  
# =================================================================
# Total params: 330,465
# Trainable params: 330,465
# Non-trainable params: 0
# _________________________________________________________________

#Fit the best hypertuned model.
best_model.fit(x_train, y_train, epochs=10, validation_split=0.1)

# Epoch 1/10
# 1155/1155 [==============================] - 13s 9ms/step - loss: 80.0099 - val_loss: 64.9712
# Epoch 2/10
# 1155/1155 [==============================] - 10s 9ms/step - loss: 65.6189 - val_loss: 62.9452
# Epoch 3/10
# 1155/1155 [==============================] - 10s 8ms/step - loss: 62.8443 - val_loss: 59.9478
# Epoch 4/10
# 1155/1155 [==============================] - 9s 8ms/step - loss: 61.7201 - val_loss: 59.5722
# Epoch 5/10
# 1155/1155 [==============================] - 10s 9ms/step - loss: 60.7775 - val_loss: 59.1899
# Epoch 6/10
# 1155/1155 [==============================] - 10s 9ms/step - loss: 60.0641 - val_loss: 57.8616
# Epoch 7/10
# 1155/1155 [==============================] - 10s 9ms/step - loss: 59.7251 - val_loss: 58.3385
# Epoch 8/10
# 1155/1155 [==============================] - 10s 9ms/step - loss: 59.1028 - val_loss: 56.6811
# Epoch 9/10
# 1155/1155 [==============================] - 10s 9ms/step - loss: 58.7261 - val_loss: 56.4427
# Epoch 10/10
# 1155/1155 [==============================] - 11s 9ms/step - loss: 58.3225 - val_loss: 55.7493

#Compile the model with required metrics to get the results results.
best_model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

test_loss, test_mse, test_mae = best_model.evaluate(x_test, y_test)

#Test data error measurement.
print(f"Test Loss: {test_loss}")
print(f"Test Mean Squared Error (MSE): {test_mse}")
print(f"Test Mean Absolute Error (MAE): {test_mae}")



#Predict the delivery time.
y_pred = best_model.predict(x_test)

#Get the predicted and original output in the same format.
y_pred = y_pred.flatten()
y_test = y_test.flatten()

fig = px.scatter(
    x=y_test, 
    y=y_pred, 
    labels={'x': 'True Values', 'y': 'Predicted Values'}, 
    title='True vs. Predicted Values'
)

fig.add_shape(
    type="line", 
    x0=min(y_test), y0=min(y_test), 
    x1=max(y_test), y1=max(y_test),
    line=dict(color="red", dash="dash")
)

fig.show()

fig.write_image("../output/Final.png")

best_model.save('best_model.h5')

# 143/143 [==============================] - 2s 4ms/step - loss: 55.9720 - mse: 55.9720 - mae: 5.8443
# Test Loss: 55.97201156616211
# Test Mean Squared Error (MSE): 55.97201156616211
# Test Mean Absolute Error (MAE): 5.8442816734313965
# 143/143 [==============================] - 1s 4ms/step