import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas as pd
import os

cwd = os.getcwd()
file_path = os.path.join(cwd, 'files', 'cal_housing_clean.csv')

housing = pd.read_csv(file_path)
y_val = housing['medianHouseValue']
x_data = housing.drop('medianHouseValue', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_val, test_size=0.3, random_state=101)

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train = pd.DataFrame(data=scaler.transform(X_train), columns=X_train.columns, index=X_train.index)

X_test = pd.DataFrame(data=scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

age = tf.feature_column.numeric_column('housingMedianAge')
rooms = tf.feature_column.numeric_column('totalRooms')
bedrooms = tf.feature_column.numeric_column('totalBedrooms')
pop = tf.feature_column.numeric_column('population')
households = tf.feature_column.numeric_column('households')
income = tf.feature_column.numeric_column('medianIncome')

feat_cols = [age, rooms, bedrooms, pop, households, income]

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

model = tf.estimator.DNNRegressor(hidden_units=[6, 6, 6], feature_columns=feat_cols)

model.train(input_fn=input_func, steps=20000)

# prediction
predict_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)
prediction_gen = model.predict(input_fn=predict_input_func)

final_preds = []
for pred in prediction_gen:
    final_preds.append(pred['predictions'])

# RMSE

mse = mean_squared_error(y_test, final_preds)**0.5
print(mse)



