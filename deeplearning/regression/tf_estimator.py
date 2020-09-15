"""
TF ESTIMATOR
"""

from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

x_data = np.linspace(0.0, 10, 1000000)  # 1 million points
noise = np.random.randn(len(x_data))

y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

my_data = pd.concat([x_df, y_df], axis=1)

feat_cols = [tf.feature_column.numeric_column('x', shape=1)]
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

X_train, X_test, y_train, y_test = train_test_split(x_data, y_true, test_size=0.3, random_state=101)

'''
print(X_train.shape)
print(X_test.shape)

(700000,)
(300000,)
'''

input_func = tf.estimator.inputs.numpy_input_fn({'x': X_train}, y_train, batch_size=8, num_epochs=None, shuffle=True)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x': X_train}, y_train, batch_size=8,
                                                      num_epochs=1000, shuffle=False)

test_input_func = tf.estimator.inputs.numpy_input_fn({'x': X_test}, y_test, batch_size=8,
                                                     num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_func, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)

test_metrics = estimator.evaluate(input_fn=test_input_func, steps=1000)

'''
Over fitting checker

Eval metrics loss should be as less or as equal to the training loss
'''

# Prediction

new_data = np.random.randint(1, 10, size=1)  # single point to predict the y_label
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x': new_data}, shuffle=False)
# estimator.predict(input_fn=input_fn_predict) # this is generator

predictions = []

for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])

my_data.sample(250).plot(kind='scatter', x='X data', y='Y')
plt.plot(new_data, predictions, 'r*')
plt.show()
