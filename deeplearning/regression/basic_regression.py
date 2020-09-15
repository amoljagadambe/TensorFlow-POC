"""
Simple Regression Example
"""
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

x_data = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

y_label = np.linspace(0, 10, 10) + np.random.uniform(-1.5, 1.5, 10)

# plt.plot(x_data, y_label, '*')
# plt.show()

'''
y = mx + b
'''

m = tf.Variable(0.44)
b = tf.Variable(0.87)

error = 0

for x, y in zip(x_data, y_label):
    y_hat = m * x + b
    error += (y - y_hat) ** 2

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(error)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    training_steps = 1

    for i in range(0, training_steps):
        sess.run(train)

    final_slope, final_intercepts = sess.run((m, b))

x_test = np.linspace(-1, 11, 10)
y_pred = final_slope*x_test + final_intercepts

plt.ylim((0, 15))
plt.plot(x_test, y_pred, 'b')
plt.plot(x_data, y_label, '*')
plt.show()