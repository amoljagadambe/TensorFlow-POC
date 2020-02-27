"""
Before running this code please make sure you have installed 
https://www.graphviz.org/ . this lib help to convet the model 
into .png file

This POC is based upon the below article
https://machinelearningmastery.com/keras-functional-api-deep-learning/
"""

from keras.models import Model
from keras.layers import Dense, Input
from keras.utils import plot_model
import os

"""
here we set the $PATH so it will not throw an error 
while converting the graph
"""

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

# Connecting Layers
visible = Input(shape=(2,))
hidden1 = Dense(10, activation='relu', name='first_layer')(visible)
hidden2 = Dense(20, activation='relu',name='feature_layer')(hidden1)
hidden3 = Dense(10, activation='relu', name='kernel_layer')(hidden2)
output = Dense(1, activation='sigmoid', name='output_layer')(hidden3)


# Creating the Model
model = Model(inputs=visible, outputs=output)


# summarize layers
print(model.summary())

# plot graph
plot_model(model, to_file='multilayer_perceptron_graph.png')