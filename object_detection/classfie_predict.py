from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import os
import numpy as np

cwd = os.getcwd()
model_save_path = cwd + '/models/'
loaded_model = load_model(model_save_path + "cnn_30_epochs.h5")

loaded_model.compile(loss='binary_crossentropy',
                     optimizer='rmsprop',
                     metrics=['accuracy'])

image = load_img('D:/ml_datasets/dog_cat_classification/data/test/cat_1.jpg', target_size=(150, 150))
input_arr = img_to_array(image)
input_arr = np.array([input_arr])
predictions = loaded_model.predict(input_arr)
print('predictions>>>>', predictions)

dog = float(0)

if dog in predictions:
    print('This is cat')

else:
    print('Hey, this is Dog')
