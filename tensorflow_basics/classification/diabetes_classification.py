import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
import pandas as pd
import os

cwd = os.getcwd()
file_path = os.path.join(cwd, 'files', 'pima-indians-diabetes.csv')

diabetes = pd.read_csv(file_path)

cols_to_norm = ['Number_pregnant', 'Glucose_concentration', 'Blood_pressure', 'Triceps',
                'Insulin', 'BMI', 'Pedigree']

diabetes[cols_to_norm] = diabetes[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

num_preg = tf.feature_column.numeric_column('Number_pregnant')
plasma_gluc = tf.feature_column.numeric_column('Glucose_concentration')
dias_press = tf.feature_column.numeric_column('Blood_pressure')
tricep = tf.feature_column.numeric_column('Triceps')
insulin = tf.feature_column.numeric_column('Insulin')
bmi = tf.feature_column.numeric_column('BMI')
diabetes_pedigree = tf.feature_column.numeric_column('Pedigree')
age = tf.feature_column.numeric_column('Age')

# Use below method if you know the group names and groups are smaller
assigned_group = tf.feature_column.categorical_column_with_vocabulary_list('Group', ['A', 'B', 'C', 'D'])

# if you don't know the group use hash_bucket
# assigned_group = tf.feature_column.categorical_column_with_hash_bucket('Group', hash_bucket_size=10)

diabetes['Age'].hist(bins=20)
# plt.show()

# Change continuous columns into categorical
age_bucket = tf.feature_column.bucketized_column(age, boundaries=[20, 30, 40, 50, 60, 70, 80])

feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, diabetes_pedigree, assigned_group, age_bucket]

x_data = diabetes.drop(['Class'], axis=1)
labels = diabetes['Class']

X_train, X_test, y_train, y_test = train_test_split(x_data, labels, test_size=0.3, random_state=101)

input_func = tf.estimator.inputs.pandas_input_fn(x=X_train, y=y_train, batch_size=10, num_epochs=1000, shuffle=True)

model = tf.estimator.LinearClassifier(feature_columns=feat_cols, n_classes=2)

# Train model
model.train(input_fn=input_func, steps=1000)

# Evaluate
eval_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, y=y_test, batch_size=10, num_epochs=1, shuffle=False)

results = model.evaluate(eval_input_func)

# predictions
pred_input_func = tf.estimator.inputs.pandas_input_fn(x=X_test, batch_size=10, num_epochs=1, shuffle=False)

predictions = model.predict(pred_input_func)
# print(list(predictions))

'''
DNN Classifier
'''
'''
For DNN to run we need to change categorical columns into embedded columns
'''

embedded_group_col = tf.feature_column.embedding_column(assigned_group, dimension=4)

feat_cols = [num_preg, plasma_gluc, dias_press, tricep, insulin, diabetes_pedigree, embedded_group_col, age_bucket]

dnn_model = tf.estimator.DNNClassifier(hidden_units=[10, 10, 10], feature_columns=feat_cols, n_classes=2)
dnn_model.train(input_fn=input_func, steps=1000)


dnn_results = dnn_model.evaluate(eval_input_func)
print(dnn_results)

