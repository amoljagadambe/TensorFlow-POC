"""
Scaling the data and transforming the data into value between 0 and 1
"""
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

data = np.random.randint(0, 100, (10, 2))
"""
output:
[[14 25]
 [55 19]
 [80  5]
 [28 82]
 [18 85]
 [83 83]
 [59 64]
 [62 67]
 [ 3 81]
 [73 57]]

"""
scaler_model = MinMaxScaler()
# scaler_model.fit(data)
# scaler_model.transform(data)  # the alternative is doing fit nd transform at once but don't fit testing data
scaler_model.fit_transform(data)
"""
output:
[[0.         0.70666667]
 [0.79220779 0.86666667]
 [0.79220779 0.48      ]
 [0.46753247 0.84      ]
 [0.01298701 1.        ]
 [0.28571429 0.34666667]
 [0.62337662 0.        ]
 [0.32467532 0.12      ]
 [0.68831169 0.78666667]
 [1.         0.81333333]]

"""

df = pd.DataFrame(data=np.random.randint(0, 100, (50, 4)), columns=['f1', 'f2', 'f3', 'label'])
"""
output:
    f1  f2  f3  label
0   70  32  70     90
1   41  70  83     52
2   93  48  56     37
3   15  24  36     80
4   84  17  35     69
5   26  84  29     69
.   .   .   .      .
49   60  71  26     52

"""

X = df[['f1', 'f2', 'f3']]  # this is features
y = df['label']  # this is labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                    random_state=101)  # random_state is like setting seed in numpy
"""
above line we split the data into train test with ration of 30% 
"""