import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

"""
Numpy Revision - seed, arange, reshape, boolean masking
"""
np.random.seed(101)  # this will always generate same random value foe randint when we restart the program

array = np.arange(0, 100).reshape(10, 10)  # reshape convert the array into 10*10 dimension
"""
output: 
[[ 0  1  2  3  4  5  6  7  8  9]
 [10 11 12 13 14 15 16 17 18 19]
 [20 21 22 23 24 25 26 27 28 29]
 [30 31 32 33 34 35 36 37 38 39]
 [40 41 42 43 44 45 46 47 48 49]
 [50 51 52 53 54 55 56 57 58 59]
 [60 61 62 63 64 65 66 67 68 69]
 [70 71 72 73 74 75 76 77 78 79]
 [80 81 82 83 84 85 86 87 88 89]
 [90 91 92 93 94 95 96 97 98 99]]

"""
mask = np.where(array <= 50, 0, 1)  # used foe encoding labels in dataset
"""
output:
[[0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 0 0 0 0 0 0 0 0 0]
 [0 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1]
 [1 1 1 1 1 1 1 1 1 1]]

"""
boolean_masking = array[array > 50]
"""
output:
[51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74
 75 76 77 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98
 99]

"""

"""
Pandas Revision - read_csv, describe, as_matrix, masking
"""

cwd = os.getcwd()
file_path = os.path.join(cwd, 'files', 'salaries.csv')

df = pd.read_csv(file_path)  # read the csv file  and store it in dataframe
"""
output:
     Name  Salary  Age
0    John   50000   34
1   Sally  120000   45
2  Alyssa   80000   27

"""
df.describe()  # this will help to get mean, std. and all other info about columns
"""
output :
              Salary        Age
count       3.000000   3.000000
mean    83333.333333  35.333333
std     35118.845843   9.073772
min     50000.000000  27.000000
25%     65000.000000  30.500000
50%     80000.000000  34.000000
75%    100000.000000  39.500000
max    120000.000000  45.000000

"""
mean_of_column = df['Age'].mean()  # this way we will get only mean of columns
"""
output:
35.333333333333336
"""
salary_division = df[df['Salary'] > 50000]  # this is masking same as numpy array masking
"""
output:
     Name  Salary  Age
1   Sally  120000   45
2  Alyssa   80000   27

"""
df.as_matrix()  # convert dataframe  into numpy array
"""
output:
[['John' 50000 34]
 ['Sally' 120000 45]
 ['Alyssa' 80000 27]]

"""
df.plot(x='Salary', y='Age', kind='scatter')  # this will generate scatter plot we can pass any kind


"""
Matplotlib Revision
"""
x = np.arange(0, 10)
y = x ** 2
plt.plot(x, y, 'm')  # this will plot x and y with red line
"""
character color
'b'     blue
'g'     green
'r'     red
'c'     cyan
'm'     magenta
'y'     yellow
'k'     black
'w'     white

character description
'-'     solid line style
'--'    dashed line style
'-.'    dash-dot line style
':'     dotted line style
'.'     point marker
','     pixel marker
'o'     circle marker
'v'     triangle_down marker
'^'     triangle_up marker
'<'     triangle_left marker
'>'     triangle_right marker
'1'     tri_down marker
'2'     tri_up marker
'3'     tri_left marker
'4'     tri_right marker
's'     square marker
'p'     pentagon marker
'*'     star marker
'h'     hexagon1 marker
'H'     hexagon2 marker
'+'     plus marker
'x'     x marker
'D'     diamond marker
'd'     thin_diamond marker
'|'     vline marker
'_'     hline marker

"""
plt.xlim(0)  # this will help to limit x axis start at 0  we can also set endpoint
plt.ylim(0, 81)  # this will help to limit y axis start at 0  we can also set endpoint

# title and labels
plt.title('My title', color='m', fontsize=20)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
"""
using imshow on 2D matrix
"""
matrix_data = np.random.randint(0, 1000, (10, 10))
plt.imshow(matrix_data, cmap='PuBuGn')
plt.colorbar()  # this will create color bar with value on right hand side of imshow map
plt.show()
