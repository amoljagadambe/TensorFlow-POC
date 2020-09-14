"""
remove INFO and WARNING from tensorflow lib
"""
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf

hello = tf.constant(10)
world = tf.constant(20)
my_mat = tf.fill((4, 4), value=2)
my_random = tf.random_normal((4, 4))
tf_ops = [my_mat, my_random]

"""
Matmul Operation
"""
a = tf.constant([[1, 2],
                 [3, 4]])

b = tf.constant([[10],
                 [100]])

mat_mul = tf.matmul(a, b)

with tf.Session() as sess:
    result = sess.run(hello + world)
    mat_res = sess.run(mat_mul)
    for op in tf_ops:
        op_res = sess.run(op)
        print(op_res)
        print('\n')

print(result)
print('\n')
print('MATMUL IS:\n{}'.format(mat_res))

"""
TF Variable and placeholders
"""

my_tensor = tf.random_uniform((4, 4))
my_var = tf.Variable(initial_value=my_tensor)
init = tf.global_variables_initializer()  # this will initialize the variable declared above
with tf.Session() as session:
    session.run(init)
    session.run(my_var)


ph = tf.placeholder(tf.float32)   # placeholder's
