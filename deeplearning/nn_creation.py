import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs


class Operation:

    def __init__(self, input_nodes=[]):
        self.input_nodes = input_nodes
        self.output_nodes = []

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)

    def compute(self):
        pass


class add(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var


class multiply(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class matmul(Operation):

    def __init__(self, x, y):
        super().__init__([x, y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var.dot(y_var)


class Placeholder:
    def __init__(self):
        self.output_nodes = []

        _default_graph.placeholders.append(self)


class Variable:

    def __init__(self, initial_vaule=None):
        self.value = initial_vaule
        self.output_nodes = []

        _default_graph.variables.append(self)


class Graph:

    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self


def traverse_postorder(operation):
    """
    PostOrder Traversal of Nodes. Basically makes sure computations are done in
    the correct order (Ax first , then Ax + b). Feel free to copy and paste this code.
    It is not super important for understanding the basic fundamentals of deep learning.
    """

    nodes_postorder = []

    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)

    recurse(operation)
    return nodes_postorder


class Session:

    def run(self, operation, feed_dict={}):
        """
            operation: The operation to compute
            feed_dict: Dictionary mapping placeholders to input values (the data)
          """

        nodes_postorder = traverse_postorder(operation)

        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]

            elif type(node) == Variable:
                node.output = node.value

            else:
                # OPERATION
                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)  # args

            if type(node.output) == list:
                node.output = np.array(node.output)

        return operation.output


# create graph and run session
"""
commented cause using same variable in below graph
g = Graph()
g.set_as_default()
A = Variable([[10, 20], [30, 40]])
b = Variable([1, 2])
x = Placeholder()
y = matmul(A, x)
z = add(y, b)
sess = Session()
result = sess.run(z, feed_dict={x: 10})"""
"""
output:(result)
[[101 202]
 [301 402]]

"""

"""
Activation Function
"""


class Sigmoid(Operation):

    def __init__(self, z):
        super().__init__([z])

    def compute(self, z_val):
        return 1 / (1 + np.exp(-z_val))


data = make_blobs(n_samples=50, n_features=2, centers=2, random_state=75)
features = data[0]
labels = data[1]
plt.scatter(features[:, 0], features[:, 1], c=labels, cmap='coolwarm')

x = np.linspace(0, 11, 10)
y = -x + 5
plt.plot(x, y)

# create graph and run session
g = Graph()
g.set_as_default()
b = Variable(-5)
w = Variable([1, 1])
x = Placeholder()
z = add(matmul(w, x), b)
a = Sigmoid(z)
sess = Session()
result = sess.run(operation=a, feed_dict={x: [8, 10]})
print(result)
# result_1 = sess.run(operation=a, feed_dict={x: [2, -10]})
# print(result_1)