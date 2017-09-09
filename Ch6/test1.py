import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn import datasets

iris = datasets.load_iris()
x_vals = np.array([x[0:3] for x in iris.data])
y_vals = np.array([x[3] for x in iris.data])
sess = tf.Session()
