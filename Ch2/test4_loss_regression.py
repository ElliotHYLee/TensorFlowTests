import matplotlib.pyplot as plt
import tensorflow as tf

sess = tf.Session()

x_vals = tf.linspace(-1., 1, 500)
target = tf.constant(0.)

#L2 Loss
l2_y_vals = tf.square(target - x_vals)
l2_y_out = sess.run(l2_y_vals)

#L1 Loss
l1_y_vals = tf.abs(target - x_vals)
l1_y_out = sess.run(l1_y_vals)

#Pseudo-Huber Loss
delta1 = tf.constant(0.25)
phuber1_y_vals = tf.multiply(tf.square(delta1), tf.sqrt(1. + tf.square((target - x_vals)/delta1))-1.)
phuber1_y_out = sess.run(phuber1_y_vals)

delta2 = tf.constant(5.)
phuber2_y_vals = tf.multiply(tf.square(delta2), tf.sqrt(1. + tf.square((target - x_vals)/delta2))-1.)
phuber2_y_out = sess.run(phuber2_y_vals)

#plot
x_array = sess.run(x_vals)
plt.plot(x_array, l2_y_out, 'b-', label='L2 Loss')
plt.plot(x_array, l1_y_out, 'r--', label='L2 Loss')
plt.plot(x_array, phuber1_y_out, 'k-', label='L2 Loss')
plt.plot(x_array, phuber2_y_out, 'g:', label='L2 Loss')
plt.ylim(-0.2, 0.4)
plt.legend(loc='lower right', prop={'size': 11})
plt.show()
