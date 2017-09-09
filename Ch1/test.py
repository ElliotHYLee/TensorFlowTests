import tensorflow as tf

row_dim = 10
col_dim = 3

zero_tsr = tf.zeros([row_dim, col_dim])
ones_tsr = tf.ones([row_dim, col_dim])
filled_tsr = tf.fill([row_dim, col_dim], 42)
constant_tsr = tf.constant([1,2,3])

zeros = tf.zeros_like(constant_tsr)
ones = tf.ones_like(constant_tsr)

sess  = tf.Session()

print(sess.run(ones_tsr))
