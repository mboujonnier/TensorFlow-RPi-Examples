'''
A logistic regresstion algorithm example using TensorFlow Python and C++ library
to run on the Raspberry Pi.
This example will train a Logistic Regression model, export the graph definition and data
into files then perform a testing phase with test data. This testing phase is identical to the
C++ code example running on the Raspberry Pi.

Author: Matthieu Boujonnier
Original Author: Kim Sung (https://github.com/codertimo/Tensorflow-Study)

Licensed under the MIT license
'''

from __future__ import print_function
from tensorflow.contrib.session_bundle import exporter

import freeze_graph
import tensorflow as tf
import os.path

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Flags
tf.app.flags.DEFINE_string('model_dir', 'data', """Relative directory where to write model proto to import in c++""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'data',"""Relative directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('training_iteration', 1000, 'number of training iterations.')
tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
tf.app.flags.DEFINE_string('work_dir', 'data', 'Relative working directory.')
FLAGS = tf.app.flags.FLAGS

def freeze_my_graph(sess, output_node_names):

  checkpoint_state_name = "checkpoint_state"
  input_graph_name = "input_graph.pb"
  output_graph_name = "output_graph.pb"
  checkpoint_prefix = os.path.join(FLAGS.checkpoint_dir, "saved_checkpoint")

  # Export checkpoint state
  saver = tf.train.Saver()
  saver.save(sess, checkpoint_prefix, global_step=0, latest_filename=checkpoint_state_name)

  # export graph definition
  tf.train.write_graph(sess.graph.as_graph_def(), FLAGS.model_dir, input_graph_name)
  print('graph definition saved in dir: ', FLAGS.model_dir)

  # We save out the graph to disk, and then call the const conversion routine.
  input_graph_path = os.path.join(FLAGS.model_dir, input_graph_name)
  input_saver_def_path = ""
  input_binary = False
  input_checkpoint_path = checkpoint_prefix + "-0"
  restore_op_name = "save/restore_all"
  filename_tensor_name = "save/Const:0"
  output_graph_path = os.path.join(FLAGS.model_dir, output_graph_name)
  clear_devices = False
  initializer_nodes = ""

  # freeze_graph is in TensorFlow codebase (https://github.com/tensorflow/tensorflow/blob/HEAD/tensorflow/python/tools/freeze_graph.py)
  freeze_graph.freeze_graph(input_graph_path, input_saver_def_path, input_binary, input_checkpoint_path,
                            output_node_names, restore_op_name, filename_tensor_name,
                            output_graph_path, clear_devices, initializer_nodes)

# define sigmoid/sigma and sigmoid gradient (sigmaprime) functions
# tf.sigmoid(x, name=None)
def sigma(x):
    return tf.div(tf.constant(1.0), tf.add(tf.constant(1.0), tf.exp(tf.neg(x))))


def sigmaprime(x):
    return tf.mul(sigma(x), tf.sub(tf.constant(1.0), sigma(x)))
    #return tf.mul(tf.sigmoid(x), tf.sub(tf.constant(1.0), tf.sigmoid(x)))


def main(argv=None):

  # Parameters
  nb_iterations = 10001
  learning_rate = 0.1
  display_step = 1000

  # setup neural network layers

  # layer 1 - inputs
  a_0 = tf.placeholder(tf.float32, [None, 784])
  y = tf.placeholder(tf.float32, [None, 10])
  # layer 2 - hidden
  middle = 30
  w_1 = tf.Variable(tf.truncated_normal([784, middle]))
  b_1 = tf.Variable(tf.truncated_normal([1, middle]))
  # layer 3 - output
  w_2 = tf.Variable(tf.truncated_normal([middle, 10]))
  b_2 = tf.Variable(tf.truncated_normal([1, 10]))

  # forward propagation
  z_1 = tf.add(tf.matmul(a_0, w_1), b_1)
  a_1 = sigma(z_1)
  z_2 = tf.add(tf.matmul(a_1, w_2), b_2)
  a_2 = sigma(z_2)

  # backpropagation
  # output: diff=delta(k)=a(k) - y(k)
  diff = tf.sub(a_2, y)

  # hidden layer: delta2=transpose(theta2)*delta3.* sigmoidGradient(z2)
  d_z_2 = tf.mul(diff, sigmaprime(z_2))
  d_b_2 = d_z_2
  d_w_2 = tf.matmul(tf.transpose(a_1), d_z_2)

  # input layer
  d_a_1 = tf.matmul(d_z_2, tf.transpose(w_2))
  d_z_1 = tf.mul(d_a_1, sigmaprime(z_1))
  d_b_1 = d_z_1
  d_w_1 = tf.matmul(tf.transpose(a_0), d_z_1)

  # compute big Delta
  eta = tf.constant(0.5)
  step = [
      tf.assign(w_1,   tf.sub(w_1, tf.mul(eta, d_w_1)))
    , tf.assign(b_1, tf.sub(b_1, tf.mul(eta, tf.reduce_mean(d_b_1, reduction_indices=[0]))))
    , tf.assign(w_2, tf.sub(w_2, tf.mul(eta, d_w_2)))
    , tf.assign(b_2, tf.sub(b_2, tf.mul(eta, tf.reduce_mean(d_b_2, reduction_indices=[0]))))
  ]


  # The following will be able to train the network and test it in the meanwhile, using mini-batches of 10.
# Here, I chose to test with 1000 items from the test set, every 1000 mini-batches.
  acct_mat = tf.equal(tf.argmax(a_2, 1), tf.argmax(y, 1))
  acct_res = tf.reduce_sum(tf.cast(acct_mat, tf.float32))

  sess = tf.InteractiveSession()
  sess.run(tf.initialize_all_variables())

  print("Training the model using steps")
  for i in xrange(nb_iterations):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(step, feed_dict = {a_0: batch_xs, y : batch_ys})
    if i % display_step == 0:
        res = sess.run(acct_res, feed_dict = {a_0: mnist.test.images[:1000], y : mnist.test.labels[:1000]})
        print("after ", i, " iterations, ", res, " correct tests on 1000")


  sess.run(tf.initialize_all_variables())
  cost = tf.mul(diff, diff)
  train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  print("Training the model TF gradient descent optimizer")
  for i in xrange(nb_iterations):
    batch_xs, batch_ys = mnist.train.next_batch(10)
    sess.run(train, feed_dict = {a_0: batch_xs, y : batch_ys})
    if i % display_step == 0:
        res = sess.run(acct_res, feed_dict = {a_0: mnist.test.images[:1000], y : mnist.test.labels[:1000]})
        print("after ", i, " iterations, ", res, " correct tests on 1000")

if __name__ == '__main__':
  tf.app.run()
