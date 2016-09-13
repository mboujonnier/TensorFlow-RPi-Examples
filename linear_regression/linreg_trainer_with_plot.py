'''
A linear regression learning algorithm example using TensorFlow Python and C++ library
to run on the Raspberry Pi.
This example will train a simple Linear Regression model, export the graph definition and data
into files then perform a testing phase with test data. This testing phase is identical to the
C++ code example running on the Raspberry Pi.

This example also plots the fitted line passing thru the training data then the testing data.

Author: Matthieu Boujonnier
Original Author: Aymeric Damien (https://github.com/aymericdamien/TensorFlow-Examples/)

Licensed under the MIT license
'''

from __future__ import print_function
from tensorflow.contrib.session_bundle import exporter

import freeze_graph
import tensorflow as tf
import numpy
import os.path
import matplotlib.pyplot as plt

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

def main(argv=None):

  # Parameters
  learning_rate = 0.01
  training_epochs = 1000
  display_step = 50

  # Training Data
  train_X = numpy.asarray([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1]) # features
  train_Y = numpy.asarray([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,2.827,3.465,1.65,2.904,2.42,2.94,1.3]) # labels
  n_samples = train_X.shape[0]

  # tf Graph Input
  X = tf.placeholder(tf.float32, name="input_node/X")
  Y = tf.placeholder(tf.float32, name="input_node/Y")
  N = tf.placeholder(tf.float32, name="input_node/N") #tf.int32 would be better but it trigger an conversion error on the cost formula

  # Set initial model parameters (y=Wx+b) using random values
  W = tf.Variable(tf.random_uniform([1],-1.0,1.0), name="output_weight")
  b = tf.Variable(tf.random_uniform([1],-1.0,1.0), name="output_bias")

  # Construct a linear model
  pred = tf.add(tf.mul(X, W), b)

  # Mean square error
  cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*N)

  # Use a Gradient descent to minimize the cost
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # Initializing the variables
  init = tf.initialize_all_variables()

  # Launch the graph
  with tf.Session() as sess:
      sess.run(init)

      # Fit all training data
      for epoch in range(training_epochs):
          for (x, y) in zip(train_X, train_Y):
              sess.run(optimizer, feed_dict={X: x, Y: y, N: n_samples})

          # Display logs per epoch step
          if (epoch+1) % display_step == 0:
              c = sess.run(cost, feed_dict={X: train_X, Y:train_Y, N:n_samples})
              print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

      print("Optimization Finished!")
      training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y, N: n_samples})

      tf.add(cost, 0, name="output_cost")   # dumb way to add the cost to the graph, any better way ?
      tf.add(training_cost, 0, name="output_training_cost") # dumb way to save the training cost value to the graph, any better way ?

      print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

      # important, time to freeze the model now, you need to specify all the output node names !
      freeze_my_graph(sess, "output_weight,output_bias,output_cost,output_training_cost")
      print('Done exporting!')

      # Graphic display
      plt.plot(train_X, train_Y, 'ro', label='Original data')
      plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
      plt.legend()
      plt.show()

      # Testing example
      test_X = numpy.asarray([6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
      test_Y = numpy.asarray([1.84, 2.273, 3.2, 2.831, 2.92, 3.24, 1.35, 1.03])

      print("Testing... (Mean square loss Comparison)")

      testing_cost = sess.run(cost, feed_dict={X: test_X, Y: test_Y, N: test_X.shape[0]})  # same function as cost above

      print("Testing cost=", testing_cost)
      print("Absolute mean square loss difference:", abs(
        training_cost - testing_cost))

      plt.plot(test_X, test_Y, 'bo', label='Testing data')
      plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
      plt.legend()
      plt.show()

if __name__ == '__main__':
  tf.app.run()
