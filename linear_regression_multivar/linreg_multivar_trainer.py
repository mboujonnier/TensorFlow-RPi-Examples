'''
A linear regression learning algorithm example using TensorFlow Python and C++ library
to run on the Raspberry Pi.
This example will train a simple Linear Regression model, export the graph definition and data
into files then perform a testing phase with test data. This testing phase is identical to the
C++ code example running on the Raspberry Pi.

Author: Matthieu Boujonnier
Original Author: Aymeric Damien (https://github.com/aymericdamien/TensorFlow-Examples/)

Licensed under the MIT license
'''

from __future__ import print_function
from tensorflow.contrib.session_bundle import exporter

import freeze_graph
import tensorflow as tf
import numpy as np
import os.path

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
  learning_rate = 0.1
  training_epochs = 100
  display_step = 10

  # Training Data
  xy = np.loadtxt('data_houses.txt',unpack=True,dtype='float32')
  train_X = xy[0:-1] # features
  train_Y = xy[-1] # labels
  n_samples = train_X.shape[1]
  n_features = train_X.shape[0]

  #normalization
  original_means = np.array([0., 0., 0.])
  original_stds = np.array([0., 0., 0.])
  for f in range(1, n_features):
      original_means[f] = np.mean(train_X[f])
      original_stds[f] = np.std(train_X[f])
      print("not normalized mean for ", f, " : ", original_means[f], "and std deviation: ",  original_stds[f] )
      #train_X[f] = ( train_X[f] - np.mean( train_X[f] )) / (np.max(train_X[f]) - np.min(train_X[f]))
      train_X[f] = ( train_X[f] - original_means[f] ) / original_stds[f]
      print("normalized mean",  f, " : ", np.mean(train_X[f]))

  print(np.mean(train_Y))

  print(train_X)
  print(train_Y)
  print("Found ", n_samples, " samples with ", n_features, " features")

  # tf Graph Input
  X = tf.placeholder(tf.float32, [n_features, n_samples], name="input_node/X")
  Y = tf.placeholder(tf.float32, [n_samples], name="input_node/Y")
  N = tf.placeholder(tf.float32, name="input_node/N") #tf.int32 would be better but it trigger an conversion error on the cost formula

  # Set initial model parameters (y=Wx) using random uniform values
  W = tf.Variable(tf.random_uniform([1, n_features],-1.0,1.0), name="output_weight")

  # Construct a linear model
  pred = tf.matmul(W, X)

  # Mean square error
  cost = tf.reduce_sum(tf.square(pred-Y)) / (2*N)

  # Use a Gradient descent to minimize the cost
  optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

  # Initializing the variables
  init = tf.initialize_all_variables()

  # Launch the graph
  with tf.Session() as sess:
      sess.run(init)

      print("Training")

      # Fit all training data
      for epoch in range(training_epochs):
          #for (x, y) in zip(train_X, train_Y):
          sess.run(optimizer, feed_dict={X: train_X, Y: train_Y, N: n_samples})

          # Display logs per epoch step
          if (epoch+1) % display_step == 0:
              c = sess.run(cost, feed_dict={X: train_X, Y:train_Y, N:n_samples})
              print("Epoch:", '%04d' % (epoch+1), "cost=", c, "W=", sess.run(W))

      print("Optimization Finished!")
      training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y, N: n_samples})

      tf.add(cost, 0, name="output_cost")   # dumb way to add the cost to the graph, any better way ?
      tf.add(training_cost, 0, name="output_training_cost") # dumb way to save the training cost value to the graph, any better way ?

      print("Training cost=", training_cost, "W=", sess.run(W), '\n')

      X2 = tf.placeholder(tf.float32, [3, 1], name="input_node/X2")
      h = tf.matmul(W, X2)
      tf.add(h, 0, name="output_price")
      tf.add(original_means, 0, name="output_means")
      tf.add(original_stds, 0, name="output_stds")

      # important, time to freeze the model now, you need to specify all the output node names !
      freeze_my_graph(sess, "output_weight,output_cost,output_training_cost,output_price,output_means,output_stds")
      print('Done exporting!')

      # infering model
      bedrooms = 3.0
      surface = 1650.0
      model_test = [[1],[surface],[bedrooms]]

      # normalized inputs
      for f in range(1, n_features):
          model_test[f] = ( model_test[f] - original_means[f]) / (original_stds[f])

      price = sess.run(h, feed_dict={X2: model_test})

      print("Estimated price for an house of ", surface, " sqfeet and ", bedrooms, " bedrooms: ", price[0,0], " dollars")

if __name__ == '__main__':
  tf.app.run()
