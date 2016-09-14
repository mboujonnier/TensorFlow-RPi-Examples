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
  learning_rate = 0.01
  training_epochs = 10000
  display_step = 1000

  # Training Data
  xy = np.loadtxt('data.txt',unpack=True,dtype='float32')
  train_X = xy[0:-1] # features
  train_Y = xy[-1] # labels
  n_samples = train_X.shape[1]
  n_features = train_X.shape[0]

  print(train_X)
  print(train_Y)
  print("Found ", n_samples, " samples with ", n_features, " features")

  # tf Graph Input
  X = tf.placeholder(tf.float32, name="input_node/X")
  Y = tf.placeholder(tf.float32, name="input_node/Y")
  N = tf.placeholder(tf.float32, name="input_node/N") #tf.int32 would be better but it trigger an conversion error on the cost formula

  # Set initial model parameters (y=Wx) using random uniform values
  W = tf.Variable(tf.random_uniform([1, n_features],-1.0,1.0), name="output_weight")

  # Construct a linear model
  h = tf.matmul(W, X)

  # Sigmoid function
  hyp = tf.div(1.,1.+tf.exp(-h), name="output_hyp")

  # Mean square error
  cost = -tf.reduce_sum( Y*tf.log(hyp) + (1-Y)*tf.log(1-hyp), name="output_cost") / (2*N)

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
          sess.run(optimizer, feed_dict={X: train_X, Y: train_Y, N: n_samples})

          # Display logs per epoch step
          if (epoch+1) % display_step == 0:
              c = sess.run(cost, feed_dict={X: train_X, Y:train_Y, N:n_samples})
              print("Epoch:", '%04d' % (epoch+1), "cost=", c, "W=", sess.run(W))

      print("Optimization Finished!")
      training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y, N: n_samples})
      tf.convert_to_tensor(training_cost, name="output_training_cost") #way to save the training cost value to the graph, any better way ?

      print("Training cost=", training_cost, "W=", sess.run(W), '\n')

      # important, time to freeze the model now, you need to specify all the output node names !
      freeze_my_graph(sess, "output_weight,output_cost,output_training_cost,output_hyp")
      print('Done exporting!')

      print('Testing the model')
      print("X:[[1],[2],[2]] => ", sess.run(hyp,feed_dict={X:[[1],[2],[2]]}) >0.5)
      print("X:[[1],[5],[5]] => ", sess.run(hyp,feed_dict={X:[[1],[5],[5]]}) >0.5)
      print("X:[[1,1],[4,3],[3,5]] => ", sess.run(hyp,feed_dict={X:[[1,1],[4,3],[3,5]]}) >0.5)

if __name__ == '__main__':
  tf.app.run()
