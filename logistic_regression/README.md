### Logistic Regression example

This folder contains a simple Logistic Regression example using a Gradient Descent

#### Prerequisistes

 The following dependencies should be installed first:
 * TensorFlow on a Raspberry Pi (C++ API and libraries) with the development environment (gcc-4.8,...)
 * TensorFlow on Linux or a Raspberry Pi (Python packages)

#### Training set:
 * ``data.txt``: a matrice of 6 samples with 3 features, taken from Kim Sung's examples.

#### Sourcecode:
 * ``logistic_reg_trainer.py`` : python script to run on a PC or RPi (with Python packages built) to train the model and inference some test data
 * ``inference.cc`` : the C++ sourcecode to run the model on the same test data, as in the python script.

#### Training the model:

On the Python environment:
``mkdir data``
``python logistic_reg_trainer.py`` will train the model and export it in ``data/``

#### Result

```
[[ 1.  1.  1.  1.  1.  1.]
 [ 3.  3.  3.  5.  7.  2.]
 [ 1.  2.  4.  5.  5.  5.]]
[ 0.  0.  0.  1.  1.  1.]
Found  6  samples with  3  features
Training
Epoch: 1000 cost= 0.234035 W= [[-1.21327817 -0.18834169  0.64121062]]
Epoch: 2000 cost= 0.215043 W= [[-1.64305842 -0.13860211  0.6916008 ]]
Epoch: 3000 cost= 0.199811 W= [[-2.02791286 -0.0959967   0.73897213]]
Epoch: 4000 cost= 0.187393 W= [[-2.37538123 -0.05974096  0.78418303]]
Epoch: 5000 cost= 0.177106 W= [[-2.69157314 -0.02874129  0.8274442 ]]
Epoch: 6000 cost= 0.168458 W= [[ -2.98142147e+00  -2.03500525e-03   8.68856788e-01]]
Epoch: 7000 cost= 0.161088 W= [[-3.24890614  0.02115369  0.90852392]]
Epoch: 8000 cost= 0.154731 W= [[-3.49726176  0.04143506  0.94655728]]
Epoch: 9000 cost= 0.149188 W= [[-3.7291286   0.05929188  0.98307848]]
Epoch: 10000 cost= 0.144306 W= [[-3.94667912  0.07510878  1.01819992]]
Optimization Finished!
Training cost= 0.144306 W= [[-3.94667912  0.07510878  1.01819992]]

graph definition saved in dir:  data
Converted 1 variables to const ops.
26 ops in the final graph.
Done exporting!
Testing the model
X:[[1],[2],[2]] =>  [[False]]
X:[[1],[5],[5]] =>  [[ True]]
X:[[1,1],[4,3],[3,5]] =>  [[False  True]]
```

#### Execution of the model:

On the Raspberry Pi
 1. Copy the folder linear_regression in ``{tensorflow_root}/tensorflow/contrib/pi_examples/``
 2. at ``{tensorflow_root}`` call: ``make -f tensorflow/contrib/pi_examples/logistic_regression/Makefile``
 3. copy the exported model (data/) in ``{tensorflow_root}/tensorflow/contrib/pi_examples/logistic_regression/data/``
 4. run the inferer: 
 
 ```
 cd {tensorflow_root}/tensorflow/contrib/pi_examples/logistic_regression
 ./gen/bin/inference
 ````

#### Result

Normally you should see something like that : 
```
I tensorflow/contrib/pi_examples/logistic_regression/inference.cc:72] Run the model
I tensorflow/contrib/pi_examples/logistic_regression/inference.cc:90] output_hyp: [[1],[2],[2]] => False
I tensorflow/contrib/pi_examples/logistic_regression/inference.cc:90] output_hyp: [[1],[5],[5]] => True
I tensorflow/contrib/pi_examples/logistic_regression/inference.cc:90] output_hyp: [[1],[4],[3]] => False
```
