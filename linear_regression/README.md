### Linear Regression example

This folder contains a very simple 1 variable Linear Regression exammple using a Gradient Descent

#### Prerequisistes

 The following dependencies should be installed first:
 * TensorFlow on a Raspberry Pi (C++ API and libraries) with kernel support for 'pow' with the development environment (gcc-4.8,...)
 * TensorFlow on Linux or a Raspberry Pi (Python packages)
 * matplotlib python's library (tru pip or apt-get, easier)

#### Training set:
 * an array of 8 samples with one feature (x) with their labels (y)

#### Sourcecode:
 * ``linreg_trainer.py`` : python script to run on a PC or RPi (with Python packages built) to train the model and test it
 * ``linreg_trainer_with_plot.py`` : python script to run on a PC or RPi (with Python packages built) to train the model, display trhe fitted line and test it
 * ``linreg_check.cc`` : the C++ sourcecode to run the model on the RPi and test it against the same testing data

#### Training the model:

On the Python environment:
``python linreg_trainer.py``

#### Execution of the model:

On the Raspberry Pi
 1. Copy the folder linear_regression in ``{tensorflow_root}/tensorflow/contrib/pi_examples/``
 2. at {tensorflow_root} call: ``make -f tensorflow/contrib/pi_examples/linear_regression/Makefile``
 3. copy the exported model (data/) in ``{tensorflow_root}/tensorflow/contrib/pi_examples/linear_regression/data/``
 4. run the checker: 
 
 ```
 cd {tensorflow_root}/tensorflow/contrib/pi_examples/linear_regression
 ./gen/bin/checker
 ````

#### Results

Normally you should see something like that (depending on your model training, values will differ a little) : 

```
I tensorflow/contrib/pi_examples/linear_regression/checker.cc:119] testing cost: 0.0780729
I tensorflow/contrib/pi_examples/linear_regression/checker.cc:120] training cost: 0.0770895
I tensorflow/contrib/pi_examples/linear_regression/checker.cc:128] Absolute mean square loss difference:-0.00098338

```
