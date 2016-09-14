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
