### Linear Regression example

This folder contains a normalized multi-variables Linear Regression example using a Gradient Descent

#### Prerequisistes

 The following dependencies should be installed first:
 * TensorFlow on a Raspberry Pi (C++ API and libraries) with the development environment (gcc-4.8,...)
 * TensorFlow on Linux or a Raspberry Pi (Python packages)

#### Training set:
 * ``data_houses.txt``: a matrice of 47 samples of houses with 2 features (surface in sqfeet and number of bedrooms) and for them the selling price. Taken from Andrew Ng's Coursera assignments.

#### Sourcecode:
 * ``linreg_multivar_trainer.py`` : python script to run on a PC or RPi (with Python packages built) to train the model and test it
 * ``inference.cc`` : the C++ sourcecode to run the model on the RPi and estimate the price of an house of 1650 sq ft and 3 bedrooms, as in the python script.

#### Training the model:

On the Python environment:
``python linreg_multivar_trainer.py``

#### Execution of the model:

On the Raspberry Pi
 1. Copy the folder linear_regression in ``{tensorflow_root}/tensorflow/contrib/pi_examples/``
 2. at ``{tensorflow_root}`` call: ``make -f tensorflow/contrib/pi_examples/linear_multivar_regression/Makefile``
 3. copy the exported model (data/) in ``{tensorflow_root}/tensorflow/contrib/pi_examples/linear_multivar_regression/data/``
 4. run the inferer: 
 
 ```
 cd {tensorflow_root}/tensorflow/contrib/pi_examples/linear_multivar_regression
 ./gen/bin/inference
 ````

#### Result

Normally you should see that : ``Estimated price for an house of 1650 sqfeet and 3 bedrooms: $293214``
