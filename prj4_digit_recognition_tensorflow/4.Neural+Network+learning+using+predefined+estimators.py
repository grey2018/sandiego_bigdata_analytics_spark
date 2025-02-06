
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np

import warnings
## Tensorflow produces a lot of warnings. We generally want to suppress them. The below code does exactly that. 
warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)


# ## Using predefined estimators
# In this notebook we recreate the neural network defined first in notebook [3.Neural-Networks.ipynb](3.Neural-Networks.ipynb)  
# using the predefined estimator `DNNClassifier`

# ## Benefits of Estimators
# 
# - Estimator-based models are independent of operating environment
#     - local host
#     - GPUs
#     - CPU clusters

# ### More benefits
# - Simplify model sharing between developers
# - State of the art model architectures with more intuitive high-level code

# Consult https://www.tensorflow.org/programmers_guide/estimators for more advantages of using Estimators as described by the developers of TensorFlow.

# ## Read Data
# The MNist dataset is available from within TensorFlow tutorials.

# In[2]:


from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

# Wrap input as a function (THE "input function" will be defined below)
def input(dataset):
    return dataset.images, dataset.labels.astype(np.int32)


# ##  Define feature columns

# In[3]:


# Specify feature
feature_columns = [tf.feature_column.numeric_column("x", shape=[28, 28])]


# ## Define Neural Network

# In[4]:


# Build 2 layer DNN classifier
classifier = tf.estimator.DNNClassifier(
    feature_columns=feature_columns,
    hidden_units=[256, 256],
    optimizer=tf.train.AdamOptimizer(1e-4),
    n_classes=10,
    dropout=0.1,
    model_dir="./tmp/mnist_model_256_256"   # Location for storing checkpoints.
)


# ## Define training input function
# * Supplies data for training, evaluation, prediction
# * Should yield tuples of:
#     - Python dict `features`: key = name of feature, value = array of feature values
#     - Array `label` : label for every example

# In[5]:


# Define the training inputs
train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.train)[0]},
    y=input(mnist.train)[1],
    num_epochs=None,
    batch_size=50,
    shuffle=True
)


# ## Train the neural network
# * Checkpoint used for "warm start"
# * Checkpoints saved

# In[6]:


tf.logging.set_verbosity(tf.logging.INFO)
classifier.train(input_fn=train_input_fn, steps=10000)


# In[7]:


# Have a look at the checkpoint directory.
get_ipython().system('ls -lrt ./tmp/mnist_model_256_256/ | tail')


# ## Define test input function

# In[8]:


test_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": input(mnist.test)[0]},
    y=input(mnist.test)[1],
    num_epochs=1,
    shuffle=False
)


# ## Evaluate accuracy

# In[9]:


accuracy_score = classifier.evaluate(input_fn=test_input_fn)["accuracy"]
print("\nTest Accuracy: {0:f}%\n".format(accuracy_score*100))


# In[ ]:




