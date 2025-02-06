
# coding: utf-8

# ## Logistic Regression using TensorFlow
# 
# * This notebook is adapted from [Aymeric Damian's logistic regression notebook](https://github.com/aymericdamien/TensorFlow-Examples/blob/master/notebooks/2_BasicModels/logistic_regression.ipynb) 
# 
# * Clone the full collection [here](https://github.com/aymericdamien/TensorFlow-Examples).

# ### multi-label logistic regression
# Is the next step after linear regression. 
# * Like linear regression, there are only an input layer and and output layer.
# * Unlike linear regression, the relation of output to input is not linear.
# * Also, we have ten output nodes, instead of one.

# [Logistic regression](https://en.wikipedia.org/wiki/Logistic_regression) refers to a soft-classifier over $k$ classes based on a linear function of the input. We will look at an example where we want to classify handwritten digits into one of $k=10$ classes: $0-9$
# 
# The logistic regression model works in a similar fashion to a linear regression model except that the final sum of the product between the weights and dependent variable is passed through a function that transforms the unbounded outputs of the linear operation into a normalized conditional probability over the $k$ classes.

# ## MNIST Dataset Overview
# 
# This example is using MNIST handwritten digits. The dataset contains 60,000 examples for training and 10,000 examples for testing. The digits have been size-normalized and centered in a fixed-size image (28x28 pixels) with values from 0 to 1. For simplicity, each image has been flattened and converted to a 1-D numpy array of 784 features (28*28).
# 
# ![MNIST Dataset](http://neuralnetworksanddeeplearning.com/images/mnist_100_digits.png)
# 
# More info: http://yann.lecun.com/exdb/mnist/

# ## The logistic model
# 
# We use the logistic model as a classifier which maps each digit image to an integer number between 0 and 9 which corresponds to the identity of the digit

# The inputs (placeholders) are:
# 
# * $X$ - a 784 dimensional vector.

# * $y$ - the label corresponding to $X$. Encoded using 1-hot encoding. I.e. **0**=(1,0,...0), **1**=(0,1,0,....,0) etc.

# There are 10 sets of parameters, one for each digit $j=0,\ldots,9$ :
# * $W_j$= a 784 dimensional vector
# * $b_j$= a scalar.

# The logistic funcion defines a distribution over the digits. We predict with the digit with the highest probability.
# $$
# p(y=j | X) = g(s_j)\;\;\mbox{ where }\;\; s_j=W_j \cdot X +b_j $$
# 

# $$\mbox{and  } g(s_j) = \frac{\exp(s_j)}{\sum_{i=0}^9 \exp(s_i)}
# $$is the [softmax function](https://en.wikipedia.org/wiki/softmax_function)

# ### The cross-entropy cost
# As our model outputs a vector of 10 conditional probabilities we use the negative cross-entropy as the cost function: 
# $$ Cost \left(\{W_j,b_j\}_{j=0}^9\right)
# =-\frac{1}{N} \sum_{i=1}^N \sum_{j=0}^n y^i_j \log g(s^i_j) $$

# ### Data Flow Diagram
# <img src="img/TensorFlow.png">

# ## Coding Logistic Regression in Tensorflow 

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 

import warnings
## Tensorflow produces a lot of warnings. We generally want to suppress them. The below code does exactly that. 
warnings.filterwarnings('ignore')
tf.logging.set_verbosity(tf.logging.ERROR)

rng = np.random
logs_path = 'logs/Logistic_regression'


# In[2]:


# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


# ### Defining the logistic model

# ```python
# # Placeholder1: flattened images of dimension 28*28 = 784
# x = tf.placeholder(dtype = tf.float32, shape = [None, 784], name = "inputData") 
# # Placeholder2: one-hot encoded labels for the 10 classes
# y = tf.placeholder(dtype = tf.float32, shape = [None, 10], name = "actualLabel")
# 
# W = tf.Variable(initial_value = tf.zeros([784, 10]), name = "weight")
# b = tf.Variable(initial_value = tf.zeros([10]), name = "bias")
# 
# with tf.name_scope('model'):
#     prediction = tf.nn.softmax(tf.add(b, tf.matmul(x, W))) 
# ```

# * The operation defined in out model is:
# ```python
# prediction = tf.nn.softmax(tf.add(b, tf.matmul(x, W))) # Softmax
# ```
# * it is a composition of:
#    * `tf.matmul(x, W))` : performs dot product between the input vector `x` and the weights matrix `W`, yielding a vector of dimension 10.
#    * `tf.add(b, tf.matmul(x,W))` : returns the tensor sum between the tensors b and the output of the inner computation 
#    * `tf.nn.softmax(A)` : applies the softmax function on each value of the input tensor (default is along the first dimension)

# In[3]:


# Lets run the code we just described

x = tf.placeholder(dtype = tf.float32, shape = [None, 784], name = "inputFeatures") # mnist data image of shape 28*28=784
y = tf.placeholder(dtype = tf.float32, shape = [None, 10], name = "actualLabel") # 0-9 digits recognition => 10 classes

W = tf.Variable(initial_value = tf.zeros([784, 10]), name = "weight")
b = tf.Variable(initial_value = tf.zeros([10]), name = "bias")

with tf.name_scope('model'):
    prediction = tf.nn.softmax(tf.add(b, tf.matmul(x, W))) # Softmax


# ### Adding a regularization term
# One way to reduce over-fitting is to add a **regularization term** to the loss. 
# This term is also referred to as **weight decay** because it pushes the weights towards zero.

# We use an L2 regularizer, given the weight vectors $W_j$ and the biases $b_j$ 
# the regularization term is 
# $$ l2\left(\{W_j,b_j\}_{j=0}^9\right) = \sum_{j=0}^9  \left[\sum_{i=1}^{784} W_{ji}^2 +b_j^2\right] $$

# In[4]:


# Parameters
#learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 5
lamb = 0.01 #This is the hyperparameter that controls the strength of the regularization

# Minimize error using cross entropy loss
# reduce_mean calculates the mean across dimensions of a tensor
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(prediction), axis=1)  
                      + lamb * (tf.nn.l2_loss(W) + tf.nn.l2_loss(b)))
                     
# Logging commands
tf.summary.scalar("loss", loss)
merged_summary_op = tf.summary.merge_all()
                          
with tf.name_scope('Optimizer'):
    optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# In[5]:


init = tf.global_variables_initializer()


# ### Executing the optimization in a session

# In[6]:


# Start training
sess=tf.Session()
sess.run(init)

summary_writer = tf.summary.FileWriter(logs_path + "/logistic", graph=tf.get_default_graph())


# In[7]:


# Training cycle
for epoch in range(training_epochs):
    avg_loss = 0.
    total_batch = int(mnist.train.num_examples/batch_size) # there would be 600 batches

    # Loop over all batches
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)

        # Fit training using batch data
        _, c = sess.run([optimizer, loss], feed_dict={x: batch_xs,
                                                      y: batch_ys})
        # Compute average loss
        avg_loss += c / total_batch

    # Display logs per epoch step
    if (epoch+1) % display_step == 0:
        print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss))

print("Optimization Finished!")


# #### Optional excercise
# * Add to the log print line the two components of the loss: the entropy loss and the regularization term.

# In[8]:


# Calculate test set accuracy, i.e. number of mistakes final model makes on test set
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))

# Calculate accuracy for 3000 examples
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:", accuracy.eval({x: mnist.test.images[:3000], y: mnist.test.labels[:3000]},session=sess))


# ## Using Tensorboard to View Graph Structure 

# We can have a look at the computational graph that we have just defined on Tensorboard. We have installed a jupyter extension that makes connecting to Tensorboard very simple. To do this, 
# 
# In your Jupyter directory tree view, select the log directory for lesson 1 and click the <font color = "red">**Tensorboard**</font> button as shown in the picture.
# <img src = "img/TensorboardInit1.PNG">

# Next, go to the <font color = "red">**Running**</font> tab, and choose the Tensorboard instance corresponding to the correct log directory as shown in the screenshot.
# <img src = "img/TensorboardInit2.PNG">
# 
# 
