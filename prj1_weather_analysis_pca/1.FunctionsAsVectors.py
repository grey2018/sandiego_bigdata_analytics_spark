
# coding: utf-8

# In[1]:


get_ipython().magic('pylab inline')

import numpy as np
#import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual,widgets
import ipywidgets as widgets

print('version of ipwidgets=',widgets.__version__)

from lib.Reconstruction_plots import *
from lib.decomposer import Eigen_decomp
from lib.YearPlotter import YearPlotter

import warnings
warnings.filterwarnings('ignore')
_figsize=[8,6]


# ## High-dimensional vectors
# 
# We are used to work with 2 and 3 dimensional vectors, which we can think of as arrows in the plane or arrows in 3D space respectively.

# * How can we visualize vectors that are in dimension higher than 3?

# * One good way to visualize a $d$-dimensional vector is to draw it as a function from $1,2,\ldots,d$ to the reals.

# * All of the vector operations are well defined, including approximating a function using an orthonormal set of functions.

# To get an intuition about the working of the PCA, we used an example in the plane, or $R^2$.
# While useful for intuition, this is not the typical case in which we use PCA. Typically we are interested in vectors in a space whose dimension is in the hundreds or more.
# 
# How can we depict such vectors? If the coordinates of the vector have a natural order. For example, if the coordinates correspond to a grid of times, then a good representation is to make a plot in which the $x$-axis is the time and the $y$-axis is the value that corresponds to this time. 
# 
# Later in this class we will consider vectors that correspond to the temperature at a particular location each day of the year. These vectors will be of length 365 (we omit the extra days of leap years) and the PCA analysis will reveal the low dimensional subspace.

# ### Function approximation
# For now, we will consider the vectors that are defined by sinusoidal functions.

# In[2]:


# We define a grid that extends from o to 2*pi
step=2*pi/365
x=arange(0,2*pi,step)
len(x)


# #### Define an orthonormal set
# 
# The dimension of the space is 365 (arbitrary choice: the number of days in a year).
# 
# We define some functions based on $\sin()$ and $\cos()$ 

# In[3]:


c=sqrt(step/(pi))
v=[]
v.append(np.array(cos(0*x))*c/sqrt(2))
v.append(np.array(sin(x))*c)
v.append(np.array(cos(x))*c)
v.append(np.array(sin(2*x))*c)
v.append(np.array(cos(2*x))*c)
v.append(np.array(sin(3*x))*c)
v.append(np.array(cos(3*x))*c)
v.append(np.array(sin(4*x))*c)
v.append(np.array(cos(4*x))*c)

print("v contains %d vectors"%(len(v)))


# In[4]:


# plot some of the functions (plotting all of them results in a figure that is hard to read.
figure(figsize=_figsize)
for i in range(5):
    plot(x,v[i])
grid()
legend(['const','sin(x)','cos(x)','sin(2x)','cos(2x)']);


# #### Check that it is  an orthonormal basis
# This basis is not **complete** it does not span the space of all functions. It spans a 9 dimensional sub-space.
# 
# We will now check that this is an **orthonormal** basis. In other words, the length of each vector is 1 and every pair of vectors are orthogonal.

# In[5]:


for i in range(len(v)): 
    print()
    for j in range(len(v)):
        a=dot(v[i],v[j])
        a=round(1000*a+0.1)/1000
        print('%1.0f'%a, end=' ')


# #### Rewriting the set of vectors as a matrix
# 
# Combining the vectors as rows in a matrix allows us use very succinct (and very fast) matrix multiplications instead of for loops with vector products.

# In[6]:


U=vstack(v).transpose()
shape(U)


# ### Approximating an arbitrary function
# We now take an unrelated function $f=|x-4|$
# and see how we can use the basis matrix `U` to approximate it. 

# In[7]:


f1=abs(x-4)


# In[8]:


figure(figsize=_figsize)
plot(x,f1);
grid()


# #### Approximations  of increasing accuracy
# To understand how we can use a basis to approximate functions, we create a sequence of approximations $g(i)$ such that $g(i)$ is an approximation that uses the first $i$ vectors in the basis.
# 
# The larger $i$ is, the closer $g(i)$ is to $f$. Where the distance between $f$ and $g(i)$ is defined by the euclidean norm:
# $$   \| g(i)- f \|_2
# $$

# #### Plotting the approximations
# Below we show how increasing the number of vectors in the basis improves the approximation of $f$.

# In[9]:


eigen_decomp=Eigen_decomp(x,f1,np.zeros(len(x)),U)
recon_plot(eigen_decomp,year_axis=False,Title='Best Reconstruction',interactive=False,figsize=_figsize);


# In[10]:


eigen_decomp=Eigen_decomp(x,f1,np.zeros(len(x)),U)
plotter=recon_plot(eigen_decomp,year_axis=False,interactive=True,figsize=_figsize);
display(plotter.get_Interactive())


# #### Food for thought
# Visually, it is clear that $g(i)$ is getting close to $f$ as $i$ increases. To quantify the improvement, compute 
# $ \| g(i)- f \|_2 $ as a function of $i$

# ### Recovering from Noise

# In[11]:


noise=np.random.normal(size=x.shape)
f1=2*v[1]-4*v[5]
f2=f1+0.3*noise


# In[12]:


figure(figsize=_figsize)
plot(x,f1,linewidth=5);
plot(x,f2);
grid();


# In[13]:


eigen_decomp=Eigen_decomp(x,f2,np.zeros(len(x)),U)
plotter=recon_plot(eigen_decomp,year_axis=False,interactive=True,figsize=_figsize);
display(plotter.get_Interactive())


# ## Summary
# * Functions can be thought of as vectors and vice versa.
# * The **fourier** basis is a set of orthonormal fuctions made of `sin`s and `cosine`s
# * Orthonormal functions can be used to remove the noise added to an underlying distribution
