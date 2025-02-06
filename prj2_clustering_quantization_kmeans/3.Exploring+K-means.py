
# coding: utf-8

# ## K-means, measures of performance
# 
# In this notebook we will switch from using our own code to the KMeans implementation in SKLearn

# In[1]:


from sklearn.cluster import KMeans
get_ipython().magic('pylab inline')
import numpy as np
import pandas as pd

from lib.Kmeans_generate_and_analyze import *


# ## There are three main ways to evaluate the quality of a k-means result
# 1. Number of labeling errors.
# 2. Errors in the locations of the centroids.
# 3. RMSE

# ### number of labling errors
# * Consider a particular data point $\vec{x}$
# * We can think of the index $i$ of the closest centroid $\vec{c}_i$ as the preducted label of $\vec{x}$
# * In the good case, there is a 1-1 mapping between the true labels and the predicted labels such that most points match.
# * **Unrealistic:** Requires knowing the locations of the *True* centers.

# ### Distance between true centers and Centroids
# * We can match each centrer with the closest true centroid.
# * In the good case, the distances between each center and it's matched centroid is small (much smaller than the distance between the centers.
# * **Unrealistic:** Requires knowing the number and locations of the *True* centers.

# ### Elbow in the RMSE curve
# * The RMSE is the average square distance between a data point and the closest centroid.
# * Does not require additional knowledge.
# * **The Elbow** if there is a value of $k$ such that the RMSE decreases rapidly below that $k$ and decreases slowly above this $k$, then we can conclude that this value of $k$ is correct, i.e. is equal to the number of true centers.
# 
# <img alt="" src="Figs/Elbow.png" style="width:500px" />

# ## Analyzing KMeans under different settings. 

# #### Getting the documentation about `analyze`
# Be issuing the command `analyze?`
# 
# You recieve the following blurb:
# ```
# Signature: analyze(k, d, n, radius, plotRMSEbsK=True)
# Docstring:
# Generate k spherical gaussian clusters and analyze the performance of the Kmeans algorithm.
# The gaussian are placed at equal angular intervals on a circle of radius "radius" in dimensions 1 and 2.
#     Parameters:
#        k = number of generated clusters clusters
#        d = dimension of embedding space
#        n = number of examples per cluster
#        radius: the distance of the clusters from the origin (compare to std=1 for each cluster in each dimension)
# File:      ..../Section3-Kmeans-dim-reduction/BasicAnalysis/lib/Kmeans_generate_and_analyze.py
# Type:      function
# ```

# ### A first, very easy case
# with 4 clusters and a radius of 4 the clusters are very well separated and KMeans converges to a very good solution.
# * The RMSE has a clear elbow at $k=4$. In other words we can identify the number of clusters.
# 
# **Measures that require ground truth**  
# * There is only one classification mistake (the point with the blck half-circle at the bottom)
# * The locations of the centroids found by KMeans (yellow triangles) are very close to the true centers.

# In[2]:


X=analyze(k=4,d=2,n=100,radius=4)


# ### Decreasing the radius
# 
# Decreasing the radius of the circle on which the cluster centers are placed bring the clusters closer together so KMeans has a harder time finding the cluster centers. All measures degrade as the radius is decreased to 3 and then to 2.

# In[3]:


X=analyze(k=4,d=2,n=100,radius=3)


# #### When the radius is decreased to 2 even the true centers make mistakes
# Recall that the true label is associated with the spherical gaussian that **generated** the point. It might now be the cluster whose center is **closest** to the point

# In[4]:


X=analyze(k=4,d=2,n=100,radius=2)


# ### Reducing the number of points in each cluster
# Makes the centroids further from the centers. However if the clusters are sufficiently far from each other (radius =3) this does not hurt the labeling.

# In[5]:


X=analyze(k=4,d=2,n=100,radius=3)


# In[6]:


X=analyze(k=4,d=2,n=5,radius=3)


# In[7]:


X=analyze(k=4,d=2,n=100,radius=4)


# ### Increasing the number of clusters
# As we increase the number of clusters we create a ring.

# In[8]:


X=analyze(k=6,d=2,n=100,radius=4)


# In[9]:


X=analyze(k=8,d=2,n=100,radius=4)


# #### Wrong centroids but a good partition
# Even if the centroids are in the wrong place, they partition the ring into segments.
# * The centers are along a perfect circle
# * The centroids are along a distorted circle, but still a circle.

# In[10]:


X=analyze(k=16,d=2,n=100,radius=4)


# In[11]:


get_ipython().system('ls')


# ### Vector Quantization
# In the last example the clusters ae so close together that the resulting distribution is a more or less uniform ring.

# In this case it might be impossible to find the true centers of the clusters.

# This is a common situation, especially in the context of coding and lossy compression.

# In that case we are just interested in finding a set of centroids, called a **codebook** in this context.

# The compression scheme is simple, each vector is represented by the closest centroids. 
# * Assuming there are $k$ centroids, we can encode the identity of the centroid using $\log_2 k$ bits.
# * On the reciever end, the number $i$ corresponds to the centroid $\vec{c}_i$
# * This is called **vector quantization (VQ)**. 
# * For an example of using vector quantization to encode colors, see
# [this notebook](plot_color_quantization.ipynb)

# ### Increasing the dimension
# 
# The parameter $d$ controls the dimension of the **embedding space**. The centers are still along a circle in 2D. But the spherical gaussians are $d$ dimensional.
# 
# We always project on the two first dimensions - in which the centers define a circle.

# Strange things happen when the data is high dimensional:
# 1. The distance between a randomly selected pair of points becomes highly concentrated around one value.
# 2. The projection in which we see a ring gets harder and harder to find. Most random projections of the data will show a single gaussian.
# 3. The RMSE decreases with the number of centroids, but only very slowly.
# 3. KMeans breaks down.

# In[12]:


X=analyze(k=4,d=2,n=500,radius=2)


# #### Note the difference between zoomed and unzoomed scales
# As we increase the dimenssion, the RMSE changezs less and less with the increase in K.

# In[13]:


analyze(k=4,d=20,n=500,radius=2);


# #### When the dimension is larger than the  number of examples there is no hope
# The centroids have no relation to the centers or to the plane in which they lie.
# 
# Performance is very close to the performance if all the points were from a single spheical gaussian.

# In[14]:


X=analyze(k=4,d=500,n=100,radius=2)


# ## Summary
# * Performance of KMeans can be measured in various ways, but most require knowledge of the ground truth.
# * The main measure that does not require ground truth is the behaviour of the RMSE.
#   * We used it to estimate the number of centers. Can you think how to use it to estimate whether a particular centroid is in the location of the underlying centroid?

# ### Some experiments we did:
# * Making the clusters closer makes the clustering problem harder.
# * Decreasing the number of examples per cluster increases the error in the centroids, but it might not hurt the labeling error.
# * Increasing the number of clusters makes the distribution into a ring. K-means will find a good partition along the ring, but not necessarily the the one iduced by the original centers.
# * High dimensions breaks KMeans. The RMSE decreases very slowly with K.

# ##  See you next time!
