
# coding: utf-8

# ## K-means, the theory

# ##  K-means improves RMSE each step
# 
# * **Centroids Set** $C_t$ : the set of $k$ centroids used in iteration $t$.
# * **Vector-representative association: **$\vec{c}_i(\vec{x})$ : the centroid $\vec{c} \in C_i$ that is closest to  $\vec{x}$.
# * **RMSE:** The sqrt of the mean square distance between a point and its representative centroid.
#   $$ RMSE = \sqrt{\frac{1}{|X|} \sum_{\vec{x} \in X} \|\vec{x} - \vec{c}(\vec{x})\|^2} $$

# ### Step 1: Updating association
# Associate each point $\vec{x} \in X$ to the closest centroid: $\vec{x} \in C_i$ where $\mbox{argmin}_i \|\vec{x} - \vec{c}_i\|_2$
# * Associating each point $\vec{x}$ with a closer centroid cannot increase RMSE.

# ### Step 2: Updating centers
# Update each centroid to be the mean of the points in $X$ that were assigned to it:
# $$ \vec{c}_{i+1}  = \frac{1}{m} \sum_{\vec{x} \in C_i} \vec{x}$$
# 
# * The point with minimal RMSE to a set of points is equal to the mean of those points.

# ### In each iteration of Kmeans, RMSE either decreases or stays the same.
# * If the association of points to centers does not change between iteraations. It never will

# ### Kmeans converges after a finite number of iterations
# 
# * Does not follow from the fact that RMSE decreases and cannot become negative.
# * **Think:** Can you come up with a strictly decreasing sequence of numbers that is never negative?

# ### The number of associations of points to centers is finite.
# 
# * As the number of points is finite (say $n$) and the number of centers is $k$, the number of possible asscoations is at most $k^n$.
# * Therefor the number of different RMSE values is finite.
# * Therefor the sequence of RMSE must coverge to a local minimum.

# ### Local minimum vs. Global Minimum
# * The **Global** minimum of the Kmeans problem is a set of $k$ representatives that minimizes the RMSE.
# * A **Local** minimum of the Kmeans problem is is a set of $k$ representatives that an iteration of the Kmeans algrithm does not change.
# * The global minimum is one of potentially many local minima.
# * The K-means algorithm does not converge to the global minimum.

# ### Finding the Global minimum is much much harder than finding a local minimum.
# * [Sanjoy Dasgupta](http://cseweb.ucsd.edu/~dasgupta/) has shown that [finding the global minimum is NP-hard]()
# * In other words, it is not just that Kmeans does not find the global minimum, but that, unless [**P=NP**](https://en.wikipedia.org/wiki/P_versus_NP_problem),
# there is **No** polynomial time algorithm for finding the global minimum.

# ## Summary
# * The mean distance between a point and it's closest centroid either decreases or stays the same forever.
# * Kmeans find a local minimum, which can be much worse than the global minimum.
# * Finding the global minimum is NP-hard.
# * Next time: different ways of measuring Kmeans performance.
