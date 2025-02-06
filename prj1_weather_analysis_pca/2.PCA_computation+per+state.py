
# coding: utf-8

# ## Computing PCA using RDDs

# ##  PCA
# 
# The vectors that we want to analyze have length, or dimension, of 365, corresponding to the number of 
# days in a year.
# 
# We will perform [Principle component analysis (PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis)
# on these vectors. There are two steps to this process:

# 1) Computing the covariance matrix: this is a  simple computation. However, it takes a long time to compute and it benefits from using an RDD because it involves all of the input vectors.

# 2) Computing the eigenvector decomposition. this is a more complex computation, but it takes a fraction of a second because the size to the covariance matrix is $365 \times 365$, which is quite small. We do it on the head node usin `linalg`

# ### Computing the covariance matrix
# Suppose that the data vectors are the column vectors denoted $x$ then the covariance matrix is defined to be
# $$
# E(x x^T)-E(x)E(x)^T
# $$
# 
# Where $x x^T$ is the **outer product** of $x$ with itself.

# If the data that we have is $x_1,x_2,x_n$ then  we estimate the covariance matrix:
# $$
# \hat{E}(x x^T)-\hat{E}(x)\hat{E}(x)^T
# $$
# 
# the estimates we use are:
# $$
# \hat{E}(x x^T) = \frac{1}{n} \sum_{i=1}^n x_i x_i^T,\;\;\;\;\;
# \hat{E}(x) = \frac{1}{n} \sum_{i=1}^n x_i
# $$

# ## Computing the covariance matrix where the `nan`s are
# ### The effect of  `nan`s in arithmetic operations
# * We use an RDD of numpy arrays, instead of Dataframes.
# * Why? Because unlike dataframes, `numpy.nanmean` treats `nan` entries correctly.

# ### Calculating the mean of a vector with nan's
# * We often get vectors $x$ in which some, but not all, of the entries are `nan`. 
# * We want to compute the mean of the elements of $x$. 
# * If we use `np.mean` we will get the result `nan`. 
# * A useful alternative is to use `np.nanmean` which removes the `nan` elements and takes the mean of the rest.

# In[1]:


import numpy as np
a=np.array([1,np.nan,2,np.nan,3,4,5])
print('a=',a)
print('np.mean(a)=',np.mean(a))
print('np.mean(np.nan_to_num(a))=',np.mean(np.nan_to_num(a))) # =(1+0+2+0+3+4+5)/7
print('np.nanmean(a)=',np.nanmean(a)) # =(1+2+3+4+5)/5


# ### The outer poduct of a vector with `nan`s with itself

# In[2]:


np.outer(a,a)


# ### When should you not use `np.nanmean` ?
# Using `n.nanmean` is equivalent to assuming that choice of which elements to remove is independent of the values of the elements. 
# * Example of bad case: suppose the larger elements have a higher probability of being `nan`. In that case `np.nanmean` will under-estimate the mean

# ### Computing the covariance  when there are `nan`s
# The covariance is a mean of outer products.
# 
# We calculate two matrices:
# * $S$ - the sum of the matrices, whereh `nan`->0
# * $N$ - the number of not-`nan` element for each matrix location.
# 
# We then calculate the mean as $S/N$ (division is done element-wise)

# ## Computing the mean together with the covariance
# To compute the covariance matrix we need to compute both $\hat{E}(x x^T)$ and $\hat{E}(x)$. Using a simple trick, we can compute both at the same time.

# Here is the trick: lets denote a $d$ dimensional **column vector** by $\vec{x} = (x_1,x_2,\ldots,x_d)$ (note that the subscript here is the index of the coordinate, not the index of the example in the training set as used above). 
# 
# The augmented vector $\vec{x}'$ is defined to be the $d+1$ dimensional vector $\vec{x}' = (1,x_1,x_2,\ldots,x_d)$.

# The outer product of $\vec{x}'$ with itself is equal to 
# 
# $$ \vec{x}' {\vec{x}'}^T
# = \left[\begin{array}{c|ccc}
#     1 &  &{\vec{x}}^T &\\
#     \hline \\
#     \vec{x} & &\vec{x} {\vec{x}}^T \\ \\
#     \end{array}
#     \right]
# $$
# 
# Where the lower left matrix is the original outer product $\vec{x} {\vec{x}}^T$ and the first row and the first column are $\vec{x}^T$ and $\vec{x}$ respectively.

# Now suppose that we do the take the average of the outer product of the augmented vector and convince yourself that:
# $$
# \hat{E}(\vec{x}' {\vec{x}'}^T) = \frac{1}{n} \sum_{i=1}^n {\vec{x}'}_i {\vec{x}'}_i^T
# = \left[\begin{array}{c|ccc}
#     1 &  &\hat{E}(\vec{x})^T &\\
#     \hline \\
#     \hat{E}(\vec{x}) & &\hat{E}(\vec{x} {\vec{x}}^T) \\ \\
#     \end{array}
#     \right]
# $$
# 
# So indeed, we have produced the outer product average together with (two copies of) the average $\hat{E}(\vec{x})$

# In[3]:


# Set to True if running notebook on AWS/EMR
EMR=False 


# In[4]:


import os
os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

from pyspark import SparkContext,SparkConf

def create_sc(pyFiles):
    sc_conf = SparkConf()
    sc_conf.setAppName("Weather_PCA")
    sc_conf.set('spark.executor.memory', '3g')
    sc_conf.set('spark.executor.cores', '1')
    sc_conf.set('spark.cores.max', '4')
    sc_conf.set('spark.default.parallelism','10')
    sc_conf.set('spark.logConf', True)
    print(sc_conf.getAll())

    sc = SparkContext(conf=sc_conf,pyFiles=pyFiles)

    return sc 

sc = create_sc(pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStatistics.py'])


# In[5]:


from pyspark.sql import *
sqlContext = SQLContext(sc)

import numpy as np
from lib.computeStatistics import *


# ### Climate data
# 
# The data we will use here comes from [NOAA](https://www.ncdc.noaa.gov/). Specifically, it was downloaded from This [FTP site](ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/).
# 
# There is a large variety of measurements from all over the world, from 1870 will 2012.
# in the directory `../../Data/Weather` you will find the following useful files:
# 
# * data-source.txt: the source of the data
# * ghcnd-readme.txt: A description of the content and format of the data
# * ghcnd-stations.txt: A table describing the Meteorological stations.
# 
# 

# ### Data cleaning
# 
# * Most measurements exists only for a tiny fraction of the stations and years. We therefor restrict our use to the following measurements:
# ```python
# ['TMAX', 'SNOW', 'SNWD', 'TMIN', 'PRCP', 'TOBS']
# ```
# 
# * 8 We consider only measurement-years that have at most 50 `NaN` entries
# 
# * We consider only measurements in the continential USA
# 
# * We partition the stations into the states of the continental USA (plus a few stations from states in canada and mexico).

# In[6]:


state='NY'
if not EMR:
    data_dir='../Data/Weather'
    tarname=state+'.tgz'
    parquet=state+'.parquet'

    get_ipython().system('rm -rf $data_dir/$tarname')

    command="curl https://mas-dse-open.s3.amazonaws.com/Weather/by_state/%s > %s/%s"%(tarname,data_dir,tarname)
    print(command)
    get_ipython().system('$command')
    get_ipython().system('ls -lh $data_dir/$tarname')

    cur_dir, = get_ipython().getoutput('pwd')
    get_ipython().magic('cd $data_dir')
    get_ipython().system('tar -xzf $tarname')
    get_ipython().system('du ./$parquet')
    get_ipython().magic('cd $cur_dir')


# In[7]:


if EMR:  # not debugged, should use complete parquet and extract just the state of interest.
    data_dir='/mnt/workspace/Data'
    get_ipython().system('hdfs dfs -mkdir /weather/')
    get_ipython().system('hdfs dfs -CopyFromLocal $data_dir/$parquet /weather/$parquet')

    # When running on cluster
    #!mv ../../Data/Weather/NY.parquet /mnt/workspace/Data/NY.parquet

    get_ipython().system('aws s3 cp --recursive --quiet /mnt/workspace/Data/NY.parquet s3://dse-weather/NY.parquet')

    get_ipython().system('aws s3 ls s3://dse-weather/')

    local_path=data_dir+'/'+parquet
    hdfs_path='/weather/'+parquet
    local_path,hdfs_path

    get_ipython().system('hdfs dfs -copyFromLocal $local_path $hdfs_path')

    get_ipython().system('hdfs dfs -du /weather/')
    parquet_path=hdfs_path


# In[8]:


parquet_path = data_dir+'/'+parquet
get_ipython().system('du -sh $parquet_path')


# In[9]:


get_ipython().run_cell_magic('time', '', 'df=sqlContext.read.parquet(parquet_path)\nprint(df.count())\ndf.show(5)')


# In[10]:


sqlContext.registerDataFrameAsTable(df,'table')


# In[11]:


Query="""
SELECT Measurement,count(Measurement) as count 
FROM table 
GROUP BY Measurement
ORDER BY count
"""
counts=sqlContext.sql(Query)
counts.show()


# In[12]:


from time import time
t=time()

N=sc.defaultParallelism
print('Number of executors=',N)
print('took',time()-t,'seconds')


# In[13]:


get_ipython().system('ls lib')


# In[14]:


# %load lib/spark_PCA.py
import numpy as np
from numpy import linalg as LA

def outerProduct(X):
    """Computer outer product and indicate which locations in matrix are undefined"""
    O=np.outer(X,X)
    N=1-np.isnan(O)
    return (O,N)

def sumWithNan(M1,M2):
    """Add two pairs of (matrix,count)"""
    (X1,N1)=M1
    (X2,N2)=M2
    N=N1+N2
    X=np.nansum(np.dstack((X1,X2)),axis=2)
    return (X,N)

### HW: Replace the RHS of the expressions in this function (They need to depend on S and N.
def HW_func(S,N):
    E=      np.ones([365]) # E is the sum of the vectors
    NE=     np.ones([365]) # NE is the number of not-nan antries for each coordinate of the vectors
    Mean=   np.ones([365]) # Mean is the Mean vector (ignoring nans)
    O=      np.ones([365,365]) # O is the sum of the outer products
    NO=     np.ones([365,365]) # NO is the number of non-nans in the outer product.
    return  E,NE,Mean,O,NO


# In[17]:


def computeCov(RDDin):
    """computeCov recieves as input an RDD of np arrays, all of the same length, 
    and computes the covariance matrix for that set of vectors"""
    RDD=RDDin.map(lambda v:np.array(np.insert(v,0,1),dtype=np.float64)) # insert a 1 at the beginning of each vector so that the same 
                                           #calculation also yields the mean vector
    OuterRDD=RDD.map(outerProduct)   # separating the map and the reduce does not matter because of Spark uses lazy execution.
    (S,N)=OuterRDD.reduce(sumWithNan)

    E,NE,Mean,O,NO=HW_func(S,N)

    Cov=O/NO - np.outer(Mean,Mean)
    # Output also the diagnal which is the variance for each day
    Var=np.array([Cov[i,i] for i in range(Cov.shape[0])])
    return {'E':E,'NE':NE,'O':O,'NO':NO,'Cov':Cov,'Mean':Mean,'Var':Var}


# In[18]:


if __name__=="__main__":
    # create synthetic data matrix with j rows and rank k
    
    V=2*(np.random.random([2,10])-0.5)
    data_list=[]
    for i in range(1000):
        f=2*(np.random.random(2)-0.5)
        data_list.append(np.dot(f,V))
    # compute covariance matrix
    RDD=sc.parallelize(data_list)
    OUT=computeCov(RDD)

    #find PCA decomposition
    eigval,eigvec=LA.eig(OUT['Cov'])
    print('eigval=',eigval)
    print('eigvec=',eigvec)


# In[19]:


get_ipython().run_cell_magic('writefile', 'lib/tmp', '# %load lib/computeStatistics.py\n\n\nfrom numpy import linalg as LA\nimport numpy as np\n\nfrom numpy_pack import packArray,unpackArray\nfrom spark_PCA import computeCov\nfrom time import time\n\ndef computeStatistics(sqlContext,df):\n    """Compute all of the statistics for a given dataframe\n    Input: sqlContext: to perform SQL queries\n            df: dataframe with the fields \n            Station(string), Measurement(string), Year(integer), Values (byteArray with 365 float16 numbers)\n    returns: STAT, a dictionary of dictionaries. First key is measurement, \n             second keys described in computeStats.STAT_Descriptions\n    """\n\n    sqlContext.registerDataFrameAsTable(df,\'weather\')\n    STAT={}  # dictionary storing the statistics for each measurement\n    measurements=[\'TMAX\', \'SNOW\', \'SNWD\', \'TMIN\', \'PRCP\', \'TOBS\']\n    \n    for meas in measurements:\n        t=time()\n        Query="SELECT * FROM weather\\n\\tWHERE measurement = \'%s\'"%(meas)\n        mdf = sqlContext.sql(Query)\n        print(meas,\': shape of mdf is \',mdf.count())\n\n        data=mdf.rdd.map(lambda row: unpackArray(row[\'Values\'],np.float16))\n\n        #Compute basic statistics\n        STAT[meas]=computeOverAllDist(data)   # Compute the statistics \n\n        # compute covariance matrix\n        OUT=computeCov(data)\n\n        #find PCA decomposition\n        eigval,eigvec=LA.eig(OUT[\'Cov\'])\n\n        # collect all of the statistics in STAT[meas]\n        STAT[meas][\'eigval\']=eigval\n        STAT[meas][\'eigvec\']=eigvec\n        STAT[meas].update(OUT)\n\n        print(\'time for\',meas,\'is\',time()-t)\n    \n    return STAT\n\n# Compute the overall distribution of values and the distribution of the number of nan per year\ndef find_percentiles(SortedVals,percentile):\n    L=int(len(SortedVals)/percentile)\n    return SortedVals[L],SortedVals[-L]\n  \ndef computeOverAllDist(rdd0):\n    UnDef=np.array(rdd0.map(lambda row:sum(np.isnan(row))).sample(False,0.01).collect())\n    flat=rdd0.flatMap(lambda v:list(v)).filter(lambda x: not np.isnan(x)).cache()\n    count,S1,S2=flat.map(lambda x: np.float64([1,x,x**2]))\\\n                  .reduce(lambda x,y: x+y)\n    mean=S1/count\n    std=np.sqrt(S2/count-mean**2)\n    Vals=flat.sample(False,0.0001).collect()\n    SortedVals=np.array(sorted(Vals))\n    low100,high100=find_percentiles(SortedVals,100)\n    low1000,high1000=find_percentiles(SortedVals,1000)\n    return {\'UnDef\':UnDef,\\\n          \'mean\':mean,\\\n          \'std\':std,\\\n          \'SortedVals\':SortedVals,\\\n          \'low100\':low100,\\\n          \'high100\':high100,\\\n          \'low1000\':low100,\\\n          \'high1000\':high1000\n          }\n\n# description of data returned by computeOverAllDist\nSTAT_Descriptions=[\n(\'SortedVals\', \'Sample of values\', \'vector whose length varies between measurements\'),\n (\'UnDef\', \'sample of number of undefs per row\', \'vector whose length varies between measurements\'),\n (\'mean\', \'mean value\', ()),\n (\'std\', \'std\', ()),\n (\'low100\', \'bottom 1%\', ()),\n (\'high100\', \'top 1%\', ()),\n (\'low1000\', \'bottom 0.1%\', ()),\n (\'high1000\', \'top 0.1%\', ()),\n (\'E\', \'Sum of values per day\', (365,)),\n (\'NE\', \'count of values per day\', (365,)),\n (\'Mean\', \'E/NE\', (365,)),\n (\'O\', \'Sum of outer products\', (365, 365)),\n (\'NO\', \'counts for outer products\', (365, 365)),\n (\'Cov\', \'O/NO\', (365, 365)),\n (\'Var\', \'The variance per day = diagonal of Cov\', (365,)),\n (\'eigval\', \'PCA eigen-values\', (365,)),\n (\'eigvec\', \'PCA eigen-vectors\', (365, 365))\n]\n\n')


# In[20]:


get_ipython().run_cell_magic('time', '', '### This is the main cell, where all of the statistics are computed.\nSTAT=computeStatistics(sqlContext,df)')


# In[21]:


print("   Name  \t                 Description             \t  Size")
print("-"*80)
print('\n'.join(["%10s\t%40s\t%s"%(s[0],s[1],str(s[2])) for s in STAT_Descriptions]))


# In[22]:


## Dump STAT and STST_Descriptions into a pickle file.
from pickle import dump

filename=data_dir+'/STAT_%s.pickle'%state
dump((STAT,STAT_Descriptions),open(filename,'wb'))
get_ipython().system('ls -l $data_dir')


# In[23]:


X=STAT['TMAX']['Var']
for key in STAT.keys():
    Y=STAT[key]['Var']
    print(key,sum(abs(X-Y)))


# In[24]:


get_ipython().system('ls -l ../Data/Weather/STAT*')


# In[25]:


get_ipython().system('gzip -f -k ../Data/Weather/STAT*.pickle')
get_ipython().system('ls -l ../Data/Weather/STAT*')


# ### Summary
# * We discussed how to compute the covariance matrix and the expectation matrix when there are `nan` entries.
# * The details are all in `computeStatistics`, which is defined in python files you can find in the directory `lib`
