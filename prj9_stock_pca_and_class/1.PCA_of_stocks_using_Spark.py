
# coding: utf-8

# # How to work on this final
# 
# This final exam consists of **TWO** notebooks, **DO NOT HIT "SUBMIT" UNLESS YOU HAVE FINISHED BOTH NOTEBOOKS OR YOU WILL MISS THE OTHER NOTEBOOK, CAUSING YOU TO LOSE UP TO 84 POINTS AND POTENTIALLY FAIL TO RECEIVE YOUR CERTIFICATE.** The first notebook covers Spark and PCA using Spark. The second covers Boosted trees. 
# 
# You have two hours to complete this exam. The timer starts when you start the exam and increments even if you log out and log back in. In other words 
# 
# **YOU HAVE TWO WALL-CLOCK HOURS TO COMPLETE THE EXAM FROM START TO FINISH**
# 
# This exam is built around a single medium-sized dataset of the historical prices of stocks. Much of the content is an explanation of the data and the analysis approach. There are only 12 cells in the two notebooks where you need to enter your answers (4 in this notebook for 50 points, 7 in the next, for 84 points). In most cells you need to write only **one line of code**. Two hours should be plenty of time to finish this exam, assuming you reviewed the classes on PCA and on XGBoost and that you managed your time well.
# 
# Our recommendation is that you first jump to answer cells (search for "YOUR CODE HERE") so that you have an idea of what you need to do. **After** doing that read the notebook as a whole so that you understand the data and the context. You need to decide for yourself how much of the context is sufficient to come up with the answer.
# 
# In addition: you are allowed to consult:
# 1. the documentation for [python](https://docs.python.org/3/), [Jupyter Notebooks](https://jupyter-notebook.readthedocs.io/en/stable/), [numpy](https://docs.scipy.org/doc/numpy-1.14.0/reference/), [matplotlib](https://matplotlib.org/), [spark](https://spark.apache.org/docs/latest/), and [XGboost](https://xgboost.readthedocs.io/en/latest/python/python_intro.html)
# 2. Any article in [Wikipedia](https://www.wikipedia.org/).
# 
# Use your time well, **REMEMBER TO HIT SUBMIT BEFORE TIMER RUNS OUT OR YOUR FINAL WILL NOT BE SUBMITTED, RESULTING IN A 0 POINT ON YOUR FINAL**: focus on what you need to accomplish and the information you need.
# 
# Good Luck!

# # Introduction: determining stock sector from stock behaviours
# 
# In the next notebook, you will be given the history of stock prices for anonymous stocks and we will use a combination of **PCA** and **Boosting** to assign each stock to it's most likely sector. If you are new to stock price analysis this might seem like complete magic. However, as you will see, the predictions are based on a common sense understanding of the behaviour of stock prices. Much of the analysis has been done, your job is to fill in the missing parts correctly.
# 
# * In this notebook we will focus on preparing informative features using PCA.
# * In the next notebook we will use Boosted decision trees to predict the sector from the values of the features.

# ### Create a Spark context and import dependencies

# In[1]:


import pyspark
from pyspark import SparkContext

import os
os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

sc = SparkContext(master="local[4]",pyFiles=['lib/spark_PCA.py'])

from pyspark.sql import *
sqlContext = SQLContext(sc)

get_ipython().magic('pylab inline')
import sys
sys.path.append('./lib')

import numpy as np
import pickle

from spark_PCA import computeCov


# ### Read data
# 
# Read the file `SP500.csv` into a `Spark` dataframe.

# In[2]:


# read the file in a dataframe.
df=sqlContext.read.csv('data/SP500.csv',header='true',inferSchema='true')
columns=df.columns
col=[c for c in columns if '_D' in c]


# ### Visualize using Pandas
# To visualize the data we transform the `Spark` DataFrame to a `pandas` DataFrame. 
# 
# (((You don't need to know pandas to complete this final. However, if you are interested, you can take the first class in this micro-masters: DSE200x: Python for data analysis.)))

# In[3]:


Diffs=df.toPandas()
Diffs=Diffs.set_index('Date')
Diffs.tail(3)


# ### Explanation of the table
# * The rows of the table are indexed by date. The data starts in 1962, a time when most of the stocks we are tracking did not exist, and ends in 2015.
# * The columns come in pairs `XXX/YYY_P` and `XXX/YYY_D`. The prefix `XXX = train` for stocks whose *ticker* is `YYY` or  `XXX=test` for stocks whose ticker (and sector) are hidden. `YYY` for test columns is an integer index.
#    * The columns ending with `_P` give the **Adjusted Open Price** of stocks, which are the prices when the stock exchange opens in the morning. We use the **adjusted** prices which eliminate technical adjustments such as stock splits.
#    * The columns ending with `_D` are the log price ratio, as described below
# 
# 

# ### A naive plot of stocks.
# To get started, we plot the stock prices.

# In[4]:


# plot some stocks
Diffs[['train/AAPL_P','train/MSFT_P','train/IBM_P','test/8_P']].plot(figsize=(14,10));
plt.grid()


# ### A few observations
# 
# * Most of the stock grow over the long term, however, on the short term, their prices fluctuate significantly.
# * Suppose $p_s(t)$ is the price of stock $s$ on day $t$. It does not make much sense to consider **additive** changes to the stock price: $p(t+1)-p(t)$ becuase the changes in price of, say **IBM** around 1980 is much smaller than than change of the same stock around 2010. Instead, it makes more sense to consider the **relative** change in prices: $\frac{p(t+1)}{p(t)}$.
# * As it is convenient to make the price changes **additive** it makes sense to replace $\frac{p(t+1)}{p(t)}$ with $\log\frac{p(t+1)}{p(t)}$.
#    * This allows us to write $\frac{p(t+k)}{p(t)} = \sum_{i=1}^k \log \frac{p(t+i)}{p(t+i-1)}$
# * Unlike the price, whose value can change a lot between different times. The **log price ratio** $\log\frac{p(t+1)}{p(t)}$ is always close to zero: it is slightly above 0 if the price increased on a particular day and is slightly negative if it decreased.
# 
# To strengthen your intuition in this regard, consider **Black Monday**:

# ### Black Monday
# 
# One of the biggest crashes in the US stock market happened on
# the **Black Monday** on Oct 19 1987  
# 
# We will look at the stocks around that date

# In[5]:


#Focus on "Black Monday:" the stock crash of Oct 19 1987

import datetime
format = "%b-%d-%Y"

_from = datetime.datetime.strptime('Sep-1-1987', format)
_to = datetime.datetime.strptime('Nov-30-1987', format)

Diffs.loc[_from:_to,['train/AAPL_P','train/MSFT_P','train/IBM_P']].plot(figsize=(14,10));
plt.grid()


# **Why does it seems that the price of IBM fell much more than those of Apple and microsoft?**
# 
# Because IBM's price started so much higher. As explained above it is more informative to consider $\log\frac{p(t+1)}{p(t)}$

# In[6]:


Diffs.loc[_from:_to,['train/AAPL_D','train/MSFT_D','train/IBM_D']].plot(figsize=(14,10));
plt.grid()


# **Observe:** while it is hard to see the effect of the crash when considering the actual prices, plotting the log price ratio makes it clear that On october 19 all stocks went down significantly. Moreover, if you look at surrounding dates, indeed, if you look at **any** dates, you see that the price changes are highly correlated with each other, which, as we will explained below, is the rationale behind using **PCA** to relate the stocks.
# 
# Interestingly, the stocks regained most of the loss on Oct 20th, which shows that it is never a good idea to sell at a panic. *However* it is also not true that the stock market came back to where it was before the crash, that took much longer.

# ## Analysis of stock prices using Principal Component Analysis 
# As you have observed in the graphs above, the stock prices around black monday are highly correlated. This is hardly surprising, as that was a day where traders, and the public at large, have lost faith in the market and, in their panic, tried to sell their stocks, creating a snowball effect.
# 
# Luckily, such crashes are rare. However, the stock market is always volatile. The market reacts quickly to news and reflects their expected effect in the stock prices. Importantly, different news affect different sectors of the market in different ways. News related to home construction will affect the price of Real-Estate based stocks while news regarding changes in health-care policy will affect stocks in the health-care industry.
# 
# In fact, **even without knowing what the common cause is** we can use the correlations between stocks to identify stocks that belong to the same sector. This is where **PCA** comes into play. Finding the dominant eigenvectors of the log-price-ratio number will allow us to map the stocks into a low dimensional space in which stocks with similar dependencies on outside factors will be close to each other.

# #### Read data
# 
# Here, we read `Tickers.pkl` which is a dictionary with the keys: `Tickers` and `TickerInfo`. 
# 
# `Tickers` contains the ticker names and `TickerInfo` is a Pandas dataframe containing Company name, Sector and SectorID for each ticker.
# 
# We will use this table in the second notebook to define the label (sector) for each of the training stocks.

# In[7]:


import pandas as pd
TickerInfo=pd.read_csv('data/tickerInfo.tsv',sep='\t')
TickerInfo.head(5)


# ### Partition Columns

# In[8]:


#Extract the price/ratio columns(ending with _D)
#and partition those columns of df into a training set and a test set, each sorted alphabetically.
def partition_columns(df):
    train_col=[]
    test_col=[]
    columns=df.columns
    col=[c for c in columns if '_D' in c]
    for i in range(len(col)):
        if 'train' in col[i]:
            train_col.append(col[i])
        else:
            test_col.append(col[i])
    train_col = sorted(train_col)
    test_col=sorted(test_col)
    return  train_col+test_col


# In[9]:


columns = partition_columns(df)
df=df.select(columns)


# In[77]:


#grey2018 test
#columns[0:10]


# In[76]:


#grey2018 test
# df - spark DataFrame
#df.select(df['train/AAPL_D']).show()


# ### Create an RDD of numpy arrays
# In order to use the module `lib/spark_PCA.py` you need to transform the dataframe `df` into an RDD of numpy vectors.

# #### Function `make_array`
# Complete the function `make_array(row)` that takes as input a row of `df` and returns a numpy array (`dtype=np.float64`) that contains the values stored in the `col` fields of the Row. In other words, using each column in `col` as the index for the `row` parameter to get a value. Store all such values in a numpy array which you will return. Use `np.nan_to_num` to transform `nan`s into zeros.
# 
# * Direct Input: A row of real values
# * Indirect input: col - defines the columns of df that are to be used.,
# 
# * Output: numpy array of diff columns
# 
# **all but one of the lines of the function have been filled. You need only fill the one missing line that generates the numpy array from the row object**

# In[53]:


# the sequence of columns is define by the variable `col` in the immediate environment.
# create the "array" variable 
def make_array(row):
    ###
    ### YOUR CODE HERE
    ###
    
    #array = np.array(row).astype(np.float64)
    # grey2018: it works => why do we need col???????
    
    dim = len(col)
    array = np.ndarray(dim)
    
    for i in range(dim):
        #print(i, col[i], row[col[i]])
        array[i] = row[col[i]]
        
    array = array.astype(np.float64)
    
    
    array = np.nan_to_num(array)
    return array


# In[55]:


#grey2018 test
expected = pickle.load(open('data/row1.pkl', 'rb'))
#print(expected)
rows_test = df.take(1)
col = columns
observed = make_array(rows_test[0])
#print(type(observed), observed)


# In[56]:


#visible test
row1_correct=pickle.load(open('data/row1.pkl','rb'))
rows=df.take(2)
col=columns
row1=make_array(rows[1])
assert (type(row1)==numpy.ndarray), "make_array did not return the correct type"
assert (row1==row1_correct).all(),"make_array did not return the correct array"


# #### Create RDD
# Create an RDD from the all SP500 rows `df`. Then on this RDD, use Spark's `map` function in conjunction with our previously written `map_array` function to create an RDD of the numpy arrays specified by `columns`. Name your RDD with the name `Rows`. 
# 
# **You need only fill the one missing line that creates the RDD that satisfies the specification**

# In[64]:


# indirect parameter "col" for make_array function
col=columns

###
### YOUR CODE HERE
###

Rows = df.rdd.map(make_array)
#Rows.first()


# In[65]:


# TESTS
rowf = Rows.first()
assert type(Rows) == pyspark.rdd.PipelinedRDD, 'Incorrect return type'
assert type(rowf) == numpy.ndarray, 'Incorrect return type'
assert len(rowf) == 481, 'Incorrect dimensions'


# In[20]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### Compute covariance matrix
# 
# Here, we compute the covariance matrix of the data using `computeCov` in `spark_PCA.py`. The covariance matrix is of dimension `481 x 481`

# In[66]:


OUT=computeCov(Rows)
OUT.keys()


# ### Computing eigenvalues and eigenvectors

# In[67]:


from numpy import linalg as LA
def compute_eig(cov):
    eigval,eigvec=LA.eigh(cov)
    eigval=eigval[-1::-1] # reverse order
    eigvec=eigvec[:,-1::-1]
    return eigval, eigvec


# In[68]:


eigval_all,eigvec_all = compute_eig(OUT['Cov'])


# ### Function `compute_PCA`
# 
# Complete the function `compute_PCA` that takes as input a list of tickers and computes the eigenvalues and eigenvectors.
# 
# Input: `tickers` - list of tickers
# 
# Output: `eigval`, `eigvec` - numpy arrays of eigenvalues and eigenvectors
# 
# **all but one of the lines of the function have been filled. You need only fill the one missing line that creates the "OUT" variable, which represents the covariance matrix. Read the neighboring section of this notebook carefully for hints**
# 

# In[69]:


def compute_PCA(tickers):
    col=tickers
    Rows= df.select(col).rdd.map(make_array)
    ###
    ### YOUR CODE HERE
    ###
    
    OUT = computeCov(Rows)
    
    eigval, eigvec = compute_eig(OUT['Cov'])
    return eigval, eigvec


# In[70]:


## Compute the PCA of the test vectors alone
col = [c for c in columns if 'test' in c]
eigval_test,eigvec_test=compute_PCA(col)


# In[71]:


np.testing.assert_almost_equal(eigvec_test.dot(eigvec_test.T), eye(89), err_msg="eigvec_test not orthonormal")


# In[72]:


eigval_c,eigvec_c=pickle.load(open('data/PCAtest1.pkl','rb'))


# In[28]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[29]:


# HIDDEN TESTS
###
### AUTOGRADER TEST - DO NOT REMOVE
###


# In[30]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### Compute percentage-of-variance explained graph
# 
# In the cell below, write a function that plots the fraction of variance explained as a function of the number of top eigenvectors used. The function should return a numpy array where the value in location `i` is the fraction explained by the first `i+1` eigenvectors.
# 
# You should get a figure similar to this:
# 
# ![percent-var-explained](figs/percentageOfVarianceExplained.png)
# 
# Hint:
# 1. Pay close attention to the numbers on the axes.
# 
# **all but one of the lines of the function have been filled. You need only fill the one missing line that works on existing variable `cum`, making it contain the fractions of variance explained by different number of top eigenvectors. Careful examination of the meaning of each element in variable `cum` is strongly suggested**
# 

# In[73]:


def var_explained(eigval):
    cum=cumsum(eigval)
    
    ###
    ### YOUR CODE HERE
    ###
    
    cum = cum / sum(eigval)
    
    plot(cum[:50])
    grid()
    return cum

cum=var_explained(eigval_all)
print(cum[:5])


# In[75]:


assert abs(cum[-1] - 1) < 1e-8, 'The fraction of the variance explained by all eigenvectors is one'
for i in range(len(cum)-1):
    assert cum[i]<=cum[i+1], "cum cannot be decreasing"


# In[33]:


###
### AUTOGRADER TEST - DO NOT REMOVE
###


# ### Saving the information for the next stage

# In[78]:


from pickle import dump
dump({'columns':columns,
     'eigvec':eigvec_all,
     'eigval':eigval_all},
    open('data/PCA.pickle','wb'))
len(columns),eigvec_all.shape,eigval_all.shape


# ### Checking your calculations
# One good way to check your calculations is to create a scatter-plot projecting the data on two of the largest variance eigenvectors.
# 
# In the directory `figs` you will find scatter plots corresponding to the six combinations of the top 4 eigenvectors.
# 
# In these scatter-plots the ticker is replaced by the sector ID.
# 
# Stocks from the same sector tend to have similar fluctuations. That is because they have similar sensitivities to costs (labor cost, energy) and profits (income distribution, holiday shopping). For example check out `figs/scatter.2.1.pdf` in which regions that are dominated by Finance, Energy or IT have been marked. 
# 
# In this section, you will create similar scatter plots and compare them with the given graphs. Your scatter-plots will be slightly different, because of the stocks you have eliminated. But spectral analysis is pretty robust, so your scatter plots should be quite similar (remember that the inverse of an eigenvector is also an eigenvector, so horizontal or vertical reflections of the scatter plot are meaningless).

# #### Map tickers to Sector IDs

# In[79]:


def map_sectorID(columns):
    NN=TickerInfo[['Ticker','SECTOR_ID']]
    Ticker2Sector={ a[1]:a[2] for a in NN.to_records()}
    sectors=[]
    unknown=known=0
    for i in range(len(columns)):
        ticker=columns[i]
        if ticker[1] == 'r':
            ticker = ticker[6:]
        else:
            ticker = ticker[5:]
        ticker = ticker[:-2]
        if ticker in Ticker2Sector:
            #print(ticker)
            sectors.append(Ticker2Sector[ticker])
            known+=1
        else:
            sectors.append(ticker)
            unknown+=1
            #print('missing category for ',ticker)
    return sectors, known, unknown


# In[80]:


sectors, known, unknown = map_sectorID(columns)


# #### Generate Scatter plots
# 
# Complete the function `Scatter_Stocks` to generate a scatter plot of the stocks on the given pair of eigenvectors. The function takes as input the indices of the two eigenvectors and generates a scatter plot of the data projected on the pair of eigenvectors.
# 
# Input: <br>
# `eigvec` - Eigenvectors<br>
# `eigval` - Eigenvalues<br>
# `i0`, `i1` - Eigenvector indices
# 
# Example Input: i0=0, i1=2 (eigenvectors 0 and 2 - eigvec[:, 0] and eigvec[:, 2])
# 
# Steps:
# 1. Using the `plt.subplots` function, set the figure size to (20, 20) in order that the stock ticker names are readable. Store the objects returned by `plt.subplots` in `fig` and `ax`
# 2. Set the X and Y axis limits to the minimum and maximum of the eigenvectors to be plotted on each axis using the `plt.xlim` and `plt.ylim` functions
# 3. Label the axes as follows: Coeff 0, Coeff 1, using `plt.xlabel` and `plt.ylabel`
# 4. for each ticker in `columns` that you generated in `Partition Columns` section, call the `ax.annotate` function in `matplotlib` using the `ax` object returned in step `1` and annotate each point with the respective sectorID in `sectors`
# 5. The figure in `fig` is then saved according to the command given 

# In[81]:


def Scatter_Stocks(eigval,eigvec,i0=0,i1=1):
    fig, ax = plt.subplots(figsize=(20,20));  # In order that the stock ticker names are readable we make the plot very large
    plt.xlim([np.amin(eigvec[:,i0]),np.amax(eigvec[:,i0])]);
    plt.ylim([np.amin(eigvec[:,i1]),np.amax(eigvec[:,i1])]);
    plt.title('SP500 stocks scatter on '+str(i0)+', '+str(i1),fontsize=20);
    plt.xlabel('Coeff %d'%i0);
    plt.ylabel('Coeff %d'%i1);
    for i in range(len(columns)):
        ax.annotate(sectors[i], (eigvec[i,i0],eigvec[i,i1]),fontsize=10);
    
    fig.savefig('figs/scatter.'+str(i0)+'.'+str(i1)+'.pdf', format='PDF');
    # After exporting, we clear the figure so that the plot does not appear in the notebook.
    fig.clear();
    return None


# In[82]:


for i0 in range(4):
    for i1 in range(i0):
        print(i0,i1)
        Scatter_Stocks(eigval_all,eigvec_all,i0,i1);


# #  DO NOT HIT "SUBMIT" UNLESS YOU HAVE FINISHED BOTH NOTEBOOKS!

# # REMEMBER TO SUBMIT BEFORE TIMER RUN OUT OR YOUR FINAL WILL NOT BE SUBMITTED

# ## Check 
# Check that your `figs/scatter.2.1.pdf` is similar to `figs/scatter.2.1.annotated.pdf`. Note that the orientation of the eigenvectors can be flipped.

# In[83]:


# CHECKING YOUR GENERATED PDF LIKE THIS: 
from IPython.display import IFrame
IFrame("./figs/scatter.2.1.pdf", width=600, height=800)


# In[86]:


#grey2018 test
IFrame('./figs/scatter.2.1.annotated.pdf', width=600, height=800)


# Total points 50
