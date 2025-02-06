
# coding: utf-8

# ## Analyze whether early or late snow changes more year to year or place to place.
# 
# * We know from previous notebooks that the value of `coef_2` corresponds to whether the snow season is early or late. 
# * We want to study whether early/late season is more dependent on the year or on the location.
# * We will use RMS Error to quantify the strength of these dependencies.

# In[1]:


import pandas as pd
import numpy as np
import urllib
import math


# In[2]:


import os
os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

from pyspark import SparkContext
#sc.stop()
sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStatistics.py'])

from pyspark import SparkContext
from pyspark.sql import *
sqlContext = SQLContext(sc)


# In[3]:


get_ipython().magic('pylab inline')
import numpy as np
from lib.numpy_pack import packArray,unpackArray
from lib.spark_PCA import computeCov
from lib.computeStatistics import *


# In[4]:


### Read the data frame from pickle file

data_dir='../Data/Weather'
state='NY'
meas='SNWD'

from pickle import load

#read statistics
filename=data_dir+'/STAT_%s.pickle'%state
STAT,STAT_Descriptions = load(open(filename,'rb'))
print('keys from STAT=',STAT.keys())


# In[5]:


#!ls -ld $data_dir/*.parquet

#read data
filename=data_dir+'/decon_%s_%s.parquet'%(state,meas)

df=sqlContext.read.parquet(filename)
print(df.count())


# In[6]:


tmp=df.filter(df.Station=='USC00306411').toPandas()
tmp.head(1)


# In[7]:


#extract longitude and latitude for each station
feature='coeff_1'
sqlContext.registerDataFrameAsTable(df,'weather')
Features='station, year, coeff_2'
Query="SELECT %s FROM weather"%Features
print(Query)
pdf = sqlContext.sql(Query).toPandas()
pdf.head()


# In[8]:


year_station_table=pdf.pivot(index='year', columns='station', values='coeff_2')
year_station_table.tail(5)


# In[9]:


station_nulls=pd.isnull(year_station_table).mean()
station_nulls.hist();
xlabel('Fraction of years that are undefined')
ylabel('Number of stations')


# In[10]:


year_nulls=pd.isnull(year_station_table).mean(axis=1)
year_nulls.plot();
grid()
ylabel('fraction of stations that are undefined')


# In[11]:


pdf2=pdf[pdf['year']>1960]
year_station_table=pdf2.pivot(index='year', columns='station', values='coeff_2')
year_station_table.tail(5)


# In[12]:


station_nulls=pd.isnull(year_station_table).mean()
station_nulls.hist();
xlabel('Fraction of years that are undefined')
ylabel('Number of stations')


# ### Estimating the effect of the year vs the effect of the station
# 
# To estimate the effect of time vs. location on the second eigenvector coefficient we
# compute:
# 
# * The average row: `mean-by-station`
# * The average column: `mean-by-year`
# 
# We then compute the RMS before and after subtracting either  the row or the column vector.

# In[13]:


def RMS(Mat):
    return np.sqrt(np.nanmean(Mat**2))

mean_by_year=np.nanmean(year_station_table,axis=1)
mean_by_station=np.nanmean(year_station_table,axis=0)
tbl_minus_year = (year_station_table.transpose()-mean_by_year).transpose()
tbl_minus_station = year_station_table-mean_by_station

print('total RMS                   = ',RMS(year_station_table))
print('RMS removing mean-by-station= ',RMS(tbl_minus_station),'reduction=',RMS(year_station_table)-RMS(tbl_minus_station))
print('RMS removing mean-by-year   = ',RMS(tbl_minus_year),'reduction=',RMS(year_station_table)-RMS(tbl_minus_year))


# ### Conclusion Of Analysis
# The effect of time is about four times as large as the effect of location.

# ### Iterative reduction
# * After removing one component, the other component can have an effect.
# * We can use **alternating minimization** to remove the combined effect of location and time.

# In[14]:


T=year_station_table
print('initial RMS=',RMS(T))
for i in range(5):
    mean_by_year=np.nanmean(T,axis=1)
    T=(T.transpose()-mean_by_year).transpose()
    print(i,'after removing mean by year    =',RMS(T))
    mean_by_station=np.nanmean(T,axis=0)
    T=T-mean_by_station
    print(i,'after removing mean by stations=',RMS(T))


# In[15]:


T['mean_by_year']=mean_by_year
T['mean_by_year'].head()


# In[16]:


figure(figsize=(10,6))
T['mean_by_year'].plot();
grid()
title('A graph showing that in NY state, Snow season has been getting earlier ');


# ## Summary
# * The problem of missing data is prevalent and needs to be addressed.
# * RMS can be used to quantify the effect of different factors (here, time vs. space)
# * The snow season in NY has been getting earlier and earlier since 1960.

# In[ ]:




