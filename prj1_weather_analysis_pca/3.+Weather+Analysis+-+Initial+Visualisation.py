
# coding: utf-8

# ## Weather Data : Initial Visualization
# 
# ### For New York State

# In[1]:


import os
os.environ["PYSPARK_PYTHON"]="python3"
os.environ["PYSPARK_DRIVER_PYTHON"] = "python3"

import pandas as pd
import numpy as np
import sklearn as sk
import urllib
import math
get_ipython().magic('pylab inline')

from pyspark import SparkContext
#sc.stop()
sc = SparkContext(master="local[3]",pyFiles=['lib/numpy_pack.py','lib/spark_PCA.py','lib/computeStatistics.py'])

from pyspark import SparkContext
from pyspark.sql import *
import pyspark.sql
sqlContext = SQLContext(sc)

import numpy as np
from lib.numpy_pack import packArray,unpackArray
from lib.spark_PCA import computeCov
from lib.computeStatistics import *


# In[2]:


import warnings  # Suppress Warnings
warnings.filterwarnings('ignore')

_figsize=(10,7)


# ## Read Data
# 
# ### Read Raw data for state

# In[3]:


state='NY'
data_dir='../Data/Weather'

tarname=state+'.tgz'
parquet=state+'.parquet'

get_ipython().magic('mkdir -p $data_dir')
get_ipython().system('rm -rf $data_dir/$tarname')

command="curl https://mas-dse-open.s3.amazonaws.com/Weather/by_state/%s > %s/%s"%(tarname,data_dir,tarname)
print(command)
get_ipython().system('$command')
get_ipython().system('ls -lh $data_dir/$tarname')


# In[4]:


get_ipython().system('ls $data_dir')


# In[5]:


cur_dir, = get_ipython().getoutput('pwd')
get_ipython().magic('cd $data_dir')
get_ipython().system('tar -xzf $tarname')
get_ipython().system('du ./$parquet')
get_ipython().magic('cd $cur_dir')


# In[6]:


get_ipython().system('du -h $data_dir/$parquet')


# In[7]:


print(parquet)
weather_df=sqlContext.read.parquet(data_dir+'/'+parquet)
#weather_df=weather_df.drop('State') # we drop State because it already exists in "Stations".


# In[8]:


get_ipython().run_cell_magic('time', '', 'weather_df.count()')


# In[9]:


print('number of rows=',weather_df.cache().count())
weather_df.show(5)


# ### read statistics information for state.

# In[10]:


#read statistics
filename='STAT_%s.pickle'%state
command="curl https://mas-dse-open.s3.amazonaws.com/Weather/by_state_2/%s.gz > %s/%s.gz"%(filename,data_dir,filename)
print(command)
get_ipython().system('$command')


# In[11]:


gzpath='%s/%s.gz'%(data_dir,filename)
print(gzpath)
get_ipython().system('ls -l $gzpath')
get_ipython().system('gunzip -f $gzpath')


# In[12]:


STAT,STAT_Descriptions = load(open(data_dir+'/'+filename,'rb'))
print('keys from STAT=',STAT.keys())


# In[13]:


print("   Name  \t                 Description             \t  Size")
print("-"*80)
print('\n'.join(["%10s\t%40s\t%s"%(s[0],s[1],str(s[2])) for s in STAT_Descriptions]))


# ### Read information about US weather stations.

# In[14]:


filename='Weather_Stations.tgz'
parquet='stations.parquet'
command="curl https://mas-dse-open.s3.amazonaws.com/Weather/%s > %s/%s"%(filename,data_dir,filename)
print(command)
get_ipython().system('$command')


# In[15]:


get_ipython().magic('cd $data_dir')
get_ipython().system('tar -xzf $filename')
get_ipython().system('du -s *.parquet')
get_ipython().magic('cd $cur_dir')


# In[16]:


stations_df =sqlContext.read.parquet(data_dir+'/'+parquet)
stations_df.show(5)


# In[17]:


weather_df=weather_df#.drop('name').drop('dist_coast')
weather_df.show(3)


# In[18]:


jdf=weather_df.join(stations_df,on='Station',how='left')
jdf.show(3)
jdf.columns


# In[19]:


sqlContext.registerDataFrameAsTable(weather_df,'jdf')

#find the stations in NY with the most measurements.
sqlContext.sql('select name,count(name) as count from jdf group by name order by count desc').show(5)
#GROUP BY name ORDER BY count').show(5)


# In[20]:


#find how many measurements of each type for a particlar station
stat='ELMIRA'
Query="""
SELECT Measurement,count(Measurement) as count 
FROM jdf 
WHERE Name='%s' 
GROUP BY Measurement
"""%stat
sqlContext.sql(Query).show()


# In[21]:


#find year with all 6 measurements
Query="""
SELECT Year,count(Year) as count 
FROM jdf 
WHERE Name='%s' 
GROUP BY Year
ORDER BY count DESC
"""%stat
sqlContext.sql(Query).show(5)


# In[22]:


# get all measurements for a particular year and a particular station
year=2007
Query="""
SELECT *
FROM jdf 
WHERE Name='%s' 
and Year=%d
"""%(stat,year )
pandas_df=sqlContext.sql(Query).toPandas()
pandas_df=pandas_df.set_index('Measurement')
pandas_df


# ## Plots

# In[23]:


_tmax_20=unpackArray(pandas_df.loc['TMAX_s20','Values'],np.float16)/10.
_tmin_20=unpackArray(pandas_df.loc['TMIN_s20','Values'],np.float16)/10.
_tobs_20=unpackArray(pandas_df.loc['TOBS_s20','Values'],np.float16)/10.
_tmax=unpackArray(pandas_df.loc['TMAX','Values'],np.float16)/10.
_tmin=unpackArray(pandas_df.loc['TMIN','Values'],np.float16)/10.
_tobs=unpackArray(pandas_df.loc['TOBS','Values'],np.float16)/10.
figure(figsize=_figsize)
plot(_tmax,label='TMAX');
plot(_tmin,label='TMIN');
plot(_tobs,label='TOBS');

plot(_tmax_20,label='TMAX_s20');
plot(_tmin_20,label='TMIN_s20');
plot(_tobs_20,label='TOBS_s20');
xlabel('day of year')
ylabel('degrees centigade')
title('Temperatures for %s in %d'%(stat,year))
legend()
grid()


# ### Script for plotting yearly plots

# In[24]:


from lib.YearPlotter import YearPlotter
T=np.stack([_tmin,_tmax,_tobs])

fig, ax = plt.subplots(figsize=_figsize);
YP=YearPlotter()
YP.plot(T.transpose(),fig,ax,title='Temperatures for %s in %d'%(stat,year));
plt.savefig('percipitation.png')
#title('A sample of graphs');


# ### Distribution of missing observations
# The distribution of missing observations is not uniform throughout the year. We visualize it below.

# In[25]:


from lib.MultiPlot import *
YP=YearPlotter()
def plot_valid(m,fig,axis):
    valid_m=STAT[m]['NE']
    YP.plot(valid_m,fig,axis,title='valid-counts '+m,label=m)
    


# In[26]:


plot_pair(['TMIN','TMAX'],plot_valid)


# In[27]:


plot_pair(['TOBS','PRCP'],plot_valid)


# In[28]:


# Note that for "SNOW" there are more missing measurements in the summer
# While for SNWD there are less missing in the summer
# Question: do these anomalies involve the same stations?
plot_pair(['SNOW', 'SNWD'],plot_valid)


# ### Plots of mean and std of observations

# In[29]:


def plot_mean_std(m,fig,axis):
    scale=1.
    temps=['TMIN','TMAX','TOBS','TMIN_s20','TMAX_s20','TOBS_s20']
    percipitation=['PRCP','SNOW','SNWD','PRCP_s20','SNOW_s20','SNWD_s20']
    _labels=['mean+std','mean','mean-std']
    if (m in temps or m=='PRCP'):
        scale=10.
    mean=STAT[m]['Mean']/scale
    std=np.sqrt(STAT[m]['Var'])/scale
    graphs=np.vstack([mean+std,mean,mean-std]).transpose()
    YP.plot(graphs,fig,axis,labels=_labels,title='Mean+-std   '+m)
    if (m in temps):
        axis.set_ylabel('Degrees Celsius')
    if (m in percipitation):
        axis.set_ylabel('millimeter')



# In[30]:


plot_pair(['TMIN','TMIN_s20'],plot_mean_std)


# In[31]:


plot_pair(['PRCP','PRCP_s20'],plot_mean_std)


# In[32]:


plot_pair(['SNOW', 'SNOW_s20'],plot_mean_std)


# In[33]:


plot_pair(['SNWD', 'SNWD_s20'],plot_mean_std)


# ### Plotting percentage of variance explained by Eigen-vectors

# In[34]:


def pltVarExplained(j):
    subplot(1,3,j)
    EV=STAT[m]['eigval']
    k=5
    L=([0,]+list(cumsum(EV[:k])))/sum(EV)
    #print m,L
    plot(L)
    title('Percentage of Variance Explained for '+ m)
    ylabel('Percentage of Variance')
    xlabel('# Eigenvector')
    grid()


# In[35]:


# create a subdirectory in which to place the plots.
get_ipython().system('mkdir r_figures')


# In[36]:


f=plt.figure(figsize=(15,4))
j=1
for m in ['TMIN', 'TOBS', 'TMAX']: #,
    pltVarExplained(j)
    j+=1


# In[37]:


f=plt.figure(figsize=(15,4))
j=1
for m in ['TMIN_s20', 'TOBS_s20', 'TMAX_s20']: 
    pltVarExplained(j)
    j+=1


# In[38]:


f=plt.figure(figsize=(15,4))
j=1
for m in ['SNOW', 'SNWD', 'PRCP']:
    pltVarExplained(j)
    j+=1 


# In[39]:


f=plt.figure(figsize=(15,4))
j=1
for m in ['SNOW_s20', 'SNWD_s20', 'PRCP_s20']:
    pltVarExplained(j)
    j+=1 


# ### plotting top 3 eigenvectors

# In[40]:


def plot_eigen(m,fig,axis):
    EV=STAT[m]['eigvec']
    YP.plot(EV[:,:3],fig,axis,title='Top Eigenvectors '+m)


# In[41]:


plot_pair(['TMIN','TMAX'],plot_eigen)


# In[42]:


plot_pair(['TMIN_s20','TMAX_s20'],plot_eigen)


# In[43]:


plot_pair(['TOBS','PRCP'],plot_eigen)


# In[44]:


plot_pair(['TOBS_s20','PRCP_s20'],plot_eigen)


# In[45]:


plot_pair(['SNOW', 'SNWD'],plot_eigen)


# In[46]:


plot_pair(['SNOW_s20', 'SNWD_s20'],plot_eigen)


# ## Summary
# We saw how to plot:
# * Data from several (Station,Year,Measurement)
# * The mean+-std for a particular (Station,Measurement)
# * The percentage of cariance explained by top eigen-vectors.
# * The top eigen-vectors

# In[ ]:




