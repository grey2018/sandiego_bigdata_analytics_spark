
# coding: utf-8

# # Notebooks
# 
# 0) **README.ipynb** this notebook  
# 1) **FunctionsAsVectors.ipynb** : Demonstration of the fourier orthonormal basis for functions.  
# 2) **PCA_computation per state.ipynb** : Computing the PCA and other statistics for data from a single state.  
# 3) **Weather Analysis - Initial Visualisation.ipynb** : Visualizing the statistics.  
# 3a) **Weather Analysis - Visualisation after smoothing.ipynb** : Same as (3) but using smoothed signals.  
# 4) **Weather Analysis - reconstruction SNWD.ipynb** : Reconstruction of snow depth using top eigenvector: generates reconstruction file that is used by (5)  
# 4a) **Weather Analysis - reconstruction PRCP-Smoothed.ipynb** : Same as (4) but for smoothed percipitation (*PRCP_s20*)  
# 5) **Maps using iPyLeaflet.ipynb** : Plotting information about stations on an interactive map.  
# 6) **Is SNWD variation spatial or temporal?.ipynb** : Using the "variance explained" criteria to decide whether space or time has a bigger effect on a coefficient.  
# 7) **Smoothing.ipynb** : Code for smoothing the measurements across days.  

# # Files / Tables
# 
# ## Readme file describing the original data
# [download readme file from s3](https://mas-dse-open.s3.amazonaws.com/Weather/Info/ghcnd-readme.txt)

# ## Source Data
# 
# The source data is stored in parquet files. `ALL.parquet` contains all of the data, `NY.parquet` includes the data for all stations in New-York State. In the following sections we use NY state, but instead you can use any of:
# ```
# AB,AL,AR,AZ,BC,CA,CO,CT,DC,DE,FL,GA,IA,ID,IL,IN,KS,KY,LA,MA,MB,MD,ME,MI,MN,MO,MS,MT,NaN,NB,NC,ND,NE,NH,NJ,NM,NV,NY,OH,OK,ON,OR,PA,QC,RI,SC,SD,SK,TN,TX,UT,VA,VT,WA,WI,WV,WY
# ```
# Most of these states are in the continential USA - those contain the complete data. Others are states in Canada and Mexico and have only partial data, Finally, `NaN` contains measurements from stations that are not in any state (and therefor also not in the US).
# 
# 
# ### Schema
# 
# * **Station:** Station ID
# * **Measurement:** Type of measurement. Is one of:
#   * *TMIN* Minimal Temperature during the day.
#   * *TMAX* Maximal Temperature During the day.
#   * *TOBS* "Observed" temperature. Exact meaning not clear, but is less noisy than *TMIN* and *TMAX*
#   * *PRCP* Total percipitation during the day.
#   * *SNOW* Amount of snow that fell during the day.
#   * *SNWD* Snow Depth
#   * *TMIN_s20*, *TMAX_s20*,... A smoothed version of the six raw measurements.
# * **Year**
# * **Values:** a byte array of length 2*365 representing 365 floats (np.float16)
# * **State**
# 
# |    Station|Measurement|Year|              Values|State|
# |-----------|-----------|----|--------------------|-----|
# |USC00303452|       PRCP|1903|[00 7E 00 7E 00 7...|   NY|
# |USC00303452|       PRCP|1904|[00 00 28 5B 00 0...|   NY|
# |USC00303452|       PRCP|1905|[00 00 60 56 60 5...|   NY|
# |USC00303452|       PRCP|1906|[00 00 00 00 00 0...|   NY|
# |USC00303452|       PRCP|1907|[00 00 00 00 60 5...|   NY|
# 
# ### Reading measurement data into a dataframe
# 
# #### When using your own computer
# The files are stored on AWS as compressed tar files. The bucket `mas-dse-open` can be accessed through an HTTP connection. 
# 
# 1) Download all of the measurements for a particular state (here NY) or ALL for the file that contains all of the states
# 
# 
# > curl https://mas-dse-open.s3.amazonaws.com/Weather/by_state_2/NY.tgz 
#   \> `data_dir`/NY.tgz
# 
# > Where `data_dir` is the local data directory, here `big-data-analytics-using-spark/notebooks/Data/Weather`
# 
# 2) Untar the tar file 
# 
# > tar -xzf `data_dir`/NY.tgz
# 
# > Creates the parquet directory `data_dir`/NY.parquet
# 
# 3) Read the parquet file into a dataframe:
# 
# > df=sqlContext.read.parquet(`data_dir`/NY.parquet)

# ## Station information
# 
# Information about each station in the continental united states:
# 
# ### Schema
# * **Station:** Station ID.
# * **Dist\_coast:** Distance from the coast (shoreline) (units? 1.4 of this  per mile?)
# * **Latitude**
# * **Longitude**
# * **Elevation** in meters, missing = -999.9
# * **Name:** the name of the station.
# 
# |    Station|Dist_coast|Latitude|Longitude|Elevation|State|            Name|
# |-----------|----------|--------|---------|---------|-----|----------------|
# |USC00044534|   107.655| 36.0042|  -119.96|     73.2|   CA|  KETTLEMAN CITY|
# |USC00356784|   0.61097| 42.7519|-124.5011|     12.8|   OR|PORT ORFORD NO 2|
# |USC00243581|   1316.54| 47.1064|-104.7183|    632.8|   MT|        GLENDIVE|
# |USC00205601|   685.501|   41.75| -84.2167|    247.2|   MI|         MORENCI|
# |USC00045853|   34.2294| 37.1364|-121.6025|    114.3|   CA|         MORGAN HILL|
# 
# ### Downloading stations dataframe from S3:
# Download from S3:
# > curl https://mas-dse-open.s3.amazonaws.com/Weather/Weather_Stations.tgz > `data_dir`/Weather_Stations.tgz  
# 
# Untar. Creates a parquet directory:  
# > tar -xzf `data_dir`/Weather_Stations.tgz  
# 
# Read parquest file into dataframe:
# > stations_df=sqlContext.read.parquet(`data_dir`/Weather_Stations.parquet)
# 
# Print first 4 rows in stations_df dataframe
# > stations_df.show(4)                

# 

# ## Statistics file
# 
# A file with the name `data_dir/STAT_NY.pickle` is pickle file containing the statistics computed for the state of NY. 
# 
# To download file from S3, use
# > curl https://mas-dse-open.s3.amazonaws.com/Weather/by_state_2/STAT_NY.pickle.gz \> data_dir/STAT_NY.pickle.gz  
# 
# To unzip the file use  
# > gunzip data_dir/STAT_NY.pickle.gz  
# 
# The pickle file contains a pair: `(STAT,STAT_Descriptions)`. 
# * `STAT` contains the calculated statistics as a dictionary. 
# * `STAT_Descriptions` contains a human-readable description of each element of the dictionary `STAT`

# ## The content of `STAT_DESCRIPTION
# ```
#    Name  	                 Description             	  Size
# --------------------------------------------------------------------------------
# SortedVals	                        Sample of values	vector whose length varies between measurements
#      UnDef	      sample of number of undefs per row	vector whose length varies between measurements
#       mean	                              mean value	()
#        std	                                     std	()
#     low100	                               bottom 1%	()
#    high100	                                  top 1%	()
#    low1000	                             bottom 0.1%	()
#   high1000	                                top 0.1%	()
#          E	                   Sum of values per day	(365,)
#         NE	                 count of values per day	(365,)
#       Mean	                                    E/NE	(365,)
#          O	                   Sum of outer products	(365, 365)
#         NO	               counts for outer products	(365, 365)
#        Cov	                                    O/NO	(365, 365)
#        Var	  The variance per day = diagonal of Cov	(365,)
#     eigval	                        PCA eigen-values	(365,)
#     eigvec	                       PCA eigen-vectors	(365, 365)
# ```

# ## Reconstruction file
# 
# Stored in files named `recon_<state>_<measurement>.parquet`
# #### Fields:
# 1. **Station** :  Station ID
# 21. **State** :  The state in which the station resides
# 22. **Name** :  The name of the station
# 17. **Dist_coast** :  Distance from Coast (units unclear)
# 18. **Latitude** :  of station
# 19. **Longitude** :  of station
# 20. **Elevation** :  Elevation of station in Meters
# 2. **Measurement** :  Type of measurement (TMAX, PRCP,...)
# 3. **Values** :  A byte array with all of the value (365X2 bytes)
# 4. **Year** :  The Year
# 5. **coeff_1** :  The coefficient of the 1st eigenvector
# 6. **coeff_2** :  The coefficient of the 2nd eigenvector
# 7. **coeff_3** :  The coefficient of the 3rd eigenvector
# 8. **coeff_4** :  The coefficient of the 4th eigenvector
# 9. **coeff_5** :  The coefficient of the 5th eigenvector
# 16. **total_var** : The total variance (square distance from the mean. 
# 15. **res_mean** :  The residual variance after subtracting the mean.
# 10. **res_1** :  The residual variance after subtracting the mean and eig1 
# 11. **res_2** :  The residual variance after subtracting the mean and eig1-2
# 12. **res_3** :  The residual variance after subtracting the mean and eig1-3 
# 13. **res_4** :  The residual variance after subtracting the mean and eig1-4 
# 14. **res_5** :  The residual variance after subtracting the mean and eig1-5 

# In[ ]:




