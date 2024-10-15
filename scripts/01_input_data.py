#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 16:47:25 2024

@author: Irene Palazzoli

Create array with predictors and target variable from 1982 to 2021

Predictors: monthly precipitation, 
            monthly snow depth water equivalent, 
            monthly surface net solar radiation, 
            monthly surface air temperature, 
            monthly surface air relative humidity, 
            monthly solar-induced fluorescence
    
Target variable: monthly GRACE/GRACE-FO observations (from 2002 to 2021)

Generate a npy file for each Köppen-Geiger climatic region 

It requires tiff files of GRACE/GRACE-FO monthly observations,
raster stacks of meteorological data and sif data,
and tiff file of Köppen-Geiger classification system

All data must have 0.5 degree spatial resolution
"""

# System libraries
import datetime
import glob

# Libraries for array manipulation
import numpy as np

# Raster libraries
import rasterio as rio


# Folders path
path = "insert path to folder containing data"
data_dir = "insert path to folder containing raster files of input data"
out_dir = "insert path to folder where input arrays will be stored"

# Create time series from Jan 1982 to Dec 2021
yr_i = 1982
yr_f = 2021   

n_months = (yr_f-yr_i+1)*12

time_arr = -np.ones((n_months, 2), dtype=np.int32)
yr_list = np.repeat(np.array([i for i in range(yr_i, yr_f+1)], dtype=np.int32), 12)
mon_list = np.tile(np.array([i for i in range(1, 13)], dtype=np.int32), yr_f+1-yr_i)

time_arr[:, 0] = yr_list[:]
time_arr[:, 1] = mon_list[:]

### TARGET VARIABLE ###

### Load GRACE/GRACE-FO TWSA measurements ###

# Find all monthly TWSA tif files
list_tws = glob.glob(data_dir + "JPLmascons/rasters/mascon_lwe_thickness*.tif")
# print(list_tws)

# Get dimension of TWS rasters
with rio.open(list_tws[0]) as tws:
    tws_arr = tws.read(1)
    na_tws = tws.nodata
    
    # Get coordinates of raster
    print('TWSA has shape', tws.shape)
    height = tws.shape[0]
    width = tws.shape[1]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs, ys = rio.transform.xy(tws.transform, rows, cols)
    lons = np.array(xs)
    lats = np.array(ys)
    ids = np.arange(cols.size).reshape(lats.shape)
    print('lons shape', lons.shape)
    print('lats shape', lats.shape)

### Identify grid cells with no data in TWS rasters
tws_naID = np.isnan(tws_arr)
tws_naID.sum()

## Load data    
tws_time = np.zeros((n_months, tws_arr.shape[0], tws_arr.shape[1]))
tws_time[:] = np.nan

time_index = -np.ones(n_months, dtype=np.int32)

for f in list_tws:
        
    moy = f.split("/")[-1].split("_")[-2:]
    moy[1] = moy[1].split(".")[0]
    yr = int(moy[0][0:4])
    doy_init = int(moy[0][4:])
    doy_end = int(moy[1][4:])
    
    if yr > yr_f or yr < yr_i:
        continue
        
    date_start = datetime.datetime(yr, 1, 1) + datetime.timedelta(doy_init - 1)
    date_end = datetime.datetime(yr, 1, 1) + datetime.timedelta(doy_end - 1)
    first_day = date_start.day
    first_mon = date_start.month
    last_day = date_end.day
    last_mon = date_end.month
        
    if first_day < 15:
        mon = first_mon
    else:
        mon = last_mon

    with rio.open(f) as tws:
        tws_arr = tws.read(1)
        
    t_index = mon + 12 * (yr - yr_i) - 1

    tws_time[t_index,:,:] = tws_arr[:,:] 
    
    time_index[t_index] = 1
   

### PREDICTORS ###    
   
### Load meteorological data ###

### Precipitation
with rio.open(data_dir + "PCPdata.tif") as pcp:

    pcp_arr = pcp.read()
    na_pcp = pcp.nodata
    pcp_arr[np.isclose(pcp_arr, na_pcp)] = np.nan

### Air temperature
with rio.open(data_dir + "TMPdata.tif") as tmp:

    tmp_arr = tmp.read()
    na_tmp = tmp.nodata
    tmp_arr[np.isclose(tmp_arr, na_tmp)] = np.nan

### Water equivalent snow content
with rio.open(data_dir + "SNOWdata.tif") as snow:

    snow_arr = snow.read()
    na_snow = snow.nodata
    snow_arr[np.isclose(snow_arr, na_snow)] = np.nan

### Solar radiation
with rio.open(data_dir + "SRAD.tif") as rad:

    rad_arr = rad.read()
    na_rad = rad.nodata
    rad_arr[np.isclose(rad_arr, na_rad)] = np.nan

### Relative humidity
with rio.open(data_dir + "RH.tif") as rh:

    rh_arr = rh.read()
    na_rh = rh.nodata
    rh_arr[np.isclose(rh_arr, na_rh)] = np.nan

### Load SIF data ###
with rio.open(data_dir + "SIFdata.tif") as sif:
    sif_arr = sif.read()
    na_sif = sif.nodata
    sif_arr[np.isclose(sif_arr, na_sif)] = np.nan

    
### KOPPEN CLIMATE DATA (30 classes) ###    
with rio.open(data_dir + "CLIMAdata.tif") as clima:

    clima_arr = clima.read()
    clima_arr = clima_arr.astype(np.float32)
    na_clima = clima.nodata
    clima_arr[np.isclose(clima_arr, na_clima)] = np.nan
    
    
### Set no data value in TWS time series 
tws_time[np.isclose(tws_time, na_tws)] = np.nan


# How many NAs TWSA and predictors?
print("Show number of NaNs in TWS time series and input data:")
print("TWS #NaN:", np.isnan(tws_time).sum())  
print("PCP #NaN:", np.isnan(pcp_arr).sum())
print("TMP #NaN:", np.isnan(tmp_arr).sum())
print("SRAD #NaN:", np.isnan(rad_arr).sum())   
print("SNOW #NaN:", np.isnan(snow_arr).sum()) 
print("RH #NaN:", np.isnan(rh_arr).sum())
print("SIF #NaN:", np.isnan(sif_arr).sum()) 

### Prepare data array 
# Define dictionary for column names of data array
colname = { "pixel_ID":0, "lat":1, "lon":2, "year":3, "month":4, 
           "pcp":5, "tmp":6, "snow":7, "rad":8, "rh":9, "sif":10,
           "clima":11, "tws":12 }

# Total number of grid cells (before filtering in space)
n_pixels = tws_time.shape[1]*tws_time.shape[2]  

n_obs = tws_time.shape[0]
data_array = np.zeros((tws_time.size, len(colname)), dtype = np.float32) 
data_array[:,colname["pixel_ID"]] = np.repeat(ids.flat[:], n_obs)
data_array[:,colname["lat"]] = np.repeat(lats.flat[:], n_obs)
data_array[:,colname["lon"]] = np.repeat(lons.flat[:], n_obs)
data_array[:,colname["year"]] = np.tile(time_arr[:,0], n_pixels)
data_array[:,colname["month"]] = np.tile(time_arr[:,1], n_pixels)

# Move time index to third position and flatten data. Repeat for all variables
data_array[:,colname["pcp"]] = np.transpose(pcp_arr, axes=(1,2,0)).flat[:]
data_array[:,colname["tmp"]] = np.transpose(tmp_arr, axes=(1,2,0)).flat[:]
data_array[:,colname["snow"]] = np.transpose(snow_arr, axes=(1,2,0)).flat[:]
data_array[:,colname["rad"]] = np.transpose(rad_arr, axes=(1,2,0)).flat[:]
data_array[:,colname["rh"]] = np.transpose(rh_arr, axes=(1,2,0)).flat[:]
data_array[:,colname["sif"]] = np.transpose(sif_arr, axes=(1,2,0)).flat[:]

data_array[:,colname["clima"]] = np.repeat(np.transpose(clima_arr, axes=(1,2,0)).flat[:], n_obs)

data_array[:,colname["tws"]] = np.transpose(tws_time, axes=(1,2,0)).flat[:]


############## FILTER DATA IN SPACE ##############
### Remove grid cells w/ Nan along predictors time series

col_preds = [colname[key] for key in colname.keys()][5:11]

pixel_mask = np.isnan(data_array[:,col_preds]).any(axis=1)

id_toremove, count = np.unique(data_array[pixel_mask, colname["pixel_ID"]].astype(np.int32), return_counts=True)

pixel_mask = np.in1d(data_array[:, colname["pixel_ID"]].astype(np.int32), id_toremove, invert=True)

data_clean = data_array[pixel_mask, :]

data_clean.shape
data_array.shape

n_pixels = int(data_clean.shape[0]/n_obs)


######################### Subset data array based on climatic regions  #########################

# Climatic regions names
clima_names = ["tropical_rainforest", "tropical_monsoon", "tropical_savannah", 
               "arid_deserthot", "arid_desertcold", "arid_steppehot", "arid_steppecold", 
               "temperate_dryhotsummer", "temperate_drywarmsummer", "temperate_drycoldsummer", 
               "temperate_drywinhotsum", "temperate_drywinwarmsum", "temperate_drywincoldsum", 
               "temperate_nodryhotsum", "temperate_nodrywarmsum", "temperate_nodrycoldsum", 
               "continental_dryhotsummer", "continental_drywarmsummer", "continental_drycoldsummer", 
               "continental_drysumverycoldwin", "continental_drywinhotsum", "continental_drywinwarmsum", 
               "continental_drywincoldsum", "continental_drywinverycoldwin", "continental_nodryhotsum", 
               "continental_nodrywarmsum", "continental_nodrycoldsum", "continental_nodryverycoldwin", 
               "polar_tundra", "polar_frost"]

# Climatic regions classes
clima_class = list(range(1,31))

for clima_ID in clima_class:
    
    mask_clima = data_clean[:,colname["clima"]] == clima_ID
    data_clima = data_clean[mask_clima, :]
    np.save(out_dir + "INPUTdata_" + clima_names[clima_ID-1], data_clima)
    

