#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 11:17:27 2024

@author: Irene Palazzoli

Merge predictions for all climatic regions produced by LSTM or BiLSTM model
and save array in netCDF format

Set of predictors:
    SIF:    meteorological + vegetation dynamics data
    noSIF:  meteorological predictors only

Model hyperparameters obtained from Optuna tuning tool:
    use a combination of the original and new values of LR 
    to improvePCC in training + validation dataset of some grid cells
            
"""

# System libraries
import os, sys, gc

import netCDF4
import numpy as np
import datetime as dt
from netCDF4 import date2num,num2date


# Define dictionary for column names of predictions array
colname = { "pixel_ID":0, "lat":1, "lon":2, "year":3, "month":4, 
           "obs":5, "pred":6, "subset":7 }

clima_regs = ["tropical_rainforest", "tropical_monsoon", "tropical_savannah", "arid_deserthot",
              "arid_desertcold", "arid_steppehot", "arid_steppecold", "temperate_dryhotsummer",
              "temperate_drywarmsummer", "temperate_drycoldsummer", "temperate_drywinhotsum", 
              "temperate_drywinwarmsum", "temperate_drywincoldsum", "temperate_nodryhotsum",
              "temperate_nodrywarmsum", "temperate_nodrycoldsum", "continental_dryhotsummer", 
              "continental_drywarmsummer", "continental_drycoldsummer", 
              "continental_drysumverycoldwin", "continental_drywinhotsum", 
              "continental_drywinwarmsum", "continental_drywincoldsum", 
              "continental_drywinverycoldwin", "continental_nodryhotsum", 
              "continental_nodrywarmsum", "continental_nodrycoldsum", 
              "continental_nodryverycoldwin", "polar_tundra", "polar_frost"]

clima_class = list(range(1,31))

select_regs = [10, 13]  ### remove these climatic regions (no data!) 
for element in select_regs:
    clima_class.remove(element)


def usage():
    print("Wrong number of arguments!\n")
    print("Usage: python", sys.argv[0], "model features")
    print("- model can be LSTM or BiLSTM")
    print("- features can be SIF or noSIF")
    
    sys.exit(1)
    

def main():
    
    ### Get user"s arguments
    
    # Check number of arguments
    n = len(sys.argv)
    if n != 3:
        usage()
    
    model = sys.argv[1]                 #### LSTM or BiLSTM
    features = sys.argv[2]              #### Predictors with/without SIF data (SIF or noSIF)
    
    print("\nLoad all predictions of", model, "model")
    
    if features == "SIF":
        print("\nSet of predictors: meteorological forcings and SIF data")
        model_name = model               # Define model name
        
    else:
        print("\nSet of predictors: meteorological forcings only")  
        model_name = model + features    # Define model name
          
    # Define folders path
    path = "."
    twsa_dir = path + "/model_predictions/" + model_name + "/climatic_regions/" 
    newtwsa_dir = path + "/model_predictions/" + model_name + "/newLR_climatic_regions/" 
    out_dir = twsa_dir + "../TWSAI-rec/"
    
    isExist = os.path.exists(out_dir)
    if not isExist:
       os.makedirs(out_dir)
    
    # Folder of list of pixels w/ low PCC
    listID_dir = path + "/model_fits/" + model_name + "/lowPCC_train/"   
    
    
    for clima_num in clima_class:   
        
        clima_name = clima_regs[clima_num - 1]
        print("Load data for", clima_name + " region")
        
        id_refitPIX = np.load(listID_dir + "newLR_pixels_" + model_name + "_" + clima_name + ".npy")
        
        twsa = np.load(twsa_dir + model_name + "_predictions_" + clima_name + ".npy")
        
        newtwsa = np.load(newtwsa_dir + model_name + "_predictions_" + clima_name + ".npy")
        
        """
        Select only pixels where new LR increases PCC
        and replace those values in the predictions array
        """
        mask_new = np.in1d(newtwsa[:, colname["pixel_ID"]].astype(np.int32), id_refitPIX)
        mask_old = np.in1d(twsa[:, colname["pixel_ID"]].astype(np.int32), id_refitPIX)
        twsa[mask_old,:] = newtwsa[mask_new, :]
    
        np.save(out_dir + model_name + "_TWSAI-rec_" + clima_name, twsa)
    