#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 16:58:54 2024

@author: Irene Palazzoli

Apply model over a climatic region
and merge predictions of all pixels

Set of predictors:
    SIF:    meteorological + vegetation dynamics data
    noSIF:  meteorological predictors only

Model hyperparameters obtained from Optuna tuning tool
            
"""

# System libraries
import os, sys

# Libraries for plots and array manipulation
import numpy as np

# LSTM libraries
import tensorflow 
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

# Other libraries (da sistemare...)
from sklearn.preprocessing import RobustScaler
import joblib

# import tensorflow as tf
import optuna

### Define global variables

# Define dictionary for column names of data array
colname = { "pixel_ID":0, "lat":1, "lon":2, "year":3, "month":4, 
           "pcp":5, "tmp":6, "snow":7, "rad":8, "rh":9, "sif":10,
           "clima":11, "tws":12 }

class hyperparameters():

    def __init__(self):
        # Initialize return values of the function
        self.window = 0              ### Number of input time steps
        
    def load_optuna(self, optuna_dir, model_name, clima_name):
        
        # Load hyperparameters determined by Optuna tuning tool
        study = optuna.load_study(study_name = clima_name + "-study_OPTUNA_" + model_name, 
                                storage = "sqlite:///" + optuna_dir + clima_name + "_OPTUNA_" + model_name + ".db")
        
        # Best trial out of ~100
        trial = study.best_trial
        
        print("\nOptuna hyperparameters:")
        
        for key, value in trial.params.items():
            if key == "window":             
                self.window = value
            
            print("    {}: {}".format(key, value))
            

def usage():
    print("Wrong number of arguments!\n")
    print("Usage: python", sys.argv[0], "model features clima subclima")
    print("- model can be LSTM or BiLSTM")
    print("- features can be SIF or noSIF")
    print("- clima is any main climatic region of Köppen-Geiger classification (see paper)")
    print("- subclima is any subregion of the main climatic region (see paper)")
    
    sys.exit(1)
    

def main():
    
    ### Get user's arguments
    
    # Check number of arguments
    n = len(sys.argv)
    if n != 5:
        usage()
    
    model = sys.argv[1]                 #### LSTM or BiLSTM
    features = sys.argv[2]              #### Predictors with/without SIF data (SIF or noSIF)
    clima = sys.argv[3]                 #### Main climatic region
    subclima = sys.argv[4]              #### Climatic subregion
    
    print("\nApply", model, "model with new LR over some pixels of ", clima, subclima, "climatic region")
    
    if features == "SIF":
        print("\nSet of predictors: meteorological forcings and SIF data")
        model_name = model               # Define model name
        
    else:
        print("\nSet of predictors: meteorological forcings only")  
        model_name = model + features    # Define model name
        
    clima_name = clima + "_" + subclima
     
    # Define folders path
    path = "."
    data_dir = path + "/input_data/"                        # folder where input data are stored 
    optuna_dir = path + "/optuna/" + model_name + "/"        # folder where optuna databases are stored   

    
    ### Folders where model outputs are saved     
    out_dir = path + "/model_fits/" + model_name + "/newLR/" + clima_name + "/"
    model_dir = out_dir + "model/"          ### folder of best model fit out of 5 attempts
    scaler_dir = out_dir = path + "/model_fits/" + model_name + "/" + clima_name + "/scalers/"       ### folder of scalers of predictors and target variable
    
    # Folder of list of pixels w/ low PCC
    listID_dir = path + "/model_fits/" + model_name + "/lowPCC_train/" 
    
    twsamodel_dir = path + "/model_predictions/" + model_name + "/newLR_climatic_regions/" 
    
    isExist = os.path.exists(out_dir)
    if not isExist:
       os.makedirs(twsamodel_dir)
   
    
    ### Load Optuna hyperparameters
    hyppar = hyperparameters()
    hyppar.load_optuna(optuna_dir, model_name, clima_name)
    window = hyppar.window
    
    ### Load list of pixels
    id_refitPIX = np.load(listID_dir + "lowPCC_pixels_" + model_name + "_" + clima_name + ".npy")
    id_refitPIX = id_refitPIX[:, 0] # pixeld IDs on first column
    
    ### Load data array with target variable and predictors data from 1982 to 2021
    data_arr, id_pixels, input_cols, target_col = load_data(data_dir, clima_name, features, id_refitPIX)
    
    n_pixels = id_pixels.shape[0]
    n_ins = int(data_arr.shape[0]/n_pixels)
    
    ### Define dictionary for column names of predictions array
    colname_out = { "pixel_ID":0, "lat":1, "lon":2, "year":3, "month":4, 
                   "obs":5, "pred":6, "dataset":7 }
    
    n_obs_perpix = n_ins - int(window)
    n_row = n_obs_perpix * n_pixels
    
    all_pixel = np.zeros((n_row, len(colname)), dtype = np.float32) 
    all_pixel[:,colname_out["pixel_ID"]] = np.repeat(id_pixels, n_obs_perpix)
    all_pixel[:,colname_out["lat"]] = np.repeat(data_arr[::n_ins, colname["lat"]], n_obs_perpix)
    all_pixel[:,colname_out["lon"]] = np.repeat(data_arr[::n_ins, colname["lon"]], n_obs_perpix)
    
    ### Loop over grid cells range
    for p in range(n_pixels):
        
        pidx = int(id_refitPIX[p])
        
        # Load model
        model = load_model(model_dir + model_name + "_PIXid_" + str(pidx) + "_" + clima_name + ".h5")
        
        # Load scalers
        s_in = joblib.load(scaler_dir + model_name + "_INscaler_PIXid_" + str(pidx) + "_" +  clima_name + ".gz")
        
        s_out = joblib.load(scaler_dir + model_name + "_OUTscaler_PIXid_" + str(pidx) + "_" +  clima_name + ".gz")
        
        
        # Select data for pixel p
        mask_pixid = (data_arr[:, colname["pixel_ID"]] == pidx)
        data_pix = data_arr[mask_pixid, :]
         
        years = data_pix[window:,colname["year"]]
        months = data_pix[window:,colname["month"]]
        
        
        print("\nPrepare datasets...")
        
        dataX = data_pix.copy()
        dataY = data_pix.copy()
        
        id_X = dataX[:,colname["pixel_ID"]].astype(np.int32)
        dataX = dataX[:, input_cols]
        dataY = dataY[:, target_col]

        ### Index where TWS is not NA
        dataY_ID = ~np.isnan(dataY)
        idx = np.arange(dataY_ID.shape[0], dtype=np.int32)
        idx = idx[dataY_ID]

        ### TWSA observations from GRACE/GRACE-FO (Y)
        Y = dataY[dataY_ID]

        Y_perPix = Y.shape[0]

        ### Prepare input data (X) for all observations Y
        X = -np.ones((Y_perPix * window, len(input_cols)), dtype = np.float32)

        offset = 0

        for out in range(0, Y_perPix):
            X[offset:offset+window, :] = dataX[idx[out]-window:idx[out], :]
            offset += window
        
        
        ### Define shuffle array if p is the first pixel (it's the same for all pixels in "clima_name")
        if p == 0:
            chunkSH_arr, obs_train, obs_val = chunk_shuffle(window, Y.shape[0])

        
        ################# Apply model over the whole input time series #################
        
        Y_perINS = n_ins - window + 1
        Y_obs = dataY.copy()
        Y_obs = Y_obs[window:]
        
        dataset = -np.ones(Y_obs.shape, dtype=np.int32)
        Yobs_notNA = ~np.isnan(Y_obs)
        
        Y_off = np.repeat(Y_obs[Yobs_notNA][0], Y_perINS)

        dataset_val = np.arange(chunkSH_arr.shape[0], dtype=np.int32)

        dataset_val[chunkSH_arr[:obs_train]] = 0
        dataset_val[chunkSH_arr[obs_train:obs_train+obs_val]] = 1
        dataset_val[chunkSH_arr[obs_train+obs_val:]] = 2
        
        dataset[Yobs_notNA] = dataset_val
        
        dataINS = dataX.copy()
        dataINS[:, :] = s_in.transform(dataINS[:, :]) 

        Xall = -np.ones((Y_perINS * window, len(input_cols)), dtype = np.float32)
        
        maskXall_window = np.mod(np.arange(Xall.shape[0]), Y_perINS * window) < window
        maskdataXall_window = np.mod(np.arange(id_X.shape[0]), n_ins) < window
        
        for out in range(0, Y_perINS):
            if out==0:
                Xall[maskXall_window, :] = dataINS[maskdataXall_window, :]
            else:
                maskdataXall_window = np.roll(maskdataXall_window, 1)
                maskXall_window = np.roll(maskXall_window, window)
                Xall[maskXall_window, :] = dataINS[maskdataXall_window, :]
               
        Xall = Xall.reshape((Xall.shape[0]//window, window, Xall.shape[1]))
        
        Y_pred = model.predict(Xall[:, :, :]) 
        Y_out = s_out.inverse_transform(np.array([Y_pred[:,0]]).T)
        Yp = Y_out[:-1,0] + Y_off  ### Add back TWSA offset and stop on Dec 2021
        
        ### Update output array with data of pixel p
        mask_pix = all_pixel[:,colname_out["pixel_ID"]] == pidx
        
        all_pixel[mask_pix,colname_out["pred"]] = Yp
        all_pixel[mask_pix,colname_out["obs"]] = Y_obs
                
        all_pixel[mask_pix,colname_out["year"]] = years
        all_pixel[mask_pix,colname_out["month"]] = months
        all_pixel[mask_pix,colname_out["dataset"]] = dataset
    
    np.save(twsamodel_dir + model_name + "_predictions_" + clima_name, all_pixel)   
        

def load_data(data_dir, clima_name, features, id_refitPIX):
    
    data_arr = np.load(data_dir + "INPUTdata_" + clima_name + ".npy")
    
    ### Identify target variable and predictors data
    target_col = colname["tws"]
    
    # List of columns to use as predictors
    if features == "SIF":
        input_cols = [colname[key] for key in colname.keys()][5:11]
        
        ### Remove grid cells where SIF data are not available
        pixel_mask = np.isnan(data_arr[:,input_cols]).any(axis=1)
        id_toremove, count = np.unique(data_arr[pixel_mask, colname["pixel_ID"]].astype(np.int32), return_counts=True)
        pixel_mask = np.in1d(data_arr[:, colname["pixel_ID"]].astype(np.int32), id_toremove, invert=True)

        data_arr = data_arr[pixel_mask, :]
        
    else:
        input_cols = [colname[key] for key in colname.keys()][5:10]  ## no SIF data!
        
    ## Select pixels with low PCC
    mask_pix = np.in1d(data_arr[:,colname["pixel_ID"]].astype(np.int32), id_refitPIX)
    data_arr = data_arr[mask_pix]
    n_pixels = id_refitPIX.shape[0]

    print("Total number of grid cells in ", clima_name, "is", str(n_pixels))
    
    return data_arr, input_cols, target_col


def chunk_shuffle(window, n_obs):       ### Define array for data shuffling
    
    # Initialize width of data chunks (must be larger than window)
    chunk_wd = window + 1
    
    n_chunk = n_obs//chunk_wd 
    
    chunk_wdNEW = n_obs//n_chunk
    
    if n_obs%n_chunk:
        chunk_wdNEW += 1
        
    print("\nDivide time series in chunks of " + str(chunk_wdNEW) + " months")
            
    chunk_arr = -99*np.ones(n_chunk*chunk_wdNEW, dtype=np.int32)
    
    index = np.array(range(n_obs), dtype=np.int32)
    
    n_chunk_missing = n_chunk*chunk_wdNEW-n_obs
    
    mask_chunk = np.ones(chunk_arr.shape, dtype = np.bool)
    
    mask_chunk = mask_chunk.reshape(n_chunk, -1)
    
    mask_chunk[n_chunk-n_chunk_missing:, -1] = False
    
    chunk_arr[mask_chunk.flat[:]] = index[:]
    
    chunk_arr = chunk_arr.reshape(n_chunk, -1)
    
    shuffle_arr = np.zeros(n_chunk, dtype=np.int32)

    tmp = np.array(range(n_chunk), dtype=np.int32)
    np.random.shuffle(tmp)
    
    shuffle_arr[:] = tmp[:]
    
    chunkSH_arr = chunk_arr[shuffle_arr,:].reshape(-1)
    
    ### Number of chunks for training and validation dataset
    chunk_train = int(0.6 * n_chunk)
    chunk_val = int((n_chunk - chunk_train)/2)
    
    obs_cut = chunk_train*chunk_wdNEW
    val_cut = obs_cut + chunk_val*chunk_wdNEW
    obs_train = (chunkSH_arr[:obs_cut] > -99).sum()
    obs_val = (chunkSH_arr[obs_cut:val_cut] > -99).sum()
        
    chunkSH_arr = chunkSH_arr[chunkSH_arr > -99]
   
    return chunkSH_arr, obs_train, obs_val


if __name__ == "__main__":
    main()


    
############################ END ############################

            
    
    
    
    
    