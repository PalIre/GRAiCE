#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:35:51 2022

@author: Irene Palazzoli

Fit LSTM or BiLSTM model to produce TWSA reconstruction from 1984 to 2021

Set of predictors:
    SIF:    meteorological + vegetation dynamics data
    noSIF:  meteorological predictors only

Model hyperparameters obtained from Optuna tuning tool

"""

# System libraries
import os, sys, gc

# Libraries for plots and array manipulation
import numpy as np
import matplotlib.pyplot as plt

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
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState

### Define global variables

# Define dictionary for column names of data array
colname = { "pixel_ID":0, "lat":1, "lon":2, "year":3, "month":4, 
           "pcp":5, "tmp":6, "snow":7, "rad":8, "rh":9, "sif":10,
           "clima":11, "tws":12 }

class global_par():
    
    def __init__(self, tot_fit = 5, ptc = 30, epochs = 1000, bsize = 32):
        
        self.tot_fit = tot_fit      ## number of fit to run for each grid cell
        self.ptc = ptc              ## patience
        self.epochs = epochs        ## number of epochs
        self.bsize = bsize          ## batch size
   
### Define model hyperparameters

class hyperparameters():

    def __init__(self):
        # Initialize return values of the function
        self.l_rate = 0              ### Learning rate
        self.window = 0              ### Number of input time steps
        self.units_lay1 = 0          ### Number of units 
        self.units_in = 0            ### Number of hydden layers
        self.units_lay2 = 0          ### Number of units in second hidden layer
        
    def load_optuna(self, optuna_dir, model_name, clima_name):
        
        # Load hyperparameters determined by Optuna tuning tool
        study = optuna.load_study(study_name = clima_name + "-study_OPTUNA_" + model_name, 
                                storage = "sqlite:///" + optuna_dir + clima_name + "_OPTUNA_" + model_name + ".db")
        
        # Best trial out of ~100
        trial = study.best_trial
        
        print("\nOptuna hyperparameters:")
        
        for key, value in trial.params.items():
            if key == "learning_rate":      
                self.l_rate = value
            if key == "window":             
                self.window = value
            if key == "n_units_l0":         
                self.units_lay1 = value
            if key == "units":              
                self.units_in = value
            if key == "n_units_l1":         
                self.units_lay2 = value
            
            print("    {}: {}".format(key, value))
            

def usage():
    print("Wrong number of arguments!\n")
    print("Usage: python", sys.argv[0], "model features clima subclima pix_i pix_f")
    print("- model can be LSTM or BiLSTM")
    print("- features can be SIF or noSIF")
    print("- clima is any main climatic region of Köppen-Geiger classification (see paper)")
    print("- subclima is any subregion of the main climatic region (see paper)")
    print("- pix_i and pix_f defines range of pixels of selected climatic subregion (see paper)")
    
    sys.exit(1)

def set_random_number_generator(seed = 42):
    print("\nSeed of random sequence is " + str(seed))
    np.random.seed(seed)   
    

def main():
    
    ### Get user's arguments
    
    # Check number of arguments
    n = len(sys.argv)
    if n != 7:
        usage()
    
    model = sys.argv[1]                 #### LSTM or BiLSTM
    features = sys.argv[2]              #### Predictors with/without SIF data (SIF or noSIF)
    clima = sys.argv[3]                 #### Main climatic region
    subclima = sys.argv[4]              #### Climatic subregion
    pix_i = int(sys.argv[5])            #### Indicate range of grid cells: from ...
    pix_f = int(sys.argv[6])            #### Indicate range of grid cells: ... to
    
    ### Instantiate global parameters
    gpar = global_par()
    
    print("\nTrain", model, "model")
    
    if features == "SIF":
        print("\nSet of predictors: meteorological forcings and SIF data")
        model_name = model               # Define model name
        
    else:
        print("\nSet of predictors: meteorological forcings only")  
        model_name = model + features    # Define model name
        
    print("\nClimatic region:", clima, subclima)
    print("\nFit over grid cell:", pix_i, "-", pix_f)
    
    clima_name = clima + "_" + subclima
     
    # Define folders path
    path = "."
    data_dir = path + "/input_data/"                        # folder where input data are stored 
    optuna_dir = path + "/optuna/" + model_name + "/"        # folder where optuna databases are stored   

    
    ### Folders where output are saved     
    out_dir = path + "/model_fits/" + model_name + "/" + clima_name + "/"
    model_dir = out_dir + "model/"          ### folder to save best model fit out of 5 attempts
    scaler_dir = out_dir + "scalers/"       ### folder to save scalers of predictors and target variable
    fig_dir = out_dir + "plots/"            ### folder to save plots of best model fit
    
    ### Create output folders
    isExist = os.path.exists(out_dir)
    if not isExist:
       os.makedirs(out_dir)
       os.makedirs(scaler_dir)
       os.makedirs(model_dir)
       os.makedirs(fig_dir)
    
    # Set random number generator for shuffling of data w/ seed
    set_random_number_generator()
    
    ### Load Optuna hyperparameters
    hyppar = hyperparameters()
    hyppar.load_optuna(optuna_dir, model_name, clima_name)

        
    ### Load data array with target variable and predictors data from 1982 to 2021
    data_arr, id_pixels, input_cols, target_col = load_data(data_dir, clima_name, features)
        
    ### Loop over grid cells range
    for p in range(pix_i, pix_f):
        
        ### Initialize MAE before the first attempt of each pixel
        best_mae = np.inf   
        
        pidx = int(id_pixels[p])
        
        print("\nFit model in pixel " + str(p) + " (pixel ID: " + str(pidx) + ")")
        
        ### Extract predictors and target data for pixel p
        X, Y = extract_pixdata(data_arr, pidx, input_cols, target_col, hyppar.window)
        
        ### Define shuffle array if p is the first pixel (it's the same for all pixels in "clima_name")
        if p == pix_i:
            chunkSH_arr, obs_train, obs_val = chunk_shuffle(hyppar.window, Y.shape[0])
        
        """
        - Shuffle data 
        - Define training, validation and test datasets 
        - Fit and apply scalers for input and output data 
        """
        Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, s_in, s_out = get_datasets(X, Y, chunkSH_arr, obs_train, obs_val, hyppar.window)
        
        ### Save input scaler
        joblib.dump(s_in, scaler_dir + model_name + "_INscaler_PIXid_" + str(pidx) + "_" +  clima_name + ".gz")
        ### Save output scaler
        joblib.dump(s_out, scaler_dir + model_name + "_OUTscaler_PIXid_" + str(pidx) + "_" +  clima_name + ".gz")
        
        ### Configure model 
        model = model_config(hyppar, input_cols, pidx)
        
        
        # Display training progress by printing a single dot for each completed epoch
        class PrintDot(tensorflow.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0: print("")
                print(".", end="")        
        
        # Garbage collection at each epoch
        class GarbColl(tensorflow.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs = None):
                gc.collect()
        
        # Model fit stops if there is no improvment during ptc epochs
        early_stop = tensorflow.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = gpar.ptc)
        
        for attempt in range(1, gpar.tot_fit+1):
    
            print("\nModel fit over pixel", str(p), "- attempt n.", str(attempt))
            
            ### Train the model
            
            history = model.fit(Xtrain[:, :, :], Ytrain[:],    
                                epochs = gpar.epochs, batch_size = gpar.bsize, 
                                validation_data = (Xval[:, :, :],                 
                                                   Yval[:]), verbose = 2,         
                                shuffle=False, callbacks = [early_stop, PrintDot(), GarbColl()])               
            
            # Evaluate the model on training data
            print("\nEvaluate model on training dataset")
            results_train = model.evaluate(Xtrain, Ytrain, batch_size = gpar.bsize)
            print("Training loss, training acc:", results_train)
            
            # Evaluate the model on test data
            print("\nEvaluate model on test dataset")
            results = model.evaluate(Xtest, Ytest, batch_size = gpar.bsize)
            print("Test loss, test acc:", results)
    
            mae = history.history["mean_absolute_error"][-1]
            
            if mae < best_mae:
                
                best_mae = mae
                
                ### Save output
                
                # Save model fit
                model.save(model_dir + model_name + "_PIXid_" + str(pidx) + "_" + clima_name + ".h5")
                    
                # Plot MAE FUNCTION
                plt.plot(history.history["mean_absolute_error"], lw = 1.5)
                plt.title("model accuracy")
                plt.ylabel("MAE", fontsize = 16)
                plt.xlabel("Epoch", fontsize = 16)
                plt.legend(["train", "val"], loc="upper left")
                plt.xticks(fontsize = 14)
                plt.yticks(fontsize = 14)
                
                plt.savefig(fig_dir + model_name + "_MAE_PIXid_" + str(pidx) + "_" +  clima_name + ".png", dpi = 300)
                                
                # Plot LOSS FUNCTION
                plt.figure(figsize = (8, 6))
                plt.semilogy(history.history["loss"], lw = 1.5)
                plt.plot(history.history["val_loss"], lw = 1.5)
                plt.xlabel("Epoch", fontsize = 16)
                plt.ylabel("Loss", fontsize = 16)
                plt.legend(["train", "val"], loc="upper left")
                plt.xticks(fontsize = 14)
                plt.yticks(fontsize = 14)
                
                plt.savefig(fig_dir + model_name + "_lossF_PIXid_" + str(pidx) + "_" +  clima_name + ".png", dpi = 300)
                
                plt.close()
                

def load_data(data_dir, clima_name, features):
    
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
        
    ### Get pixel IDs and total number of grid cells
    id_pixels = np.unique(data_arr[:,colname["pixel_ID"]])
    n_pixels = id_pixels.shape[0]
    
    print("Total number of grid cells in ", clima_name, "is", str(n_pixels))
    
    return data_arr, id_pixels, input_cols, target_col


def extract_pixdata(data_arr, pidx, input_cols, target_col, window):
    
    # Select data for pixel p
    mask_pixid = (data_arr[:, colname["pixel_ID"]] == pidx)
    data_pix = data_arr[mask_pixid, :]
    
    print("\nPrepare datasets...")
    
    dataX = data_pix.copy()
    dataY = data_pix.copy()
    
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
        
    return X, Y


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
    chunk_test = n_chunk - chunk_train - chunk_val
    
    obs_cut = chunk_train*chunk_wdNEW
    val_cut = obs_cut + chunk_val*chunk_wdNEW
    obs_train = (chunkSH_arr[:obs_cut] > -99).sum()
    obs_val = (chunkSH_arr[obs_cut:val_cut] > -99).sum()
    obs_test = (chunkSH_arr[val_cut:] > -99).sum()
        
    chunkSH_arr = chunkSH_arr[chunkSH_arr > -99]
    
    print("Length of traning dataset is", str(obs_train))
    print("Length of validation and test datasets is", str(obs_val), "and", str(obs_test))

    return chunkSH_arr, obs_train, obs_val


def get_datasets(X, Y, chunkSH_arr, obs_train, obs_val, window):
    
    ################ Shuffle input data ################
    
    n_obs = Y.shape[0]
    
    ### Shuffle first dimension of X dataset
    X = X.reshape((X.shape[0]//window, window, X.shape[1]))
    
    X = X[chunkSH_arr,...]
    X = X.reshape((-1, X.shape[2]))
    
    # Define the scaler of input data and create a mask to divide datasets
    s_in = RobustScaler()   

    s_in_mask = np.zeros(n_obs * window, dtype=np.bool_)  
    s_in_mask[:obs_train * window] = True                     
    s_in.fit(X[s_in_mask, :]) #### fit scaler only over training dataset
    X[:, :] = s_in.transform(X[:, :])
    
    X = X.reshape((X.shape[0]//window, window, X.shape[1]))

    ################ Remove the first time step of the training dataset ################
    Y_obs0 = np.repeat(Y[0::Y.shape[0]], Y.shape[0])
    Y[:] = Y[:]-Y_obs0[:]

    # print("The shape of X array is ", X.shape)
    # print("The shape of Y array is ", Y.shape)
    
    ################ Shuffle output data ################

    Y = Y[chunkSH_arr]

    ### Select data for training
    mask_train = np.mod(np.arange(X.shape[0]), n_obs) < obs_train

    ### Training dataset
    Xtrain = X[mask_train, :, :]
    Ytrain = Y[mask_train]
                           
    dataX_val_test = X[~mask_train, :, :]
    dataY_val_test = Y[~mask_train]

    ### Select data for validation
    mask_val = np.mod(np.arange(dataX_val_test.shape[0]), n_obs - obs_train) < obs_val

    ### Validation and test datasets
    Xval = dataX_val_test[mask_val, :, :]
    Yval = dataY_val_test[mask_val]

    Xtest = dataX_val_test[~mask_val, :, :]
    Ytest = dataY_val_test[~mask_val]

    ### Fit scaler over Y training dataset and apply on all datasets
    s_out = RobustScaler()
    Ytrain = s_out.fit_transform(Ytrain.reshape(-1, 1))[:,0]
    Yval = s_out.transform(Yval.reshape(-1, 1))[:,0]
    Ytest = s_out.transform(Ytest.reshape(-1, 1))[:,0]
    
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, s_in, s_out
    

def model_config(hyppar, input_cols, pidx):
    
    ### Allocate memory for GPU job and define model 
    gpus = tensorflow.config.list_physical_devices("GPU")
    if gpus:
      # Restrict TensorFlow to only allocate 20GB of memory on the first GPU
      try:
        tensorflow.config.set_logical_device_configuration( gpus[0],
                [tensorflow.config.LogicalDeviceConfiguration(memory_limit=26000)])
        logical_gpus = tensorflow.config.list_logical_devices("GPU")
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
      except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)
    
    model = Sequential()
    
    if model == "LSTM":
        ### LSTM model 
        model.add(CuDNNLSTM(hyppar.units_in), input_shape=(hyppar.window, len(input_cols)))
    else:
        ### Bidirectional LSTM model 
        model.add(Bidirectional(CuDNNLSTM(hyppar.units_in),
                                input_shape=(hyppar.window, len(input_cols))))
    
    if hyppar.units_lay1 > 0 and hyppar.units_lay2 == 0:
        model.add(Dense(hyppar.units_lay1, activation = "tanh"))
    elif hyppar.units_lay2 > 0:
        model.add(Dense(hyppar.units_lay1, activation = "tanh"))
        model.add(Dense(hyppar.units_lay2, activation = "tanh"))
    model.add(Dense(1, activation = None))

    """
    Configure the model for training:
        - optimized w/ the efficient Adam version of sthocastic gradient descent method;
        - mean absolute error (MAE) regression metric during training and testing;
        - mean squared error loss function.
    """
    
    opt = Adam(lr = hyppar.l_rate)
    
    model.compile(loss = tensorflow.keras.losses.MeanSquaredError(),
                    optimizer = opt, 
                    metrics = [tensorflow.keras.metrics.MeanAbsoluteError()])
    
    # Show a summary of model
    model.summary()
        
    return model 
    

if __name__ == "__main__":
    main()



    
############################ END ############################
    
    
