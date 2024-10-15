#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 11 11:35:51 2022

@author: Irene Palazzoli

Optimize 5 hyperparameters for LSTM or BiLSTM model 

GRACE data from 2002 to 2017

Model: LSTM or BiLSTM

Set of predictors:
    SIF:    meteorological + vegetation dynamics data
    noSIF:  meteorological predictors only

Run 100 trials

"""

###### Find hyperparameters for Bidirectional LSTM model ###### 

# System libraries
import sys, gc

# Libraries for plots and array manipulation
import numpy as np
import optuna
from optuna.integration import KerasPruningCallback
from optuna.trial import TrialState

# LSTM libraries
import tensorflow 
from tensorflow.keras import layers, activations, regularizers
from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout, Input
from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

# Other libraries
from sklearn.preprocessing import RobustScaler


### Define global variables

# Define dictionary for column names of data array
colname = { "pixel_ID":0, "lat":1, "lon":2, "year":3, "month":4, 
           "pcp":5, "tmp":6, "snow":7, "rad":8, "rh":9, "sif":10,
           "clima":11, "tws":12 }

class global_par():
    
    def __init__(self, ptc = 30, epochs = 1000, bsize = 32, n_trials = 100, timeout = 36000):
        
        self.n_trials = n_trials    ## max number of trials
        self.timeout = timeout      ## stop trial after timeout
        self.ptc = ptc              ## patience
        self.epochs = epochs        ## number of epochs
        self.bsize = bsize          ## batch size
        
        
def usage():
    print("Wrong number of arguments!\n")
    print("Usage: python", sys.argv[0], "model features clima subclima")
    print("- model can be LSTM or BiLSTM")
    print("- features can be SIF or noSIF")
    print("- clima is any main climatic region of Köppen-Geiger classification (see paper)")
    print("- subclima is any subregion of the main climatic region (see paper)")
    
    sys.exit(1)

        
def set_random_number_generator(seed = 42):
    print("\nSeed of random sequence is " + str(seed))
    np.random.seed(seed)   
     

class TFKerasPruningCallback(Callback):

    def __init__(self, trial, monitor):
        # type: (optuna.trial.Trial, str) -> None

        super(TFKerasPruningCallback, self).__init__()

        self.trial = trial
        self.monitor = monitor

    def on_epoch_end(self, epoch, logs=None):
        # type: (int, Dict[str, float]) -> None

        logs = logs or {}
        current_score = logs.get(self.monitor)
        if current_score is None:
            return
        self.trial.report(current_score, step=epoch)
        if self.trial.should_prune():
            message = "Trial was pruned at epoch {}.".format(epoch)
            raise optuna.structs.TrialPruned(message)


def prepare_data(window, features):
    
    # Load data
    data_arr = np.load(data_dir + "INPUTdata_" + clima_name + ".npy")
    
    # List of columns to use as predictors
    if features == "SIF":
        
        ### Remove grid cells where SIF data are not available
        pixel_mask = np.isnan(data_arr[:,input_cols]).any(axis=1)
        id_toremove, count = np.unique(data_arr[pixel_mask, colname["pixel_ID"]].astype(np.int32), return_counts=True)
        pixel_mask = np.in1d(data_arr[:, colname["pixel_ID"]].astype(np.int32), id_toremove, invert=True)

        data_arr = data_arr[pixel_mask, :]

    id_pixels = np.unique(data_arr[:,colname["pixel_ID"]].astype(np.int32))
    n_pixels = id_pixels.shape[0]
    
    # Set random number generator for shuffling of data w/ seed
    set_random_number_generator()

    dataX = data_arr.copy()
    dataY = data_arr.copy()
    
    dataX = dataX[:, input_cols]
    dataY = dataY[:, target_col]
    
    ### Index where TWS is not NA
    dataY_ID = ~np.isnan(dataY)
    idx = np.arange(dataY_ID.shape[0], dtype=np.int32)
    idx = idx[dataY_ID]
    
    ### Observation data    
    Y = dataY[dataY_ID]
    
    Y_perPix = int(Y.shape[0]/n_pixels)
     
    ### Prepare input data (X) for all observations Y
    X = -np.ones((n_pixels * Y_perPix * window, len(input_cols)), dtype = np.float32)
    
    offset = 0
    
    for out in range(0, Y_perPix * n_pixels):
        X[offset:offset+window, :] = dataX[idx[out]-window:idx[out], :]
        offset += window
        
    ################ Shuffle input data ################
    
    X = X.reshape((n_pixels, -1, window, X.shape[1]))
    
    # Initialize width of data chunks (must be larger than window)
    chunk_wd = window + 1
    
    n_chunk = Y_perPix//chunk_wd 
    chunk_wdNEW = Y_perPix//n_chunk
    
    if Y_perPix%n_chunk:
        chunk_wdNEW += 1
        
    chunk_arr = -99*np.ones(n_chunk*chunk_wdNEW, dtype=np.int32)
    
    index = np.array(range(Y_perPix), dtype=np.int32)
    
    n_chunk_missing = n_chunk*chunk_wdNEW-Y_perPix
    
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

    ### Number of chunks for trainign and validation
    chunk_train = int(0.6 * n_chunk)
    chunk_val = int((n_chunk - chunk_train)/2)
    chunk_test = n_chunk - chunk_train - chunk_val

    obs_cut = chunk_train*chunk_wdNEW
    val_cut = obs_cut + chunk_val*chunk_wdNEW
    obs_train = (chunkSH_arr[:obs_cut] > -99).sum()
    obs_val = (chunkSH_arr[obs_cut:val_cut] > -99).sum()
    obs_test = (chunkSH_arr[val_cut:] > -99).sum()
        
    chunkSH_arr = chunkSH_arr[chunkSH_arr > -99]
    
    
    ### Shuffle first dimension of X dataset
    
    X = X[:, chunkSH_arr,...]
    X = X.reshape((-1, X.shape[3]))
    
    print("Length of traning dataset is " + str(obs_train))
    print("Length of validation and test datasets is " + str(obs_val) + " and " + str(obs_test))

    
    # Create a mask to divide datasets and define the scaler of input data
    s_in = RobustScaler()   
    
    s_in_mask = np.zeros(Y_perPix * window, dtype=np.bool_)  
    s_in_mask[:obs_train * window] = True                     
    s_in_mask = np.repeat(s_in_mask, n_pixels)
    s_in.fit(X[s_in_mask, :]) #### define scaler only on training dataset
    X[:, :] = s_in.transform(X[:, :])
    
    X = X.reshape((X.shape[0]//window, window, X.shape[1]))
    
    ################ Remove the first time step of the training dataset ################
    Y_obs0 = np.repeat(Y[0::Y_perPix], Y_perPix)
    Y[:] = Y[:]-Y_obs0[:]
   
    
    ################ Shuffle output data ################
    ### Shuffle first dimension of X dataset
    Y = Y.reshape(n_pixels, -1)
    Y = Y[:, chunkSH_arr]
    Y = Y.reshape(-1)
    
    
    ##### Define training, validation, and test datasets #####
    mask_train = np.mod(np.arange(X.shape[0]), Y_perPix) < obs_train
    
    ### Training dataset
    Xtrain = X[mask_train, :, :]
    Ytrain = Y[mask_train]
                           
    dataX_val_test = X[~mask_train, :, :]
    dataY_val_test = Y[~mask_train]
    
    mask_val = np.mod(np.arange(dataX_val_test.shape[0]), Y_perPix - obs_train) < obs_val
    
    ### Validation and test datasets
    Xval = dataX_val_test[mask_val, :, :]
    Yval = dataY_val_test[mask_val]
    
    Xtest = dataX_val_test[~mask_val, :, :]
    Ytest = dataY_val_test[~mask_val]
    
    ### Apply scaler
    # Scale input variables and observed values
    
    s_out = RobustScaler()
    Ytrain = s_out.fit_transform(Ytrain.reshape(-1, 1))[:,0]
    Yval = s_out.transform(Yval.reshape(-1, 1))[:,0]
    Ytest = s_out.transform(Ytest.reshape(-1, 1))[:,0]
    
    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest


def create_model(trial, n_input):
    
    # Define range of values for hyperparameters tuned w/ Optuna
    units = trial.suggest_int("units", 10, 100)
    window = trial.suggest_int("window", 3, 24, 3)   
    l_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    n_layers = trial.suggest_int("n_layers", 1, 2)
    
    model = Sequential()
    
    if model == "LSTM":
        ### LSTM model 
        model.add(CuDNNLSTM(units), input_shape=(window, n_input))
    else:
        ### Bidirectional LSTM model 
        model.add(Bidirectional(CuDNNLSTM(units),
                                input_shape=(window, n_input)))
    for i in range(n_layers):
        num_hidden = trial.suggest_int("n_units_l{}".format(i), 1, 100)
        model.add(Dense(num_hidden, activation="tanh"))

    model.add(Dense(1, activation = None))
    
    
    # The model is fit w/ the efficient Adam version of sthocastic gradient 
    # descent method and optimized w/ the mean absolute error loss function
    opt = Adam(lr=l_rate)
    model.compile(loss=tensorflow.keras.losses.MeanSquaredError(),
                    optimizer=opt, # tf.keras.optimizers.Adam(),
                    metrics=[tensorflow.keras.metrics.MeanAbsoluteError()])   # metrics=[tf.keras.metrics.MeanSquaredError()])

    # Show a summary of the data (layers and their order,output shape, etc..)
    model.summary()
    
    return model, window


def objective(trial):
    
    ### Instantiate global parameters
    gpar = global_par()
    
    model, window = create_model(trial, len(input_cols))
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest = prepare_data(window)
    
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
    
    ### Train the model
    
    model.fit(Xtrain[:, :, :], Ytrain[:], 
              epochs = gpar.epochs, batch_size = gpar.bsize, 
              validation_data = (Xval[:, :, :],
                                 Yval[:]), verbose = 2,
              shuffle=False, callbacks = [early_stop, PrintDot(), GarbColl()])               
        
    # Evaluate the model on the test dataset
    print("Evaluate on test data")
    results = model.evaluate(Xtest, Ytest, batch_size = gpar.bsize, verbose=0)
    print("test loss, test MAE:", results)
    
    return results[1]

    
if __name__ == "__main__":
    
    ### Get user's arguments

    # Check number of arguments
    n = len(sys.argv)
    if n != 5:
        usage()

    model = sys.argv[1]                 #### LSTM or BiLSTM
    features = sys.argv[2]              #### Predictors with/without SIF data (SIF or noSIF)
    clima = sys.argv[3]                 #### Main climatic region
    subclima = sys.argv[4]              #### Climatic subregion
    
    ### Instantiate global parameters
    gpar = global_par()

    ### Identify target variable and predictors data
    target_col = colname["tws"]

    if features == "SIF":
        print("\nSet of predictors: meteorological forcings and SIF data")
        model_name = model               # Define model name
        input_cols = [colname[key] for key in colname.keys()][5:11]    
        
    else:
        print("\nSet of predictors: meteorological forcings only")  
        model_name = model + features    # Define model name
        input_cols = [colname[key] for key in colname.keys()][5:10]  ## no SIF data!
     
        
    print("\nOPTUNA optimizer on", model_name, "for", clima, "-", subclima, "climatic region")

    clima_name = clima + "_" + subclima

    # Load TWS and input data
    # Folder path
    path = "."
    data_dir = path + "/input_data/"                        # folder where input data are stored 
    optuna_dir = path + "/optuna/" + model_name + "/"        # folder where optuna databases are stored   

    ### Create Optuna database
    study = optuna.create_study(study_name = clima_name + "-study_OPTUNA_" + model_name, 
            storage = "sqlite:///" + optuna_dir + clima_name + "_OPTUNA_" + model_name + ".db", 
            load_if_exists=True, direction = "minimize", pruner = optuna.pruners.MedianPruner())
    
    study.optimize(objective, n_trials = gpar.n_trials, timeout = gpar.timeout)
    
    print("\n              Optuna for", clima, subclima, ": starting from trial n.", len(study.trials))
    
    pruned_trials = study.get_trials(deepcopy = False, states = [TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy = False, states = [TrialState.COMPLETE])
    
    print("\nStudy statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("\nBest trial:")
    trial = study.best_trial

    print("  Value: {}".format(trial.value))

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


