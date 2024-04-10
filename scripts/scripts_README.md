# GR*Ai*CE dataset - Scripts description

This folder includes Python scripts used to develop the GR*Ai*CE dataset described in our paper (under review).  
All scripts require some input arguments to be defined by the user. They are listed below:

-	_model_: specify model type, i.e., _LSTM_ or _BiLSTM_;
-	_features_: specify set of predictors to use, i.e., _SIF_ (meteorological data + SIF data) or _noSIF_ (meteorological data only);
-	_clima_: name of the main climatic region, i.e., _tropical_, _arid_, _temperate_, _continental_, _polar_;
-	_subclima_: name of the climatic subregion, e.g., for the tropical main climatic region choose one among tropical, rainforest, savannah (for a complete list of subregions names, see the table at the end of this document);
-	_pix_i_: initial value of pixels range over which the model training will be performed;
-	_pix_f_: last value of pixels range over which the model training will be performed.

The following scripts must be executed in the order specified below (and in their file name) to reproduce the GR*Ai*CE dataset:

1.	_**01_optuna.py**_: determine 5 model hyperparameters with the Optuna tuning tool using GRACE data from 2002 to 2017. The optimization is performed over climatic regions (not per grid cell). User must specify _model_, _features_, _clima_, and _subclima_ arguments.  
2.	_**02_model_training.py**_: train model at the grid cell level using GRACE/GRACE-FO data (2002-2021). Input data (target variable and predictors) are divided in chunks and shuffled. Then 60% of data chunks are used for the training of the model, while the remaining chunks are used as validation (20%) and test (20%) datasets. For each grid cell, 5 fit attempts are performed. The fit producing the lowest MAE (mean absolute error) value will be assigned to the grid cell and the model will be saved in h5 format. The script also saves the scalers applied on input (predictors) and output (target variable) data. User must specify _model_, _features_, _clima_, and _subclima_, _pix_i_, and _pix_f_ arguments
3.	_**03_apply_model.py**_: apply model fit obtained in _02_model_training.py_ on pixels of a specific climatic subregion. The script saves an array (in .npy format), which reports data associated to all grid cells of the region. Specifically, for each pixel the array includes pixel ID, latitude, longitude, year, month, TWSA observed values, TWSA predictions, and a flag value indicating whether the observed value was used in the training (=0), validation (=1), or test (=2) procedure. User must specify _model_, _features_, _clima_, and _subclima_ arguments.  
4.	_**04_PCC_training.py**_: evaluate Pearson’s correlation coefficient produced by predictions over a specific climatic subregion. PCC is computed within different datasets (i.e., training, validation, and test) as well as across the whole GRACE/GRACE-FO time series. The script also identifies grid cells where PCC in the training + validation dataset is less than 0.60 and saves an array containing their pixel IDs and PCC values. User must specify _model_, _features_, _clima_, and _subclima_ arguments.  
5.	_**05_optunaLR.py**_: determine new learning rate (LR) value with Optuna tuning tool in pixels identified by _04_PCC_training.py_ using GRACE data from 2002 to 2017. The optimization is performed over climatic regions (not per grid cell). User must specify _model_, _features_, _clima_, and _subclima_ arguments.  
6.	_**06_newLR_model_training.py**_: train model at the grid cell level using GRACE/GRACE-FO data (2002-2021) only over pixels identified by 04_PCC_training.py. Model set-up is the same followed in _02_model_training.py_. The new best fit out of 5 is saved in a folder in h5 format. User must specify _model_, _features_, _clima_, and _subclima_, _pix_i_, and _pix_f_ arguments.    
7.	_**07_apply_newLR_model.py**_: apply the new model fit obtained in _06_newLR_model_training.py_ on pixels of a specific climatic subregion, as identified in _04_PCC_training.py_. The script saves an array as in _03_apply_model.py_, but only for the specified grid cells. User must specify _model_, _features_, _clima_, and _subclima_, _pix_i_, and _pix_f_ arguments.  
8.	_**08_PCC_comparison.py**_: evaluate Pearson’s correlation coefficient produced by predictions over pixels of a specific climatic subregion as identified in _04_PCC_training.py_. The script computes new values of PCC within the training + validation dataset and then compares them to previous values obtained in _04_PCC_training.py_, to identify grid cells where PCC has improved and save their IDs. User must specify _model_, _features_, _clima_, and _subclima_ arguments.  
9.	_**09_merge_fit.py**_: merge predictions produced by model fits obtained with original and new LR values for all climatic subregions. In particular, models based on new LR values are used only in pixels where they produce an increase in PCC (as determined in _08_PCC_comparison.py_). The script saves an array as in 03_apply_model.py, providing the final version of the GRAiCE predictions. User must specify _model_ and _features_ arguments.

Table of climatic subregion names to be specified as arguments of GR*Ai*CE scripts:

|Main clima           |Köppen-Geiger subregion name     |Subregion name in GR*Ai*CE           |
|:---|---|:---:|
|**Tropical**         |Rainforest                       |rainforest                           |
|                     |Monsoon                          |monsoon                              |
|                     |Savannah                         |savannah                             |
|**Arid**             |Desert hot                       |deserthot                            |
|                     |Desert cold                      |desertcold                           |
|                     |Steppe hot                       |steppehot                            |
|                     |Steppe cold                      |steppecold                           |
|**Temperate**        |Dry hot summer                   |dryhotsummer                         |
|                     |Dry warm summer                  |drywarmsummer                        |
|                     |Dry winter hot summer            |drywinhotsum                         |
|                     |Dry winter warm summer           |drywinwarmsum                        |
|                     |No dry season hot summer         |nodryhotsum                          |
|                     |No dry season warm summer        |nodrywarmsum                         |
|                     |No dry season cold summer        |nodrycoldsum                         |
|**Continental**      |Dry hot summer                   |dryhotsummer                         |
|                     |Dry warm summer                  |drywarmsummer                        |
|                     |Dry cold summer                  |drycoldsummer                        |
|                     |Dry summer very cold winter      |drysumverycoldwin                    |
|                     |Dry winter hot summer            |drywinhotsum                         |
|                     |Dry winter warm summer           |drywinwarmsum                        |
|                     |Dry winter cold summer           |drywincoldsum                        |
|                     |Dry winter very cold winter      |drywinverycoldwin                    |
|                     |No dry season hot summer         |nodryhotsum                          |
|                     |No dry season warm summer        |nodrywarmsum                         |
|                     |No dry season cold summer        |nodrycoldsum                         |
|                     |No dry season very cold winter   |nodryverycoldwin                     |
|**Polar**            |Tundra                           |tundra                               |
|                     |Frost                            |frost                                |
