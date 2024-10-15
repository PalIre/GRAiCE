# GR*Ai*CE dataset - Scripts description

This folder contains Python scripts used to develop the GR*Ai*CE dataset described in our paper (currently under review). Each script requires specific input arguments, which are listed below:

-	_model_: specify the model type, either _LSTM_ or _BiLSTM_;
-	_features_: specify the set of predictors to use, either _SIF_ (meteorological data + SIF data) or _noSIF_ (meteorological data only);
-	_clima_: name of the main climatic region, i.e., _tropical_, _arid_, _temperate_, _continental_, _polar_;
-	_subclima_: name of the climatic subregion, e.g., within the tropical region, you can choose one among tropical, rainforest, savannah. (See the table at the end of this document for a full list of subregion names);
-	_pix_i_: initial value of the pixels range for model training;
-	_pix_f_: final value of the pixels range for model training.

The following scripts must be executed in the order specified below (and in their file name) to reproduce the GR*Ai*CE dataset:

1.	_**01_optuna.py**_: determines the optimal values of 5 model hyperparameters using the Optuna tuning tool, based on GRACE data from 2002 to 2017. The optimization is performed at the climatic region level (not per grid cell). Users must specify _model_, _features_, _clima_, and _subclima_ arguments.
2.	_**02_model_training.py**_: trains the model at the grid-cell level using GRACE/GRACE-FO data (2002-2021). Input data (target variable and predictors) are divided into chunks and shuffled; 60% of data chunks is used for training, while the remaining chunks are used for validation (20%) and test (20%). For each grid cell, the model is fit 5 times, and the fit with the lowest Mean Absolute Error (MAE) is saved in an h5 format, along with scalers for input and output data. Users must specify _model_, _features_, _clima_, and _subclima_, _pix_i_, and _pix_f_ arguments.
3.	_**03_apply_model.py**_: applies the model fit obtained in _02_model_training.py_ to pixels within a specific climatic subregion. The script saves an array (.npy format) reporting data associated with all grid cells of the region. Specifically, for each pixel the array includes pixel ID, latitude, longitude, year, month, TWSA observed values, TWSA predictions, and a flag value indicating whether the observed value was used in the training (=0), validation (=1), or test (=2) procedure. Users must specify _model_, _features_, _clima_, and _subclima_ arguments.
4.	_**04_PCC_training.py**_: evaluates the Pearson’s correlation coefficient (PCC) for predictions across a specified climatic subregion. PCC values are computed for the training, validation, and test datasets, as well as across the whole GRACE/GRACE-FO time series. Grid cells with PCC below 0.60 in the training + validation dataset are flaged and an array containing their pixel IDs is saved further tuning. Users must specify _model_, _features_, _clima_, and _subclima_ arguments.
5.	_**05_optunaLR.py**_: re-tunes the learning rate (LR) using the Optuna tuning tool within grid cells identified by _04_PCC_training.py_, based on GRACE data from 2002 to 2017. Optimization is performed at the climatic region level (not per grid cell). Users must specify _model_, _features_, _clima_, and _subclima_ arguments.
6.	_**06_newLR_model_training.py**_: retrains the model at the grid-cell level using GRACE/GRACE-FO data (2002-2021) only for pixels identified in 04_PCC_training.py, following the same setup as in _02_model_training.py_. The best fit out of 5 is saved in h5 format. Users must specify _model_, _features_, _clima_, and _subclima_, _pix_i_, and _pix_f_ arguments.
7.	_**07_apply_newLR_model.py**_: applies the new model fit from _06_newLR_model_training.py_ to pixels identified in _04_PCC_training.py_. The script saves an array as in _03_apply_model.py_, but limited to the specified grid cells. Users must specify _model_, _features_, _clima_, and _subclima_, _pix_i_, and _pix_f_ arguments.
8.	_**08_PCC_comparison.py**_: evaluates and compares the PCC values of the retrained models within a specified climatic subregion, identifying grid cells where the new learning rate improved performance and saving their IDs. Users must specify _model_, _features_, _clima_, and _subclima_ arguments.
9.	_**09_merge_fit.py**_: merges predictions from the original and new LR models across all climatic subregions. Models using new LR values are applied only to pixels where they produce increased PCC (as determined in _08_PCC_comparison.py_). The script saves an array as in 03_apply_model.py, creating the final version of the GR*Ai*CE predictions. Users must specify _model_ and _features_ arguments.

Table of climatic subregion names (refer to this table for valid subregion names when specifying the subclima argument):

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
