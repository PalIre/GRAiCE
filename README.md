# GR*Ai*CE dataset
## Reconstructions of Terrestrial Water Storage Anomalies (TWSA)

### by Irene Palazzoli<sup>1</sup>, Serena Ceola<sup>1</sup>, and Pierre Gentine<sup>2</sup>
#### <sup>1</sup>University of Bologna, <sup>2</sup>Columbia University

Terrestrial Water Storage (TWS) is the total amount of freshwater stored on and below the Earth’s land surface, including surface water, groundwater, soil moisture, snow, and ice. As a result, TWS is a crucial variable of the global hydrologic cycle, representing an essential indicator of water availability. Since 2002, the Gravity Recovery and Climate Experiment (GRACE) mission and its follow-on (GRACE-FO) have been measuring temporal and spatial variations of TWS, namely the Terrrestrial Water Storage Anomalies (TWSA), enabling the monitoring of global hydrological changes over the last two decades. However, the lack of observations prior to 2002 along with the temporal gaps in GRACE/GRACE-FO time series limit our understanding of long-term variations of global freshwater availability.

In this study, we use Long Short-Term Memory (LSTM) and Bidirectional LSTM (BiLSTM) neural networks and two sets of predictors to develop four global monthly reconstructions of TWSA from 1984 to 2021 at 0.5º spatial resolution (GR*Ai*CE). The first set of predictors includes five fundamental meteorological forcings only, whereas the second set of predictors is given by a combination of the five meteorological forcings and data on vegetation dynamics. Specifically, the meteorological predictors are monthly averaged data of total precipitation, snow depth water equivalent, surface net solar radiation, surface air temperature, and surface air relative humidity. We derive data on vegetation dynamics from a long-term reconstruction of solar-induced fluorescence (SIF), which represents a proxy for photosynthesis. Each model is trained with monthly TWSA data of the GRACE JPL mascon dataset. The GR*Ai*CE dataset accurately reproduces GRACE/GRACE-FO observations at the global scale and across different climatic regions. Moreover, we found that the models predict the observed TWSA and water budget at the river basin scale more accurately than previous work.

This repository contains codes, arrays of input data (preditors and target variable), and models hyperparameters used to develop the models of the GR*Ai*CE dataset. Codes produced four TWSA reconstructions: two TWSA reconstructions obtained from LSTM and BiLSTM models fed with all predictors (i.e., including SIF data), and two TWSA reconstructions obtained from LSTM and BiLSTM models fed with meteorological forcings only (i.e., without SIF data). 

The GR*Ai*CE dataset was created based on the following approach:  
:bangbang: *indicates computationally heavy and/or memory intensive steps which were performed on the Columbia University HPC cluster*  

__1.	Data collection and preprocessing.__ Gridded monthly GRACE/GRACE-FO observations of TWSA (target variable) were collected as provided by the the JPL mascon dataset as well as data on meteorological forcings and vegetation dynamics (predictors). All predictors data were resampled to the 0.5º spatial resolution of JPL mascon solutions.  
__2.	Preparation of input data.__ Both the target variable and predictors were merged into the same data array to be used in hyperparameters optimization and model training. An input data array was produced for each climatic region of the Köppen-Geiger classification system, for a total number of 28 arrays/climatic regions (see input_data folder).  
__3.	Hyperparameters optimization.__ Optimal values of five key hyperparameters were evaluated across the climatic regions for each model (i.e., LSTM, BiLSTM, LSTMnoSIF, BiLSTMnoSIF) and set of predictors with the Optuna tuning tool. Best parameters values were determined after 100 trials (see optuna folder). :bangbang:  
__4.	Models training.__ Each model was trained at the grid cell level in each climatic region using the optimal values of hyperparameters associated to the climatic region, model type, and set of predictors. :bangbang:  
__5.	PCC evaluation.__ The Pearson’s correlation coefficient (PCC) was estimated between GRACE/GRACE-FO TWSA and models TWSA predictions within the training, validation and test datasets, as well as across the whole 2002-2021 time series of observations. Then, grid cells showing a PCC < 0.60 in the training + validation dataset were identified as pixels with low model performance.  
__6.	Learning rate improvement.__ The learning rate (LR) value was tuned again with Optuna in the subset of grid cells identified in step #5 in each climatic region. The best LR value was determined after 20 trials (see optuna/newLR folder). :bangbang:  
__7.	Models training with new LR.__ Each model was trained again using the new LR values only over the pixels identified in step #6. :bangbang:  
__8.	PCC comparison.__ PCC values obtained with the new LR were computed in the identified grid cells. The new PCC values of the training + validation dataset were then compared to those obtained with the original LR to establish in which grid cells the new LR has improved the model performance.  
__9.	Global predictions definition.__ Predictions obtained with the original and new LR values were merged based on pixel ID.  
__10.	Models performance analysis.__ Models performance was estimated with several performance metrics and by comparing GRAiCE predictions to other TWSA products.

For a detailed description of modelling approach, users are advised to refer to our paper (currently under review) before usage.  
Questions regarding the dataset can be directed to irene.palazzoli@unibo.it.  
The GR*Ai*CE dataset is available [here](https://doi.org/10.5281/zenodo.10953658).

