# GR*Ai*CE dataset
## Reconstructions of Terrestrial Water Storage Anomalies (TWSA)

### by Irene Palazzoli<sup>1</sup>, Serena Ceola<sup>1</sup>, and Pierre Gentine<sup>2</sup>
#### <sup>1</sup>University of Bologna, <sup>2</sup>Columbia University

Terrestrial Water Storage (TWS) is the total amount of freshwater stored on and below the Earth’s land surface, including surface water, groundwater, soil moisture, snow, and ice. TWS is a crucial variable in the global hydrologic cycle, representing an essential indicator of water availability. Since 2002, the Gravity Recovery and Climate Experiment (GRACE) mission and its follow-on (GRACE-FO) have been measuring temporal and spatial variations of TWS, namely the Terrrestrial Water Storage Anomalies (TWSA), enabling the monitoring of global hydrological changes over the last two decades. However, the lack of observations prior to 2002 along with the temporal gaps in GRACE/GRACE-FO time series limit our understanding of long-term trends in global freshwater availability.

In this study, we developed the GR*Ai*CE dataset, which offers four global monthly reconstructions of TWSA from 1984 to 2021 at a 0.5º spatial resolution. Using Long Short-Term Memory (LSTM) and Bidirectional LSTM (BiLSTM) neural networks, we employed two sets of predictors to create these reconstructions. The first set combines five essential meteorological variables with data on vegetation dynamics, while the second set includes only the meteorological variables. Specifically, the meteorological predictors consist of monthly averaged data on total precipitation, snow water equivalent, net surface solar radiation, surface air temperature, and surface air relative humidity. Vegetation dynamics are represented by long-term reconstructions of solar-induced fluorescence (SIF), which serves as a proxy for photosynthesis. Each model was trained using monthly TWSA data from the GRACE JPL mascon dataset. The GR*Ai*CE dataset accurately reproduces GRACE/GRACE-FO observations at both global and regional scales across diverse climatic zones. Our models outperform previous reference works, offering reliable water budget estimates at the river basin scale. Beyond providing continuous long-term TWSA time series, these models enable the detection and analysis of TWS changes driven by climate variability and change.

This repository contains the code, input data arrays (predictors and target variables), and model hyperparameters used to develop the GR*Ai*CE dataset. It includes four TWSA reconstructions: two from LSTM and BiLSTM models trained with all predictors (including SIF data), and two from models trained with only meteorological variables (excluding SIF data).

The GR*Ai*CE dataset was created based on the following approach:  
:bangbang: *indicates computationally intensive steps which were performed on the Columbia University HPC cluster*  

__1.	Data collection and preprocessing.__ Gridded monthly GRACE/GRACE-FO observations of TWSA (target variable) were collected from the the JPL mascon dataset, serving as the target variable. Data on meteorological forcings and vegetation dynamics were also gathered as predictors. All predictor datasets were resampled to match the 0.5º spatial resolution of the JPL mascon solutions.
__2.	Preparation of input data.__ Both the target variable and predictors were merged into the same data array for hyperparameters optimization and model training. An input data array was generated for each of the 28 climatic regions defined by the Köppen-Geiger classification system (see input_data folder).  
__3.	Hyperparameters optimization.__ Optimal values for five key hyperparameters were determined across the climatic regions for each model type (i.e., LSTM, BiLSTM, LSTMnoSIF, BiLSTMnoSIF) and set of predictors using the Optuna tuning tool. The best parameters values were established after 100 trials (see optuna folder). :bangbang:  
__4.	Models training.__ Each model was trained at the grid cell level within each climatic region using the optimal hyperparameters specific to the climatic region, model type, and set of predictors. :bangbang:  
__5.	PCC evaluation.__ Pearson’s correlation coefficient (PCC) was estimated between GRACE/GRACE-FO TWSA observations and models predictions across the training, validation and test datasets, as well as over the whole 2002-2021 observation period. Grid cells with PCC values below 0.60 in the training + validation dataset were marked as pixels of low model performance.  
__6.	Learning rate improvement.__ The learning rate (LR) value was re-tuned using Optuna within the subset of grid cells identified in step #5. The best LR was determined after 20 trials (see optuna/newLR folder). :bangbang:  
__7.	Models training with new LR.__ Models were retrained using the newly optimized LR values, byt only for the grid cells identified in step #6. :bangbang:  
__8.	PCC comparison.__ PCC values for the retrained models were computed in the identified grid cells. The new PCC values were then compared to those from the original training to determine if performance had improved.
__9.	Global predictions definition.__ Predictions from both the original and new LR values were merged based on pixel ID to produce the final global dataset. 
__10.	Models performance analysis.__ Models performance was estimated with various performance metrics and by comparing GR*Ai*CE predictions against other TWSA products.

For a detailed description of the modeling approach, users are advised to refer to our paper ([Palazzoli et al., 2025](https://doi.org/10.1038/s41597-025-04403-3)) before usage.  

Questions about the dataset can be directed to irene.palazzoli@unibo.it.  
The GR*Ai*CE dataset is available [here](https://doi.org/10.5281/zenodo.10953658).
