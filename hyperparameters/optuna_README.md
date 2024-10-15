# GR*Ai*CE dataset - Hyperparameters description

This folder contains databases of optimal values of models’ hyperperameters as determined by the Optuna tuning tool. Specifically, each model’s subfolder includes files in db format describing values of the following 5 hyperparameters for all 28 Köppen-Geiger climatic region (see table [here](../script/scripts_README.md)):

-	_l_rate_: learning rate;
-	_units_: number of LSTM/BiLSTM units;
-	_n_layers_: number of hidden layers;
-	_n_units_l1_ and _n_units_l2_: number of units of hidden layers;
-	_window_: lag in input data.

Each subfolder also contains a newLR folder with .db files of updated learning rate (LR) value determined by Optuna. The new LR was evaluated only for grid cells where the original LR resulted in a Pearson’s correlation coefficient (PCC) lower than 0.60 in the training + validation dataset. Note that in climatic regions where no grid cells had PCC < 0.60, a new LR value was not calculated.
