#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:44:03 2024

@author: Irene Palazzoli

Check PCC values within training dataset

Save array of pixels index for which PCC in training + validation is < 0.60

"""

import os, sys

# Libraries for plots and array manipulation
import numpy as np


# Define dictionary for column names of predictions array
colname = { "pixel_ID":0, "lat":1, "lon":2, "year":3, "month":4, 
           "obs":5, "pred":6, "dataset":7 }

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
    
    print("\nGet PCC in training + validation datasets for", model, "model")
    
    if features == "SIF":
        print("\nSet of predictors: meteorological forcings and SIF data")
        model_name = model               # Define model name
        
    else:
        print("\nSet of predictors: meteorological forcings only")  
        model_name = model + features    # Define model name
        
     
    # Define folders path
    path = "."
    twsamodel_dir = path + "/model_predictions/" + model_name + "/climatic_regions/" 
    out_dir = path + "/model_fits/" + model_name + "/lowPCC_train/"   ### Save list of pixels here
    
    isExist = os.path.exists(out_dir)
    if not isExist:
       os.makedirs(out_dir)
       
    for clima_num in clima_class:   
              
        clima_name = clima_regs[clima_num - 1]
        print("Load data for", clima_name + " region")
        
        twsa = np.load(twsamodel_dir + model_name + "_predictions_" + clima_name + ".npy")
        
        id_pixels = np.unique(twsa[:,colname["pixel_ID"]])
        n_pixels = id_pixels.shape[0]
        
        PCC_array = np.zeros((n_pixels, 6))
       
        for p in range(n_pixels):   
            
            pidx = int(id_pixels[p])
            PCC_array[p, 0] = pidx

            mask_pix = twsa[:,colname["pixel_ID"]] == pidx
            x_pix = twsa[mask_pix,colname["pred"]]
            
            if np.all(x_pix == 0):
                break
            
            y_pix = twsa[mask_pix,colname["obs"]]
            
            mask_train = twsa[mask_pix,colname["dataset"]] == 0
            mask_val = twsa[mask_pix,colname["dataset"]] == 1
            mask_trainVal = np.logical_or(mask_train, mask_val)
            mask_test = twsa[mask_pix,colname["dataset"]] == 2            
            mask_all = ~np.isnan(twsa[mask_pix,colname["obs"]])
            
            PCC_array[p, 1] = pcc(x_pix[mask_train], y_pix[mask_train])
            PCC_array[p, 2] = pcc(x_pix[mask_val], y_pix[mask_val])
            PCC_array[p, 3] = pcc(x_pix[mask_trainVal], y_pix[mask_trainVal])
            PCC_array[p, 4] = pcc(x_pix[mask_test], y_pix[mask_test])
            PCC_array[p, 5] = pcc(x_pix[mask_all], y_pix[mask_all])
            
        mask_lowPCC = PCC_array[:, 3] < 0.6
        
        if mask_lowPCC.sum() > 0:
            
            print("PCC < 0.6 in " + str(mask_lowPCC.sum()) + " grid cells! \n")
            
            lowPCC_pixel = PCC_array[mask_lowPCC, :]
            
            np.save(out_dir + "lowPCC_pixels_" + model_name + "_" + clima_name, lowPCC_pixel)
            
def pcc(mod, obs): 
 
    # Fitting variable
    x = obs
    y = mod
 
    # r2 value
    corr_1 = np.corrcoef(x.astype('float'), y.astype('float'))
    PCC = np.round(corr_1[0,1], 2)
    
    return PCC
        
    
if __name__ == "__main__":
    main()



    
############################ END ############################
    
    




