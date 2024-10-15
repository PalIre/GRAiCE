#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 09:44:03 2024

@author: Irene Palazzoli

Check PCC values obtained with new LR within training dataset,

compare to those associated to prevoius LR,

and determine in which grid cells new LR increases model performance

Save array of pixels index where PCC increased

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
    twsamodel_dir = path + "/model_predictions/" + model_name + "/newLR_climatic_regions/" 
    out_dir = path + "/model_fits/" + model_name + "/lowPCC_train/"   ### Save list of pixels here
    
    isExist = os.path.exists(out_dir)
    if not isExist:
       os.makedirs(out_dir)
    
    # Folder of list of pixels w/ low PCC
    listID_dir = path + "/model_fits/" + model_name + "/lowPCC_train/"    
    
    
    for clima_num in clima_class:   
              
        clima_name = clima_regs[clima_num - 1]
        print("Load data for", clima_name + " region")
        
        ### Load array w/ old PCC values for comparison and list of pixels
        oldPCC = np.load(listID_dir + "lowPCC_pixels_" + model_name + "_" + clima_name + ".npy")
        id_refitPIX = oldPCC[:, 0] # pixeld IDs on first column
        
        ### Load new predictions    
        twsa = np.load(twsamodel_dir + model_name + "_predictions_" + clima_name + ".npy")
        
        n_pixels = id_refitPIX.shape[0]
        
        newPCC = np.zeros((n_pixels, 6))
       
        for p in range(n_pixels):   
            
            pidx = int(id_refitPIX[p])
            newPCC[p, 0] = pidx

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
            
            newPCC[p, 1] = pcc(x_pix[mask_train], y_pix[mask_train])
            newPCC[p, 2] = pcc(x_pix[mask_val], y_pix[mask_val])
            newPCC[p, 3] = pcc(x_pix[mask_trainVal], y_pix[mask_trainVal])
            newPCC[p, 4] = pcc(x_pix[mask_test], y_pix[mask_test])
            newPCC[p, 5] = pcc(x_pix[mask_all], y_pix[mask_all])
            
        imp_mask = newPCC[:, 3] > oldPCC[:, 3]
        
        if imp_mask.sum() > 0:
            
            print("PCC has improved in " + str(imp_mask.sum()) + " grid cells! \n")
            
            newLR_pixel = newPCC[imp_mask, 0]
            
            np.save(out_dir + "newLR_pixels_" + model_name + "_" + clima_name, newLR_pixel)
            
         
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
    
    




