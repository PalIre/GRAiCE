# GR*Ai*CE dataset - Input data description

An array of input data, including both features and the target variable, is available for each of the 28 Köppen-Geiger climatic regions used in this study. Each npy file contains monthly data from 1982 to 2021 (total number of 480 time steps) for grid cells of a given climatic region.  
In particular, for each grid cell (identified in the column *pixel_ID*) the array shows the monthly values of the following data:

- pixel ID (*pixel_ID*);
- latitude (*lat*) [degree North];
- longitude (*lon*) [degree East];
- year (*year*);
- month (*month*);
- monthly precipitation (*pcp*) [m];
- monthly air temperature (*tmp_air*) [K];
- monthly snow water equivalent (*snow*) [m eqH<sub>2</sub>O];
- monthly solar radiation (*rad*) [J m<sup>-2</sup>];
- monthly relative humidity (*rh*) [%];
- monthly solar induced fluorescence (*lcsif*) [mW m<sup>-2</sup> nm<sup>-1</sup> sr<sup>-1</sup>];
- Köppen-Geiger climatic class (*clima*) [1:30, except 10 and 13];
- GRACE/GRACE-FO Terrestrial Water Storage Anomalies (TWSA) (*twsa*) [cm eqH<sub>-2</sub>O].

Please be aware that the variables *pixel_ID*, *lat*, *lon*, and *clima* are static, i.e., their value is constant in time for a given grid cell.
