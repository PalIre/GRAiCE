# GR*Ai*CE dataset - Input data description

An array of input data, including both features and the target variable, is provided for each of the 28 Köppen-Geiger climatic regions used in this study. Each .npy file contains monthly data from 1982 to 2021 (a total of 480 time steps) for grid cells within a given climatic region.

Specifically, for each grid cell (identified by the *pixel_ID* column) the array includes the following monthly data:

- pixel ID (*pixel_ID*);
- latitude (*lat*) [degrees North];
- longitude (*lon*) [degrees East];
- year (*year*);
- month (*month*);
- monthly precipitation (*pcp*) [m];
- monthly air temperature (*tmp_air*) [K];
- monthly snow water equivalent (*snow*) [m eqH<sub>2</sub>O];
- monthly solar radiation (*rad*) [J m<sup>-2</sup>];
- monthly relative humidity (*rh*) [%];
- monthly solar induced fluorescence (*lcsif*) [mW m<sup>-2</sup> nm<sup>-1</sup> sr<sup>-1</sup>];
- Köppen-Geiger climatic class (*clima*) [1:30, excluding 10 and 13];
- GRACE/GRACE-FO Terrestrial Water Storage Anomalies (TWSA) (*twsa*) [cm eqH<sub>-2</sub>O].

Please note that the variables *pixel_ID*, *lat*, *lon*, and *clima* are static, meaning their values remain constant over time for each grid cell.
