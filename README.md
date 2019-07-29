# ANN Based Reservoir Inflow Forecasting

This repository contains Matlab code to obtain Forecast Reservoir Inflow for lead tmes of 1-7 days using ANN
based forecasting model, setup for any drainage basin. Currently, the code is set up to generate forecast inflow for Lost Creek Dam.
A set of metrics are calculated at the end to assess the performance.


## Input 

Time series of:
1. GFS forecast (7 days lead) - Precipitation, Min/Max Temperature (in Predictors/<basin_name> folder)
2. Nowcast Satellite-based Precipitation from CHIRPS (in Predictors/CHIRPS_Precip folder)
3. Observed Antecedent Streamflow (.xls format in Predictors/ folder)

## Outputs

Time series of forecast streamflow for the training period (variable 'y')
and validation period (variable 'yV') for lead 1-7 days

NOTE: Matlab requires Neural Network / Deep Learning Toolbox 

## Citation

Please cite this work as:

Ahmad, S.K. and Hossain, F., 2019. A generic data-driven technique for forecasting of reservoir inflow: Application for hydropower maximization. Environmental Modelling & Software, 119, pp.147-165. https://doi.org/10.1016/j.envsoft.2019.06.008

## Contact

Shahryar Khalique Ahmad (https://students.washington.edu/skahmad/ | skahmad@uw.edu)

Department of Civil and Environmental Engineering, University of Washington
