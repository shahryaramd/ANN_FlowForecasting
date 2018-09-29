# ANN Based Reservoir Inflow Forecasting

This repository contains Matlab code to obtain Forecast Reservoir Inflow for lead tmes of 1-7 days using ANN
based forecasting model, setup for Pensacola dam's drainage basin.
A set of metrics are calculated at the end to assess the performance.


INPUTS 

Time series of:
1. GFS forecast (7 days lead) - Precipitation, Min/Max Temperature
2. Insitu data from GSOD - - Precipitation, Min/Max Temp, Windspeed
3. Observed Antecedent Streamflow
4. Observed Antecedent Soil Moisture

OUTPUTS

Time series of forecast streamflow for the training period (variable 'y')
and validation period (variable 'yV') for lead 1-7 days

NOTE: Matlab requires Neural Network / Deep Learning Toolbox 

Developed by: 

Shahryar Khalique Ahmad (skahmad@uw.edu)
Homepage: http://students.washington.edu/skahmad/
Department of Civil and Environmental Engineering, University of Washington
