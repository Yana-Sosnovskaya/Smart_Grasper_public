#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 13:15:03 2021

Last Modified May 22 2021

@author: Jack Kaplan
"""

import pandas as pd
import numpy as np

def position_data_analysis(csv_name):
    '''
    Takes the data from the csv and returns it in usable formats by trial and by distance. Also returns best fit information for a given distance. A trial is defined as one round of data gathering (i.e. one PWM input at one distance in one calibration loop in the arduino function position_calibration_setup.ino).

    Parameters
    ----------
    csv_name : A path object
        contains the name and path of the csv file with the relevant trial data.

    Returns
    -------
    data : list of dict
        a list of dictionaries containing all the information for each trial.
    scatter_data : dictionary
        a dictionary matching the measured distance between the jaws to the tuples containing the force-encoder pairs at that distance
    fit_data : dictionary
        a dictionary whose entries correspond to the x-values, slope, and y-intercept of a best fit line.

    '''
    
    df = pd.read_csv(csv_name)
    
    start_idx = 0
    
    trials = []
    indices = []
    
    while start_idx != -1:
        trial_idx, trial = trial_slice(df, idx0 = start_idx)
        trials.append(trial)
        indices.append(trial_idx)
        start_idx = trial_idx[1]
    
    data = []
    for trial in trials:
        trial_data = trial_analysis(trial)
        data.append(trial_data)
    
    scatter_data = position_scatter_data_generator(data)
    
    fit_data = position_linear_best_fit(scatter_data)
    
    return data, scatter_data, fit_data
    
def trial_slice(df, idx0 = 0):
    '''
    Parameters
    ----------
    df : Pandas DataFrame
        The pandas data structure containing all the data for the position calbration in a specific csv file.
    idx0 : TYPE, optional
        The first index of the trial. The default is 0.

    Returns
    -------
    trial_idx : tuple
        tuple whose first entry is the first index of the trial and whose second entry is the first index of the next trial.
    trial : Pandas DataFrame
        A pandas data structure of the trial as defined by the indexes in trial_idx.

    '''
    pwm = df['PWM Value'].values.tolist()
    
    for idx1 in range(idx0,len(pwm)):
        if pwm[idx1] != pwm[idx0]:
            break
    
    #get the index that corresponds to the beginning of the next trial. Tries to account for the fact that the last trial will not have an index corresponding to the next trial.
    try:
        idx2 = pwm.index(pwm[idx0], idx1)
    except ValueError:
        idx2 = -1
        
    trial_idx = (idx0, idx2)
    
    trial = df[idx0:idx2]
    
    return trial_idx, trial


def trial_analysis(trial):
    '''
    this function takes in the data for a particular trial and returns it in a better format that contains the measured and nominal lengths, and the average force and encoder sensor values at each PWM

    Parameters
    ----------
    trial : Pandas DataFrame
        The pandas data frame that contains the data corresponding to a particular trial.

    Returns
    -------
    trial_data : Dictionary
        dictionary that contains the Nominal Distance and Measured Distance for the trial and the average force and encoder sensor readings for each PWM input in the trial.

    '''
    pwm = trial['PWM Value'].values.tolist()
    
    indices = [0]
    idx = 0
    
    #Find the indices that correspond to the start of each PWM value
    for idxx in range(idx,len(pwm)):
        if pwm[idxx] != pwm[indices[idx]]:
            indices.append(idxx)
            idx += 1
    
    indices.append(-1)
    
    #Creates a dictionary to store the data for the trial
    pwm_data = {}
    try:
        trial_data = {'Nominal Distance': trial['Nominal Distance (mm)'][trial.index[0]], 'Measured Distance': trial['Measured Distance (mm)'][trial.index[0]], 'PWM': pwm_data}
        
    except KeyError:
        trial_data = {'Nominal Distance': trial['Nominal Distance (mm)'][trial.index[0]], 'Measured Distance': trial['Measured Length'][trial.index[0]], 'PWM': pwm_data}
    
    
    #Finds the average encoder and force sensor values for each PWM value in the trial. Then adds it to the 
    for idx in range(len(indices) - 1):
        subtrial = trial[indices[idx]:indices[idx+1]]
        
        pwm_value = subtrial['PWM Value'][subtrial.index[0]]
        
        force_values = subtrial['Force Reading (Counts)'].to_numpy()
        force_value = np.mean(force_values)
        
        encoder_values = subtrial['Encoder Value (Pulses)'].to_numpy()
        encoder_value = np.mean(encoder_values)
        
        pwm_data[pwm_value] = {'Encoder': encoder_value, 'Force': force_value}
        
    return trial_data
        

def position_scatter_data_generator(data, dist_val = 'Measured'):
    '''
    Organize all the encoder and force values by each distance (measured or nominal)

   Parameters
    ----------
    data : list of dictionaries
        a list of dictionaries consisting of the data from the CSV file organized by trial.
    dist_val : str, optional
        A string value that indicates if the measured or nominal value is considered to be the position for the purpose of plotting the data. The options are 'Measured' (or 'measured') or 'Nominal' (or 'nominal'). The default is 'Measured'.

    Raises
    ------
    Exception
        Exception raised if input dist_val does is not one of the acceptable options.

    Returns
    -------
    scatter_data : dict
        a dictionary matching the measured distance between the jaws to the tuples containing the force-encoder pairs at that distance
    '''
    
    if dist_val == 'Measured' or dist_val == 'measured':
        dist_measurement = 'Measured Distance'
    
    elif dist_val == 'Nominal' or dist_val == 'nominal':
        dist_measurement = 'Nominal Distance'
    
    else:
        raise Exception('The distance measurement input "dist_val" must be either "Nominal" or "Measured".')
    
    #Organize all the encoder and force values by each measured distance
    scatter_data = {}
    for trial in data:
        dist = trial[dist_measurement]
        for pwm in trial['PWM']:
            encoder = trial['PWM'][pwm]['Encoder']
            force = trial['PWM'][pwm]['Force']
            point = (force, encoder)  
            
            if dist in scatter_data.keys():
                scatter_data[dist].append(point)
    
            else:
                scatter_data[dist] = [point]
    
    return scatter_data

def position_linear_best_fit(scatter_data):
    '''
    Finds the slope, y-intercept, and r^2 values of the best fit line at each distance.

    Parameters
    ----------
    scatter_data : dict
        a dictionary matching the measured distance between the jaws to the tuples containing the force-encoder pairs at that distance.

    Returns
    -------
    fit_data : dict
        a dictionary whose entries correspond to the x-values, slope, and y-intercept of a best fit line.

    '''
    
    fit_data = {}
    
    for distance in scatter_data:
        x_values = []
        y_values = []
        for point in scatter_data[distance]:
            x_values.append(point[0])
            y_values.append(point[1])
        
        x_values = np.array(x_values)
        y_values = np.array(y_values)
        
        slope, intercept = np.polyfit(x_values, y_values, 1)
        correlation_matrix = np.corrcoef(x_values, y_values)
        correlation_xy = correlation_matrix[0,1]
        r2 = correlation_xy**2
        
        fit_data[distance] = {'x-values' : x_values, 'slope' : slope, 'y-intercept' : intercept, 'r squared' : r2}
        
    
    return fit_data

# # Test Code
# from pathlib import Path
# csv_name = Path.cwd().joinpath(*['Data Files', 'Position_Calibration_04_28_2021_0.csv'])
# trial_data, scatter_data, best_fit_data = position_data_analysis(csv_name)


