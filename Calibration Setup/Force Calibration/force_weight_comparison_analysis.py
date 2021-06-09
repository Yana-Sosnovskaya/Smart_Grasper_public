#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 10 12:59:17 2021

Last Modified June 7, 2021

@author: jack

Processes the data necessary for relating the external load cell readings to usable force units
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def force_weight_trials_data(csv_name):
    '''
    

    Parameters
    ----------
    csv_name : string
        the name of the CSV file (including it's path) from which data is going to be extracted.

    Returns
    -------
    trials : list of tuples
        a list of 2D tuples that correspond to data points. The first value in each tuple is the applied force during calibration measured in grams. The second value in each tuple is the average reading from the external force sensor that corresponds to that applied weight.

    '''
    
    df = pd.read_csv(csv_name)
    trials_df = trial_slice(df)
    
    trials = []
    for trial_df in trials_df:
        trial_data = trial_analysis(trial_df)
        trials.append(trial_data)
    
    return trials
    
def trial_slice(df):
    '''
    

    Parameters
    ----------
    df : pandas dataframe
        a pandas dataframe consisting of all the data generated in a particular calibration session. 

    Returns
    -------
    trials : list
        a list of dataframes with each element in the list corresponding to a different trial from the calibration session.

    '''
    trial_index = df['Trial'].values
    
    
    # This formulation accounts for issues when there are missing trial numbers
    trial_idx = []
    for num in trial_index:
        if num not in trial_idx:
            trial_idx.append(num)
    
    trials = []
    
    for trial in trial_idx:
        trial_bool = np.array(trial_index == trial)
        trials.append(df[trial_bool])
    
    return trials

def trial_analysis(trial_df):
    '''
    Records the pairing of the mass (in grams) applied to the external load cell and the corresponding load cell reading (in counts).

    Parameters
    ----------
    trial_df : pandas dataframe
        DESCRIPTION.

    Returns
    -------
    A tuple whose first entry is mass and whose second entry is load_cell
    
    mass : float
        The mass applied to the external load cell measured in grams.
    load_cell : float
        The load cell reading in counts corresponding to the applied mass.

    '''
    
    mass = np.mean(trial_df['Mass (g)'].values)
    load_cell = np.mean(trial_df['Load Cell Reading'].values)
    
    return (mass, load_cell)



def force_weight_plotter(trials, return_best_fit = True):
    '''
    
    Plots the curve mapping the external force sensor readings to their corresponding values in newtons. If desired, returns the slope and y-intercept of the line.

    Parameters
    ----------
    trials : list of tuples
        a list of 2D tuples that correspond to data points. The first value in each tuple is the applied force during calibration measured in grams. The second value in each tuple is the average reading from the external force sensor that corresponds to that applied weight.
    return_best_fit : bool, optional
        Do you want to return the best fit parameters (slope and the y-intercept) . The default is True.

    Returns
    -------
    m : float
        the slope of the fit.
    b : float
        the y-intercept of the fit.

    '''
    
    lc_vals, mass_vals = extract_fw_data(trials)
    
    #convert grams to newtons
    newtons = gram_2_newton(mass_vals)
    
    fig1 = plt.figure()
    ax = fig1.add_subplot()
    
    ax.scatter(lc_vals, newtons)
    ax.set_ylabel('Force (N)')
    ax.set_xlabel('External Force Sensor Reading (DV)')
    
    if return_best_fit:
        m,b = np.polyfit(lc_vals, newtons, 1)
        correlation_matrix = np.corrcoef(lc_vals, newtons)
        correlation_xy = correlation_matrix[0,1]
        r2 = correlation_xy**2
        legend_label = 'Best Fit Line: $R^{2}$ = ' + str(np.round(r2, 3))
        ax.plot(lc_vals, m * lc_vals + b, label = legend_label)
        plt.legend()
        plt.show()
        
        return m, b
    
    else:
        plt.legend()
        plt.show()
        
        
def extract_fw_data(trials):
    '''
    

    Parameters
    ----------
    trials : list of tuples
        a list of 2D tuples that correspond to data points. The first value in each tuple is the applied force during calibration measured in grams. The second value in each tuple is the average reading from the external force sensor that corresponds to that applied weight.

    Returns
    -------
    dv : numpy array
        a numpy array consisting of the readings from the external load cell. The nth entry in "dv" is the force reading that corresponds to the weight in the nth entry in "grams".
    grams : numpy array
        a numpy array consisting of the mass hanging from the external load cell measured in grams. The nth entry in "grams" is the weight that led to the force reading in the nth entry in "dv".

    '''
    
    dv = np.zeros(len(trials))
    grams = np.zeros(len(trials))
    
    for idx in range(len(trials)):
        dv[idx] = trials[idx][1]
        grams[idx] = trials[idx][0]
    
    return dv, grams

def gram_2_newton(grams):
    '''
    Converts grams of force to newtons of force.
    
    Parameters
    ----------
    grams : numpy array
        a numpy array consisting of force measured in grams.

    Returns
    -------
    numpy array
        a numpy array consisting of the values in "grams" converted to the corresponding values in newtons.

    '''
    return grams*9.81/1000



# # Test Code

# from pathlib import Path
# import force_error_functions as ferr

# csv_files = ['Force_Weight_Comparison_04_30_2021_0.csv']

# trials = []
# for filename in csv_files:
#     csv_name = Path.cwd().joinpath(*['Data Files', filename])
#     t_data = force_weight_trials_data(csv_name)
#     trials += t_data

# counts, grams = extract_fw_data(trials)
    
# slope, intercept = force_weight_plotter(trials) #in newtons

# meas_newtons = gram_2_newton(grams)
# calc_newtons = slope*counts + intercept
# rmse = ferr.root_mean_square_error(calc_newtons, meas_newtons)
# mad = ferr.mean_absolute_deviation(calc_newtons, meas_newtons)

# print('Root Mean Square Error of Fit: ' + str(rmse))
# print('Mean Absolute Deviation: ' + str(mad))
# print('Slope: ' + str(slope))
# print('Y-intercept: ' + str(intercept))

# err = ferr.fit_error(calc_newtons, meas_newtons)
# plt.hist(err)
