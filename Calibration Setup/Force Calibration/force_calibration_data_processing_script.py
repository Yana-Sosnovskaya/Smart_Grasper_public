#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 10:53:20 2021

@author: jack
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

def get_trial_data(csv_name):
    '''
    
    Takes the data from a .csv file and transforms it into a more useful format
    
    Parameters
    ----------
    csv_name : str
        A string file corresponding to the csv name and file path.

    Returns
    -------
    trials_data : list
        A list that contains dictionaries that contain all the informaiton for a particular trial (see trial_analysis for more information) 

    '''
    df = pd.read_csv(csv_name)
    trials_df = trial_slice(df)
    
    # Create a list of dictionaries containing all the information for each trial
    trials_data = []
    for trial_df in trials_df:
        trials_data.append(trial_analysis(trial_df)) 
    
    return trials_data

def trial_slice(df):
    '''
    
    Slices the larger pandas dataframe of data into smaller dataframes of one trial each.
    
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
    For a given trial, find the mean encoder, grasper, and external force values as well as the corresponding PWM value and jaw distance

    Parameters
    ----------
    trial_df : pandas dataframe
        A data struture consisting of all the information for one trial. One element of the list trials_df in force_data_analysis().

    Returns
    -------
    trial_data : dict
        a dictionary consisting of the PWM value, measured distance in mm, mean encoder value in counts, mean grasper force reading in counts, and mean external force reading in counts for a given trial. Does not return jaw distance if there is no jaw distance in input

    '''
    
    # clean_df = trial_clean(trial_df)
    
    #find the mean encoder, grasper internal force sensor, and external force sensor values for this trial
    encoder_value = np.mean(trial_df['Encoder Value (Pulses)'].values)
    grasper_force= np.mean(trial_df['Grasper Force Sensor (Counts)'].values)
    external_force = np.mean(trial_df['External Force Sensor (Counts)'].values)
    
    pwm_value = trial_df['PWM'].values[0] #PWM value for this trial
    
    try:
        jaw_dist = trial_df['Jaw Distance (mm)'].values[0]
        
        trial_data = {'PWM': pwm_value, 'Jaw Distance': jaw_dist, 'Encoder': encoder_value, 'Grasper Force': grasper_force, 'External Force': external_force}
    
    except KeyError:
        trial_data = {'PWM': pwm_value, 'Encoder': encoder_value, 'Grasper Force': grasper_force, 'External Force': external_force}
    
    return trial_data
    

def trial_clean(input_df):
    '''
    check to see if there is data that includes readings where the grasper is not physically grasping the sensor. Clean the data if necessary. Does this by ensuring that the internal force sensor clears some threshold min_threshold and that the internal sensor is within 2 standard deviations of the median for that trial

    Parameters
    ----------
    input_df : pandas data frame
        the raw panda data for a particular trial.

    Returns
    -------
    clean_df : pandas data frame
        a clean version of the input_df that has been filtered according to the criteria above.

    '''
    
    min_threshold = 100
    grasper_sensor = input_df['Grasper Force Sensor (Counts)'].values
    grasper_median = np.median(grasper_sensor)
    grasper_std = np.std(grasper_sensor)
    grasper_bool1 = (grasper_sensor > min_threshold)
    grasper_bool2 = (grasper_sensor > grasper_median - 2*grasper_std)
    grasper_bool = np.logical_and(grasper_bool1, grasper_bool2)
    
    clean_df = input_df[grasper_bool]
    return clean_df

def sort_by_distance(trials_data):
    '''
    Takes in all the trials and groups them by measured jaw distance. Ignores readings for which no distance was recorded

    Parameters
    ----------
    trials_data : list of dictionaries
        A list with each entry a dictionary consisting of the PWM value, measured distance in mm, mean encoder value in counts, mean grasper force reading in counts, and mean external force reading in counts for a given trial.

    Returns
    -------
    dist_data : dict
        A dictionary whose keys are the measured jaw distances in mm and whose values are lists of dictionaries containing the PWM value, mean encoder value in counts, mean grasper force reading in counts, and mean external force reading in counts for a given trial.

    '''
    dist_data = {}
    for trial in trials_data:   
        try:
            jaw_dist = trial['Jaw Distance']
            
            temp_dict = deepcopy(trial)
            temp_dict.pop('Jaw Distance')
            
            if jaw_dist in dist_data.keys():
                dist_data[jaw_dist].append(temp_dict)
            
            else:
                dist_data[jaw_dist] = [temp_dict]
    
        except KeyError:
            pass
        
    return dist_data

def force_data_plotter_3D(trials_data, x = 'Grasper Force', y = 'Encoder', z = 'External Force'):
    '''
    

    Parameters
    ----------
    trials_data : list of dict
        A list of dictionaries. Each dictonary conrresponds to one trial and contains the keys PWM, Jaw Distance, Encoder, Grasper Force, and External Force..
    x : str, optional
        The value that will become the x-value of the points graphed. CCorresponds to a key in the dictionaries in trials_data. The default is 'Grasper Force'.
    y : str, optional
        The value that will become the y-value of the points graphed. Corresponds to a key in the dictionaries in trials_data. The default is 'Encoder'.
    z : str, optional
        The value that will become the z-value of the points graphed. Corresponds to a key in the dictionaries in trials_data. The default is 'External Force'.

    Returns
    -------
    None.

    '''
    
    xvals = []
    yvals = []
    zvals = []
    
    for trial in trials_data:
        xvals.append(trial[x])
        yvals.append(trial[y])
        zvals.append(trial[z])
        
        
    
    xvals = np.asarray(xvals)
    yvals = np.asarray(yvals)
    zvals = np.asarray(zvals)
    
    fig3d = plt.figure(constrained_layout = True)
    ax = fig3d.add_subplot(projection='3d')
    
    ax.scatter(xvals, yvals, zvals)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useOffset= 1000, useMathText=True))
    ax.set_zlabel(z)
    ax.view_init(35, 15)
    plt.show()
    

def force_data_plotter_2D_superimpose(trials_data, x = 'Grasper Force', y = 'External Force', plot_legend = False):
    '''
    Plots the force data for a given jaw distance

    Parameters
    ----------
    trials_data : list
        A list of dictionaries. Each dictonary conrresponds to one trial and contains the keys PWM, Jaw Distance, Encoder, Grasper Force, and External Force.
    jaw_dist : float or int
        A value corresponding to the measured jaw distance
    x : str, optional
        The data that will become the x-value of the points graphed. The default is 'Grasper Force'.
    y : str, optional
        The data that will become the y-value of the points graphed. The default is 'External Force'.
    return_best_fit : bool, optional
        If true, returns the parameters necessary to generate a best fit line. The default is True.

    Returns
    -------
    xvals : numpy array
        an array consisting of the x-values used to generate the best fit line.
    m : float
        the slope of the best fit line.
    b : float
        the y intercept of the best fit line.

    '''
    
    fig = plt.figure(constrained_layout = True)
    # fig = plt.figure(constrained_layout = False)
    ax = fig.add_subplot()
    
    if x == 'Grasper Force':
        ax.set_xlabel('Grasper Force Sensor (counts)')
        
    else:
        ax.set_xlabel(x)
    
    if y == 'External Force':
        scaler = 10000
        ax.set_ylabel('External Force (x' + str(scaler) + ' counts)')
    
    else:
        ax.set_ylabel(y)
    
    dist_data = sort_by_distance(trials_data)
    
    for dist in dist_data:
        xvals = []
        yvals = []
        for trial in dist_data[dist]:
            xvals.append(trial[x])
            yvals.append(trial[y])
    
        xvals = np.array(xvals)
        yvals = np.array(yvals)
        
        if y == 'External Force':
            yvals = yvals/scaler
        
        
        plt.scatter(xvals, yvals
                    # , label = 'Jaw Distance: ' + str(dist)
                    )
       
        
        m,b = np.polyfit(xvals, yvals,1)
        correlation_matrix = np.corrcoef(xvals, yvals)
        correlation_xy = correlation_matrix[0,1]
        r2 = correlation_xy**2
        legend_label = 'Jaw Dist '+ str(dist) + 'mm; Best Fit Line: $R^{2}$ = ' + str(np.round(r2, 3))
        plt.plot(xvals, m * xvals + b, label = legend_label)
        # plt.title('Jaw Distance: ' + str(dist))
    
    if plot_legend:
        plt.legend()
        
def linear_best_fit(xvals, yvals):
    '''
    Gets the best fit line corresponding to the input x-values and y-values

    Parameters
    ----------
    xvals : numpy array
        a 1D numpy array of the x-values that correspond to the y-values in yvals.
    yvals : numpy array
        a 1D numpy array of the y-values that correspond to the x-values in xvals.

    Returns
    -------
    param : dict
        A dictionary containing the slope, y-intercept, and r^2 value of the best fit line. Key values are 'slope', 'y-intercept', and 'r squared', respectively. 

    '''
    
    if type(xvals) != np.ndarray:
        xvals = np.array(xvals)
    
    if type(yvals) != np.ndarray:
        yvals = np.array(yvals)
    
    
    slope, intercept = np.polyfit(xvals, yvals, 1)
    
    correlation_matrix = np.corrcoef(xvals, yvals)
    correlation_xy = correlation_matrix[0,1]
    r2 = correlation_xy**2
    
    param = {'slope': slope, 'y-intercept': intercept, 'r squared': r2}
    
    return param

def force_data_plotter_2D(trials_data, jaw_dist, x = 'Grasper Force', y = 'External Force', return_best_fit = True):
    '''
    Plots the force data for a given jaw distance

    Parameters
    ----------
    trials_data : list
        A list of dictionaries. Each dictonary conrresponds to one trial and contains the keys PWM, Encoder, Grasper Force, and External Force.
    jaw_dist : float or int
        A value corresponding to the measured jaw distance
    x : str, optional
        The data that will become the x-value of the points graphed. The default is 'Grasper Force'.
    y : str, optional
        The data that will become the y-value of the points graphed. The default is 'External Force'.
    return_best_fit : bool, optional
        If true, returns the parameters necessary to generate a best fit line. The default is True.

    Returns
    -------
    xvals : numpy array
        an array consisting of the x-values used to generate the best fit line.
    m : float
        the slope of the best fit line.
    b : float
        the y intercept of the best fit line.

    '''
    
    xvals = []
    yvals = []
    for trial in trials_data:
        xvals.append(trial[x])
        yvals.append(trial[y])
    
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    
    fig = plt.figure()
    ax = fig.add_subplot()
    
    ax.scatter(xvals, yvals)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    
    if jaw_dist == None:
        jaw_dist = 'No Value Provided'
    
    if return_best_fit:
        m,b = np.polyfit(xvals, yvals,1)
        correlation_matrix = np.corrcoef(xvals, yvals)
        correlation_xy = correlation_matrix[0,1]
        r2 = correlation_xy**2
        legend_label = 'Best Fit Line: $R^{2}$ = ' + str(np.round(r2, 3))
        ax.plot(xvals, m * xvals + b, label = legend_label)
        plt.title('Jaw Distance: ' + str(jaw_dist))
        plt.legend()
        
        return xvals, m, b
    
    else:
        plt.title('Jaw Distance: ' + str(jaw_dist))
        plt.legend()
