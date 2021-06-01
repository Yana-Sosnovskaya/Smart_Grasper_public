#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 19 20:57:15 2021

@author: jack
"""

import numpy as np

def fit_error(fit_values, measured_values):
    '''
    Plots a histogram of the error between the measured positions and the corresponding positions calculated from the surface fit.
    
    Parameters
    ----------
    fit_values : TYPE
        a numpy array consisting of the calculated position values.
    measured_values : numpy array
        a numpy array consisting of the measured position values.

    Returns
    -------
    error: numpy array
        A 1D array consisting of the difference between the measured values and the calculated values

    '''
    
    error = fit_values - measured_values
    
    return error 

def mean_absolute_deviation(fit_values, data_values):
    '''
    Finds the mean absolute deviation between the measured and calculated values.

    Parameters
    ----------
    fit_values : numpy array
        An 1D numpy array of length N that consists of the z-values that come from the surface fit function. Note that the z-values in fit_values correspond to the z-values in data_values (i.e. they have the same x and y values).
    data_values : numpy array
        A 1D numpy array of length N that consists of the measured z-values. Note that the z-values in fit_values correspond to the z-values in data_values (i.e. they have the same x and y values).

    Returns
    -------
    mad : float
        The calculated mean absolute deviation.

    '''
    
    #mean absolute deviation (mad)
    mad = np.mean(np.abs(data_values-fit_values))
    return mad

def root_mean_square_error(fit_values, data_values):
    '''
    Finds the root mean square error between the measured and calculated values.

    Parameters
    ----------
    fit_values : numpy array
        An 1D numpy array of length N that consists of the z-values that come from the surface fit function. Note that the z-values in fit_values correspond to the z-values in data_values (i.e. they have the same x and y values).
    data_values : numpy array
        A 1D numpy array of length N that consists of the measured z-values. Note that the z-values in fit_values correspond to the z-values in data_values (i.e. they have the same x and y values).

    Returns
    -------
    rmse : float
        the calculated root mean square error.

    '''
    
    diff_sq = (data_values - fit_values)**2
    rmse = np.sqrt(np.mean(diff_sq))
    
    return rmse