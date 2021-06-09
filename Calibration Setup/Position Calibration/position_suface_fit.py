#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 30 17:32:35 2021

Last modified May 31, 2021

@author: jack
"""

import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import position_data_processing_script as dps
from scipy import optimize as opt
import os


def surf_fit_linear(X, c0, c1, c2):
    '''
    A function that can be used to generate a planar surface

    Parameters
    ----------
    X : numpy arrray
        A 2xN numpy array that are the independent variables that are the inputs to the function creating a 2D surface fit.
    c0 : float
        a constant coefficeint.
    c1 : float
        a constant coefficeint.
    c2 : float
        a constant coefficeint.

    Returns
    -------
    values : numpy array
        a 1D numpy array of length N consisting of the values that are outputed from the function with the given inputs and parameters.

    '''
    Y = np.vstack((np.ones(X.shape[1]),X))
    coeff = np.array([c0, c1, c2])
    values = np.dot(coeff, Y)
    return values

def surf_fit_quad(X, c0, c1, c2, c3, c4, c5):
    '''
     A function that can be used to generate a quadratic surface

    Parameters
    ----------
    X : numpy arrray
        A 2xN numpy array that are the independent variables that are the inputs to the function creating a 2D surface fit.
    c0 : float
        a constant coefficeint.
    c1 : float
        a constant coefficeint.
    c2 : float
        a constant coefficeint.
    c3 : float
        a constant coefficeint.
    c4 : float
        a constant coefficeint.
    c5 : float
        a constant coefficeint.

    Returns
    -------
    values : numpy array
        a 1D numpy array of length N consisting of the values that are outputed from the function with the given inputs and parameters.

    '''
    
    XY = X[0,:] * X[1,:]
    
    Y = np.vstack((np.ones(X.shape[1]), X, XY, X*X))
    coeff = np.array([c0, c1, c2, c3, c4, c5])
    values = np.dot(coeff, Y)
    return values

def surf_mad(fit_values, data_values):
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


def surf_rms(fit_values, data_values):
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

def fit_error(fit_values, measured_values, fit_type, plot = True):
    '''
    Plots a histogram of the error between the measured positions and the corresponding positions calculated from the surface fit.
    
    Parameters
    ----------
    fit_values : TYPE
        a numpy array consisting of the calculated position values.
    measured_values : numpy array
        a numpy array consisting of the measured position values.
    fit_type : string
        a string describing the fit type.
    plot : boolean, optional
        Do you want to plot a histogram. The default is True.

    Returns
    -------
    error: numpy array
        A 1D array consisting of the difference between the measured values and the calculated values

    '''
    
    error = fit_values - measured_values
    
    if plot:
                
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(error, bins = 20, weights = np.ones(len(fit_values))/len(fit_values))
        # ax.hist(error, bins = 20)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        
        ax.set_xlabel('Error (mm)')
        ax.set_ylabel('Frequency')
        
        max_err = np.max(np.abs(error))
        edge = int(np.ceil(max_err))
        ax.set_xlim(left = -1*edge, right = edge)
        
                
        plt.show()
    
    return error

def get_fit_fun(fit_type = 'linear'):
    '''
    Get the surface function based on the surface fit type desired.

    Parameters
    ----------
    fit_type : string, optional
         The surface fit type desired. Value must be 'linear' (planar fit) or 'quad' (quadratic surface fit). The default is 'linear'.

    Raises
    ------
    Exception
        Raises exception if the fit_type is not one of the acceptable inputs.

    Returns
    -------
    fit_fun : function
        The surface function corresponding to the surface fit type desired.

    '''
    
    if fit_type == 'linear':
        fit_fun = surf_fit_linear
    
    elif fit_type == 'quad':
        fit_fun = surf_fit_quad
    
    else:
        raise Exception('"fit_type" input must be "linear" or "quad"')
    
    return fit_fun

def surface_fitting(measured, fit = 'linear', error = True):
    '''
    
    Creates a surface fit that best fits the measured 3D data for a given fit type

    Parameters
    ----------
    measured : numpy array
        A 3XN numpy array that consists of the measured force, encoder, and position values from the position calibration.
    fit : string, optional
        DESCRIPTION. The default is 'linear'.
    error : TYPE, optional
        Determines if you want to return the mean absolute deviation and the root mean square error of the fit function. The default is True.

    Raises
    ------
    Exception
        Makes sure that the fit is either a linear or quadratic surface fit.

    Returns
    -------
    param : numpy array
        The constant parameters associated with the best fit.
    param_cov : numpy array
        The covariance matrix associated with the fit.
    rmse : float
        The root mean square error of the fit. Only returns if the error input is True.
    mad : float
        The mean absolute deviation of the fit. Only returns if the error input is True.
    '''
    
    data = measured[0:2, :]
    values = measured[2,:]
    
    #Get the surface function based on the fit type
    fit_fun = get_fit_fun(fit)
    
    #Find the optimal surface fit for that fit function
    param, param_cov = opt.curve_fit(fit_fun, data, values)
    
    #get the values of the optimized surface fit that correspond to the measured data input values 
    fit_values = fit_fun(data, *param)
    
    fit3d = np.vstack((data, fit_values))
    measured3d = np.vstack((data, values))
    
    scatter_representation_plot(measured3d, fit3d, fit)
    
    if error:
        rmse = surf_rms(fit_values, values)
        mad = surf_mad(fit_values, values)
        return param, param_cov, rmse, mad
    
    else:
        return param, param_cov

def scatter_representation_plot(measured_data, fit_data, fit_type):
    '''
    Creates a scatter plot of the measured data and the data generated using the best fit function.

    Parameters
    ----------
    measured_data : numpy array
        A 3XN numpy array that consists of the measured force, encoder, and position values from the position calibration.
    fit_data : numpy array
        A 3XN numpy array that consists of the measured force and encoder values and the calculated fit values that correspond to them.
    fit_type : string
        A string that tells you the type of surface fit. Must be either 'linear' (linear/planar fit) or 'quad' (quadratic fit).

    Returns
    -------
    None.

    '''
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    
    ax.scatter(measured_data[0,:], measured_data[1,:], measured_data[2,:], label = 'Measured Data')
    ax.scatter(fit_data[0,:], fit_data[1,:], fit_data[2,:], label = 'Fit Data')
    ax.set_xlabel('Grasper Force Sensor (DV)')
    ax.set_ylabel('Encoder Counts')
    ax.set_zlabel('Position (mm)')
    ax.legend()
    
    if fit_type == 'linear':
        plt.title('Linear Fit')
    
    elif fit_type == 'quad':
        plt.title('Quad Fit')
        
    else:
        plt.title('Unknown Fit')
    
    plt.show()

def make_meshgrid_for_plot(data, fun, coeff, lin_len = 50):
    '''
    Makes a meshgrid for the data for use in plotting functions

    Parameters
    ----------
    data : numpy array
        A 2XN numpy array that consists of measured x and y value pairs.
    fun : function
        the function that takes in an x and y value and returns a z value. Used in this case to generate a surface.
    coeff : numpy array
        the numpy array of coefficients that are used in the function "fun".
    lin_len : int, optional
        The number of values in each dimension of the meshgrid (used in linspace). The default is 50.

    Returns
    -------
    XX : numpy array
        A 2D numpy array that contains the x-values at each point in the meshgrid.
    YY : numpy array
        A 2D numpy array that contains the y-values at each point in the meshgrid.
    ZZ : numpy array
        A 2D numpy array that contains the z-values at each point in the meshgrid.

    '''
    
    if type(lin_len) != int:
        raise Exception('Input "lin_len" must be an integer')
    
    
    x = data[0,:]
    y = data[1,:]
    
    
    xspan = np.linspace(min(x), max(x), num = lin_len)
    yspan = np.linspace(min(y), max(y), num = lin_len)
    
    X,Y = np.meshgrid(xspan, yspan, indexing = 'xy')
    
    inputs = np.array([np.ravel(X), np.ravel(Y)])
    
    fit_values = fun(inputs, *coeff)
    
    XX = inputs[0,:].reshape((lin_len, lin_len))
    YY = inputs[1,:].reshape((lin_len, lin_len))
    ZZ = fit_values.reshape((lin_len, lin_len))
    
    return XX, YY, ZZ
    
def make_surface(measured, fun, coeff):
    '''
    plots a surface using the measured values and the best fit equation found in "fun" with coefficients "coeff". Plots both the best fit surface and the measured values. Requires the function make_meshgrid_for_plot.
    
    Parameters
    ----------
     measured : numpy array
        A 3XN numpy array that consists of the measured force, encoder, and position values from the position calibration.
     fun : function
        the function that takes in an x and y value and returns a z value. Used in this case to generate a surface.
    coeff : numpy array
        the numpy array of coefficients that are used in the function "fun".

    Returns
    -------
    None.

    '''
    
    data = measured[0:2, :]
    values = measured[2,:]
    
    XX, YY, ZZ = make_meshgrid_for_plot(data, fun, coeff)
    
    force_mult = 100
    encoder_mult = 1000
    
    surf = plt.figure(constrained_layout = True)
    ax = surf.add_subplot(projection = '3d')
    # ax.plot_wireframe(XX,YY/1000,ZZ, label = 'Fit Surface', rstride = 5, cstride = 5,  alpha = 0.25)
    surface = ax.plot_surface(XX/force_mult,YY/encoder_mult,ZZ, label = 'Fit Surface', alpha = 0.4)
    
    #Hacky fix for a bug
    surface._facecolors2d = surface._facecolor3d
    surface._edgecolors2d = surface._edgecolor3d
    
    ax.scatter(data[0,:]/force_mult,data[1,:]/encoder_mult,values, label = 'Data', c = '#ff7f0e')
    
    ax.set_xlabel('Grasper Force Sensor (x ' + str(force_mult) + ' DV)')
    ax.set_ylabel('Encoder (x ' + str(encoder_mult) + ' counts)')
    ax.set_zlabel('Position (mm)') 
    # plt.title('Position Fit')
    
    ax.view_init(10,150)
    
    plt.legend()
    
    plt.show

def make_contour_plot(measured, fun, coeff):
    '''
    
    Creates a contour plot for the fit surface to the measured data. Also plots the x-y points from the measured data on the contour plot. 

    Parameters
    ----------
    easured : numpy array
        A 3XN numpy array that consists of the measured force, encoder, and position values from the position calibration.
     fun : function
        the function that takes in an x and y value and returns a z value. Used in this case to generate a surface.
    coeff : numpy array
        the numpy array of coefficients that are used in the function "fun".

    Returns
    -------
    None.

    '''
    
    data = measured[0:2, :]
    values = measured[2,:]
    
    X,Y,Z = make_meshgrid_for_plot(data, fun, coeff)
    
    # fig = plt.figure(constrained_layout = True, dpi = 600)
    fig = plt.figure()
    # fig = plt.figure(dpi = 600)
    ax = fig.add_subplot()
   
    contours = []
    for dist in values:
        if dist not in contours:
            contours.append(dist)
    
    CS = ax.contour(X, Y, Z, levels = sorted(contours))
    # CS = ax.contour(X, Y, Z, levels = range(5,20,2))
    ax.scatter(data[0,:], data[1,:], s = 5, c = 'black')
    
    ax.set_xlabel('Grasper Force Sensor (DV)')
    ax.set_ylabel('Encoder (counts)')
    ax.clabel(CS, CS.levels, inline=True, fmt=label_contour_mm, manual = False)
    
    plt.show()

def label_contour_mm(x):
    '''
    Formatter for the contour plt. Removes trailing zeros and adds mm.

    '''
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} mm"


def savefig_path(filename, loc = 'Plots'):
    '''

   Parameters
    ----------
    filename : string
        the filename of the desired file to save.
    loc : string, optional
        The folder in the current working directory in which to save the file. The default is 'Plots'.

    Returns
    -------
    p : Path
        the path and filename of the desired file

    '''
    
    directory = Path.cwd().joinpath(loc)
    
    if not os.path.isdir(directory):
        os.makedirs(directory)
    
    p = directory.joinpath(filename)
    return p

def extract_scatter_data(scatter_data):
    '''

    Parameters
    ----------
    scatter_data : dict
        a dictionary matching the measured distance between the jaws to the tuples containing the force-encoder pairs at that distance.

    Returns
    -------
    force : numpy array
        A numpy array that contains the measured grasper force readings.
    encoder : numpy array
        A numpy array that contains the measured encoder values.
    distance : numpy array
        A numpy array that contains the measured position values.

    '''
    
    #Extract the force, encoder, and measured distance data from scatter_data
    force = []
    encoder = []
    distance = []
    for dist in scatter_data:
        for point in scatter_data[dist]:
            force.append(point[0])
            encoder.append(point[1])
            distance.append(dist)
    
    
    
    return force, encoder, distance

#The names of the .csv files from which we are going to draw data
csv_files = [
    'Position_Calibration_05_22_2021_1.csv',
    'Position_Calibration_05_22_2021_2.csv'
    ]


#Get all the data from all the trials being assessed and put it together
trials = []
for filename in csv_files:
    #Find the path to the desired .csv file. Assumes that it is stored in the .\Data Files folder as described in the README
    csv_name = Path.cwd().joinpath(*['Data Files', filename])
    t_data, s_data, bf_data = dps.position_data_analysis(csv_name)
    trials += t_data


#Get the data in a usable form to generate a surface and plot
scatter_data = dps.position_scatter_data_generator(trials, dist_val='measured')

force, encoder, distance = extract_scatter_data(scatter_data)

data = np.array(np.array([force,encoder]))
value = np.array(distance)
measured = np.array([force, encoder, distance])

#Find the surface fit that best represents the data 
surf_fits = ['linear', 'quad']
# surf_fits = ['linear']
for surf_fit in surf_fits:
    popt, cov_opt, rmse, mae = surface_fitting(measured, fit = surf_fit)
    print('\n'+surf_fit)
    print('Coefficients: ' + str(popt))
    print('RMSE: ' + str(rmse))
    print('Mean Absolute Error: ' + str(mae))
    
    #Get the fitting function for the type of surface fit
    fit_fun = get_fit_fun(surf_fit)
    
    #Find the values on the fitted surface that correspond to the input values
    fit_values = fit_fun(data, *popt)
    
    err = fit_error(fit_values, value, surf_fit)
    print('Error Mean: ' + str(np.mean(err)))
    print('Error Standard Deviation: ' + str(np.std(err)))
    
    make_surface(measured, fit_fun, popt)
    make_contour_plot(measured, fit_fun, popt)

    