#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 23 18:06:15 2021

@author: jack

This code is the main code to get the force fitting functions relating the grasper strain gauge readings to the force applied at the jaw tip in Newtons.
"""

#Note, this code is very similar to position_surface_fit.py

import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import optimize as opt
import force_calibration_data_processing_script as fcps
import force_weight_comparison_analysis as fwps
import force_error_functions as ferr

    
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
    
    x = X[0,:]
    y = X[1,:]
    
    values = c0 + c1*x + c2*y 
    
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

def fit_error(fit_values, measured_values, plot = True):
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
        # fig = plt.figure()
        ax = fig.add_subplot()
        ax.hist(error, bins = 20, weights = np.ones(len(fit_values))/len(fit_values))
        # ax.hist(error, bins = 20)
        ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(xmax=1))
        
        ax.set_xlabel('Error (N)')
        ax.set_ylabel('Frequency')
        
        max_err = np.max(np.abs(error))
        edge = int(np.ceil(max_err))
        ax.set_xlim(left = -1*edge, right = edge)
        
        ax.set_aspect('auto')


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
        A 3XN numpy array that consists of the measured grasper force, encoder, and external force values from the position calibration.
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
        rmse = ferr.root_mean_square_error(fit_values, values)
        mad = ferr.mean_absolute_deviation(fit_values, values)
        return param, param_cov, rmse, mad
    
    else:
        return param, param_cov

def scatter_representation_plot(measured_data, fit_data, fit_type):
    '''
    Creates a scatter plot of the measured data and the data generated using the best fit function.

    Parameters
    ----------
    measured_data : numpy array
        A 3XN numpy array that consists of the measured force, encoder, and position values from the force calibration.
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
    ax.set_xlabel('Force Reading')
    ax.set_ylabel('Encoder Counts')
    ax.set_zlabel('Force (N)')
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

def force_data_plotter_3d_fixed(trials_data):
    '''
    A hard coded version of force_data_plotter_3d to get the exact plot that I want for the paper.

    Parameters
    ----------
    trials_data : list of dict
        A list of dictionaries. Each dictonary conrresponds to one trial and contains the keys PWM, Jaw Distance, Encoder, Grasper Force, and External Force..

    Returns
    -------
    None.

    '''
    
    
    fig3d = plt.figure(constrained_layout = False)
    ax = fig3d.add_subplot(projection='3d')
    ax.set_xlabel('Grasper Force Sensor (DV)')
    ax.set_ylabel('Encoder (x 1000 counts)')
    ax.set_zlabel('External Force (x 10000 DV)')
    ax.view_init(40,-45)
    
    d_trials = fcps.sort_by_distance(trials_data)
   
    
    for dist in d_trials:
        grasper = []
        encoder = []
        external = []
        for trial in d_trials[dist]:
            grasper.append(trial['Grasper Force'])
            encoder.append(trial['Encoder'])
            external.append(trial['External Force'])
        
        grasper = np.array(grasper)
        encoder = np.array(encoder)
        external = np.array(external)
        ax.scatter(grasper, encoder/1000, external/10000)
    
    plt.show()

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
    encoder_mult = 10000
    
    surf = plt.figure(constrained_layout = True)
    ax = surf.add_subplot(projection = '3d')
    # ax.plot_wireframe(XX,YY/1000,ZZ, label = 'Fit Surface', rstride = 5, cstride = 5,  alpha = 0.25)
    surface = ax.plot_surface(XX/force_mult,YY/encoder_mult,ZZ, label = 'Fit Surface', alpha = 0.4)
    
    #Hacky fix for a bug
    surface._facecolors2d = surface._facecolor3d
    surface._edgecolors2d = surface._edgecolor3d
    
    ax.scatter(data[0,:]/force_mult,data[1,:]/encoder_mult,values, label = 'Data', c = '#ff7f0e')
    

    
    # ax.plot_wireframe(data[0,:],data[1,:]/1000,values, label = 'Data', c = '#ff7f0e', rstride = 5, cstride = 5, alpha = 0.5)
    
    ax.set_xlabel('Grasper Force Sensor (x ' + str(force_mult) + ' DV)')
    ax.set_ylabel('Encoder (x ' + str(encoder_mult) + ' counts)')
    ax.set_zlabel('Force (N)') 
    # plt.title('Position Fit')
    
    ax.view_init(15,45)
    
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
    
    fig = plt.figure()
    # fig = plt.figure(figsize=(6, 3), dpi=600)
    ax = fig.add_subplot()
   
    contours = []
    for dist in values:
        if dist not in contours:
            contours.append(dist)
    
    # CS = ax.contour(X, Y, Z, levels = sorted(contours))
    ax.scatter(data[0,:], data[1,:], s = 5, c = 'black')
    CS = ax.contour(X, Y, Z)
    
    ax.set_xlabel('Grasper Force Sensor (DV)')
    ax.set_ylabel('Encoder (counts)')
    ax.clabel(CS, CS.levels, inline=True, fmt=label_contour_newtons, manual = False)
    
    plt.show()
   
def label_contour_newtons(x):
    '''
    Formatter for the contour plt. Removes trailing zeros and adds N

    '''
    s = f"{x:.1f}"
    if s.endswith("0"):
        s = f"{x:.0f}"
    return rf"{s} N"

def force_data_extraction(trials, x = 'Grasper Force', y = 'Encoder', z = 'External Force'):
    '''
    Extracts data from the trial information in a numpy array format
    
    Parameters
    ----------
    trials : list
        A list that contains dictionaries that contain all the informaiton for a particular trial. Each dictionary consists of the PWM value, measured distance in mm, mean encoder value in counts, mean grasper force reading in its raw form, and mean external force reading in its raw form for a given trial. Does not return jaw distance if there is no jaw distance in input
    x : string, optional
        the key whose data will be the x-value. The default is 'Grasper Force'.
    y : TYPE, optional
        the key whose data will be the y-value. The default is 'Encoder'.
    z : TYPE, optional
        the key whose data will be the z-value. The default is 'External Force'.

    Returns
    -------
    xvals : numpy array
        a numpy array of the x-values as determined by the input "x".
    yvals : numpy array
        a numpy array of the y-values as determined by the input "y".
    zvals : numpy array
        a numpy array of the z-values as determined by the input "z".

    '''
    
    xvals = []
    yvals = []
    zvals = []
    
    for trial in trials:
        xvals.append(trial[x])
        yvals.append(trial[y])
        zvals.append(trial[z])
    
    xvals = np.array(xvals)
    yvals = np.array(yvals)
    zvals = np.array(zvals)
    
    
    return xvals, yvals, zvals

def scale_external_newtons(dv):
    '''
    Takes an array of readings from the external force sensor and converts them to their equivalent value in Newtons.

    Parameters
    ----------
    dv : numpy array
        A numpy array of the external force reading in its raw discretized format.

    Returns
    -------
    newtons : numpy array
        A numpy array of the external force reading in Newtons.

    '''
    
    fwbf = bf_external_newtons()
    
    slope = fwbf['slope']
    intercept = fwbf['y-intercept']
    
    newtons = slope*dv + intercept
    
    return newtons

def bf_external_newtons():
    '''
    Uses the functions in externalforce_weight_comparison_analysis.py to generate a line that maps the external load cell value to its value in newtons. It then returns the parameters of that best fit line.

    Returns
    -------
    bf_param : dict
        A dictionary containing the slope, y-intercept, and r^2 value of the best fit line. Key values are 'slope', 'y-intercept', and 'r squared', respectively.

    '''
    
    csv_files = ['Force_Weight_Comparison_04_30_2021_4.csv']
    
    trials = []
    for filename in csv_files:
        csv_name = Path.cwd().joinpath(*['Data Files', filename])
        t_data = fwps.force_weight_trials_data(csv_name)
        trials += t_data
    
    dv, grams = fwps.extract_fw_data(trials)
    
    newtons = fwps.gram_2_newton(grams)
    
    bf_param = fcps.linear_best_fit(dv, newtons)
    
    return bf_param

def check_1D_fit(grasper, external):
    '''
    Evaluates the predictive power of a 1D curve fit linear regression mapping the grasper force sensor in discretized values. 

    Parameters
    ----------
    grasper : numpy array
        the external force sensor readings corresponding to the external force sensor readings from "external".
    external : numpy array
        the external force sensor readings corresponding to the grasper force sensor readings from "grasper".

    Returns
    -------
    None.

    '''

    
    calc_newton = scale_external_newtons(external) #convert the external value from its discretized value to newtonws
    
    # See how good a 1D linear regression fit is
    bf_force = fcps.linear_best_fit(grasper, calc_newton)
    
    m = bf_force['slope']
    b = bf_force['y-intercept']
    
    fit_newton = m*grasper + b
    rmse = ferr.root_mean_square_error(fit_newton, calc_newton)
    print('\nLinear Best Fit (Ignoring Encoder)')
    print('slope: ' + str(m))
    print('y-intercept: ' + str(b))
    print('RMSE: ' + str(rmse))
    lin_err = fit_error(fit_newton, calc_newton)
    print('Mean Error:' + str(np.mean(lin_err)))
    
    
    lfig = plt.figure()
    ax = lfig.add_subplot()
    ax.scatter(grasper, calc_newton, label = 'Data')
    ax.plot(grasper, fit_newton, c = 'orange', label = 'Best Fit Line: $R^{2}$ = ' + str(np.round(bf_force['r squared'],3)), linewidth = 2)
    ax.set_xlabel('Grasper Force Sensor (DV)')
    ax.set_ylabel('Force (N)')
    plt.legend()
    
    plt.show()

## Script to run determine the fit

#Data sources to draw from.
csv_files = [
    'Force_Calibration_05_05_2021_0.csv',
    'Force_Calibration_05_22_2021_3.csv',
    'Force_Calibration_05_22_2021_4.csv',
    'Force_Calibration_05_22_2021_5.csv',
    'Force_Calibration_05_27_2021_0.csv',
    'Force_Calibration_05_27_2021_1.csv',
    'Force_Calibration_05_27_2021_2.csv',
    'Force_Calibration_05_27_2021_3.csv', 
    'Force_Calibration_05_27_2021_4.csv'
    ]



trials = []

print('\nData Sources:')
for filename in csv_files:
    print(filename)
    csv_name = Path.cwd().joinpath(*['Data Files', filename])
    t_data = fcps.get_trial_data(csv_name)
    trials += t_data

# Plot the Raw Data Gathered
force_data_plotter_3d_fixed(trials)
fcps.force_data_plotter_2D_superimpose(trials, plot_legend = False)

grasper, encoder, external = force_data_extraction(trials)

check_1D_fit(grasper, external)

calc_newton = scale_external_newtons(external) #convert the external value from its discretized value to newtonws
measured = np.vstack((grasper, encoder, calc_newton))


#Find the surface fit that best represents the data 
surf_fits = ['linear', 'quad']
# surf_fits = ['linear']
print('\nSurface Fits:')
for surf_fit in surf_fits:
    popt, cov_opt, rmse, mae = surface_fitting(measured, fit = surf_fit)
    print(surf_fit)
    print('Coefficients: ' + str(popt))
    print('RMSE: ' + str(rmse))
    print('Mean Absolute Error: ' + str(mae))
    
    #Get the fitting function for the type of surface fit
    fit_fun = get_fit_fun(surf_fit)
    
    #Find the values on the fitted surface that correspond to the input values
    fit_values = fit_fun(measured[:2,:], *popt)
    
    err = fit_error(fit_values, measured[2,:])
    print('Error Mean: ' + str(np.mean(err)))
    print('Error Standard Deviation: ' + str(np.std(err)))
    print(' ')
    
    make_surface(measured, fit_fun, popt)
    make_contour_plot(measured, fit_fun, popt)



