# Force Calibration

This contains the code necessary for calibrating internal grasper force sensor. It also contains the data files resulting from this calibration process. 

### Scripts

The `force_surface_fit.py` file is the main interface for the user. It is the script that uses the data from the files in `./Data Files` tand uses it to generate an best fit equation that maps the encoder and force readings to the force applied by the Grasper's jaws in newtons.
- This script relies on the files in the `./Data Files` folder and all the scripts in this folder.
- To generate the calibration function, open `force_surface_fit.py` and select which data sources you want to use by commenting or uncommenting the file names as appropriate. Then, run the script.

The  `force_weight_comparison_analysis.py` file is the python script that is used to process the data relating the external load cell to known weights. It takes in the data, processes it, and then outputs the data in a more usable form.

The `force_calibration_data_processing_script.py` file is the python script that is used to process the data relating the internal load cell readings to the external load readings. It takes in the data, processes it, and then outputs the data in a more usable form.

The `force_error_functions.py` file is a python script that contains error functions that are used by the other scripts.
- `fit_error(fit_values, measured_values)` finds the difference between the measured values from the calibration process and the corresponding calculated values found from the fit function.
- `mean_absolute_deviation(fit_values, data_values)` finds the mean absolute deviation between the measured and calculated values.
- `root_mean_square_error(fit_values, data_values)` finds the root mean square error between the measured and calculated values.

### Data Files

The folder `./Data Files` contains the *.csv* data files used to generate the function giving the force applied the Grasper's jaws.

##### Relating the External Load Cell Readings to Known Weights

- The data files are named `Force_Weight_Comparison_MM_DD_YYYY_X.csv`
    - The *MM_DD_YYYY* portion of the name is the date in month-day-year format represented by 2, 2, and 4 digits, respectivley.
    - The *X* portion of the name indicates that this is the *X*<sup>th</sup> data set gathered on that particular day (indexed from 0). These will not always be sequential as errors during the calibration process would still created an ordered file name but would not yield usable data.
- The column headings of the data files are:
    - Trial: the trial number of the data points. A trial consists of a single weight being measured. 
    - Mass (g): the mass hanging from the external load cell measured in grams. This is the sum of the distances written on the mass (the masses were not  weighed on another scale).
    - Load Cell Reading: the 24-bit reading from the external load cell
- Multiple data points are gathered at each combination of inputs. Their values are later averaged during the analysis process.
- There is only one file in this format. 

##### Relating the Internal Load Cell Readings to the Extneral Load Cell Readings
- The data files are named `Force_Calibration_MM_DD_YYYY_X.csv`
    - The *MM_DD_YYYY* portion of the name is the date in month-day-year format represented by 2, 2, and 4 digits, respectivley.
    - The *X* portion of the name indicates that this is the *X*<sup>th</sup> data set gathered on that particular day (indexed from 0). These will not always be sequential as errors during the calibration process would still created an ordered file name but would not yield usable data.
- The column headings of the data files are:
    - Trial: the trial number of the data points. A trial consists of a single pulse-width-modulation input to the motor controller at a given jaw distance during a single run of the calibration process.
    - Jaw Distance (mm): the jaw distance to the nearest 0.1 mm as measured by calipers.
    - PWM: the 8-bit pulse-width-modulation value that was fed to the motor controller
    - Grasper Force Sensor (Counts): the 10-bit force reading from the Gasper's internal load cell
    - External Force Sensor (Counts): the 24-bit reading from the external load cell
    - Encoder Value (Pulses): the reading from the encoder
- Multiple data points are gathered during each trial. Their values are later averaged during the analysis process.

### How to Use

To generate the calibration function, open `position_surface_fit.py` and select which data sources you want to use by commenting or uncommenting the data file names as appropriate. Then, run the script. This will output a range of options for the function that that can be used to find the force applied at by the Grasper's jaws.
