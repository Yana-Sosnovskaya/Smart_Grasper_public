# Position Calibration

This contains the code necessary for calibrating the Grasper's position (jaw distance). It also contains the data files resulting from this calibration process. 

### Files

The `position_surface_fit.py` file is the main interface for the user. It is the script that uses the data from the files in `./Data Files` tand uses it to generate an best fit equation that maps the encoder and force readings to a jaw distance in *mm*.
- This script relies on the files in the `./Data Files` folder and on the `position_data_processing_script.py` script
- To generate the calibration function, open `position_surface_fit.py` and select which data sources you want to use by commenting or uncommenting the file names as appropriate. Then, run the script.

The `position_data_processing_script.py` file is the python script that takes reads the data from the _.csv_ files in the `./Data Files` folder, processes it, and outputs it in a more usable form.

The folder `./Data Files` contains the *.csv* data files used to generate the position calibration function found in the submitted paper.
- The data files are named `Position_Calibration_MM_DD_YYYY_X.csv`
    - The *MM_DD_YYYY* portion of the name is the date in month-day-year format represented by 2, 2, and 4 digits, respectivley.
    - The *X* portion of the name indicates that this is the *X*<sup>th</sup> data set gathered on that particular day (indexed from 0). These will not always be sequential as errors during the calibration process would still created an ordered file name but would not yield usable data.
- The column headings of the data files are:
    - Measured Distance (mm): the distance between the jaws to the neares 0.1 mm as measured by the calipers
    - Nominal Distance (mm): the nominal distance of the block being used for the position calibration. This is the distance written on the block.
    - PWM Value: the 8-bit pulse-width-modulation value that was fed to the motor controller
    - Force Reading (Counts): the 10-bit force reading from the Gasper's internal load cell
    - Encoder Value (Pulses): the reading from the encoder
- Multiple data points are gathered at each combination of inputs. Their values are later averaged during the analysis process.

### How to Use:

To generate the calibration function, open `position_surface_fit.py` and select which data sources you want to use by commenting or uncommenting the data file names as appropriate. Then, run the script.
