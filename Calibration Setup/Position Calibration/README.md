# Position Calibration

This contains the code necessary for calibrating the Grasper's position (jaw distance). It also contains the data files resulting from this calibration process. 

### Files

The `position_surface_fit.py` file is the main interface for the user. It is the script that uses the data from the files in `./Data Files` tand uses it to generate an best fit equation that maps the encoder and force readings to a jaw distance in _mm_.

    - This script relies on the files in the `./Data Files` folder and on the `position_data_processing_script.py` script
    - To generate the calibration function, open `position_surface_fit.py` and select which data sources you want to use by commenting or uncommenting the file names as appropriate. Then, run the script.

The `position_data_processing_script.py` file is the python script that takes reads the data from the _.csv_ files in the `./Data Files` folder, processes it, and outputs it in a more usable form.

The folder `./Data Files` contains the _.csv_ data files used to generate the position calibration function found in the submitted paper.
