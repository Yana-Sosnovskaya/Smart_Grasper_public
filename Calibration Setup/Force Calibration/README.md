# Force Calibration

This contains the code necessary for calibrating internal grasper force sensor. It also contains the data files resulting from this calibration process. 

## Calibration Process

### Finding the curve (or surface) mapping the internal load cell readings to the external load cell readings

##### Calibration Files

The `force_calibration_python_script.py` file is the python script that acts as the data logger for the calibration procedure and is the primary interface for the user. Currently it only works with the  `fast_force_calibration.ino` script. Future work will either modify this script or create a new version of this script to work with the regular `force_calibration.ino` script. The output of this script is saved in the folder `./Data Files` (see procedure for more information).

The `force_calibration_data_processing_script.py` file is the python script that can be used to help analyze the data from `force_calibration_python_script.py`. It plots reads the data from the CSV and plots it. It also outputs the data in a form that can be used by other scipts as well.

The folder `./force_calibration` contains the file `force_calibration.ino`. This file is code for running the physical calibration process. This version of the calibration process allows you to precisely select the pwm input ranges to calibrate. It assumes that it is being run on a Teensy 4.0 microcontroller. This script is not usable right now because of changes made during the development of `fast_force_calibration.ino`. Future work will revert it back to its previous working state. 

The folder `./fast_force_calibration` contains the file `fast_force_calibration.ino`. This file is code for running the physical calibration process. This version of the calibration process tests for pwm inputs from [30,225] inclusive. It can be run multiple times for different jaw distances. It assumes that it is being run on a Teensy 4.0 microcontroller. 

##### Calibration Procedure
To get the data needed to map the internal to the external load cell:

1. Ensure all the hardware is properly set up and powered on
    - Make sure that the grasper properly closes on the force sensor and the plastic stopper without issue.
    - More detail to come
2. Check the metadata generating function `generate_json_force` and its inputs in `force_calibration_python_script.py` to make sure that they are correct. This includes the comments. 
3. Upload the latest version of `fast_force_calibration.ino` to the microcontroller. If `fast_force_calibration.ino` has not changed and was the most recent file uploaded, you can press the reset button to begin the script from the beginning.
4. When the indicator LED blinks three times, begin running `force_calibration_python_script.py`
5. Follow the prompts from the python script to run the calibration process
6. This should generate a CSV file in the `./Data Files` folder with the name `Force_Calibration_MM_DD_YYYY_X.csv`. The metadata for this csv file is added to `Motor_Controller_Grasper_Project.json` in the `./Calibration Setup` folder.
   - The *MM_XX_YYYY* in the file name is replaced with the month, day, and year that the position calibration is being run in that format
   - The *X* in the file name is replaced by the trial number of the calibration run for that day

### Mapping the external load cell readings to known forces

##### Files

The `force_weight_comparison.py` file is the python script that acts as the data logger for the calibration procedure and is the primary interface for the user. It is used in concert with `force_weight_comparison.ino`. The output of this script is saved in the folder `./Data Files` (see procedure for more information).

The folder `./force_weight_comparison` contains the file `force_weight_comparison.ino`. This file is code for running the calibration process. It assumes that it is being run on a Teensy 4.0 microcontroller.

The `externalforce_weight_analysis.py` file is the python script that can be used to help analyze the data from `force_weight_comparison.py`. It plots reads the data from the CSV and plots it. It also outputs the data in a form that can be used by other scipts as well.

##### Calibration Procedure

1. Ensure all the hardware is properly set up and powered on
    - This involves removing the aluminum plate that is attached to the external load cell is replacing it with the 3D printed part from which weights can be hung. 
    - The load cell should be rotated 90 degrees so that it perpendicular to the grasper arm. This allows the weights to hang off of the table.
2. Check the metadata generating function `generate_json_fw` and its inputs in `force_calibration_python_script.py` to make sure that they are correct. This includes the comments. 
3. Upload the latest version of `force_weight_comparison.ino` to the microcontroller. If `force_weight_comparison.ino` has not changed and was the most recent file uploaded, you can press the reset button to begin the script from the beginning.
4. When the indicator LED blinks three times, begin running `force_weight_comparison.py`
5. Follow the prompts from the python script to run the calibration process
    - Use the weight set as the hanging masses. You can treat their nominal value as close enough to their real value for the purpose of this calibration. 
6. This should generate a CSV file in the `./Data Files` folder with the name `Force_Weight_Comparison_MM_DD_YYYY_X.csv`. The metadata for this csv file is added to `Motor_Controller_Grasper_Project.json` in the `./Calibration Setup` folder.
   - The *MM_XX_YYYY* in the file name is replaced with the month, day, and year that the position calibration is being run in that format
   - The *X* in the file name is replaced by the trial number of the calibration run for that day


## Data Processing

The data in `Force_Calibration_MM_DD_YYYY_X.csv` can be used to create a function relating the reading of internal load cell to the reading the external load sensor. The data in `Force_Weight_Comparison_MM_DD_YYYY_X.csv` can be used to create a curve mapping the external load cell readings to their value in a usable unit (e.g. g or N). These results of the second function can be substituted into the first to directly relate the internal load cell readings to the grasper force. This work is in progress.
