This contains code for calibrating both the grasper's position and the internal force sensor. It also contains a skeleton for the motor controller algorithm as well.

In `./Force_controller`, you will find a couple of source files:
 * `force_calibration_setup.ino` contains code for calibrating the strain gauges and encoder and code for feeding in calibrated data to the motor controller. This code is out of date and no longer being used
 * `Force_Controller_*.ino` contains skeleton code for the smart grasper's motor control system.

In `./Calibration Setup`, you will find four items:
 * The `./Arduino Libraries` folder contains the non-standard Arduino Libraries needed for the calibration code to run. 
 * The `./Force Calibration` folder contains the scripts needed to calibrate the grasper force sensor and the data files resulting from the calibration process
 * The `./Force Calibration` folder contains the scripts needed to calibrate the grasper position and the data files resulting from the calibration process
 * The file `Motor_Controller_Grasper_Project.json` is the metadata for all the data files generated in the calibration process

Each folder in `./Calibration Setup` will have its own `README` file. 

For additional documentation, refer to the smart grasper [google drive](https://drive.google.com/drive/u/1/folders/1FHCQTBn3cXSNdjZ81Fsf6uGQNvmu3iUK). See "June 18 2020 - Grasper Documentation.pdf" for a complete overview.

### What has been done in the project:
 * Raw data from external load cell (Adafruit TAL220B), which can be found [here](https://www.sparkfun.com/products/14729). The data from the load cell is amplified using an [HX711 breakout](https://www.sparkfun.com/products/13879), which is connected to the Arduino via I2C.
 * Raw data from the grasper's strain gauge amplified via the original futek amplifier. We are currently going to use this (as opposed to using a newer, cheap amplifier like the HX711 because after testing it amplifies enough to be usable, it has very little steady state error, it runs on 12 V, and it is consistent.
 * Raw data from the motor's quadrature encoder.
 * Data corresponding load cell located at the grasper's jaws, strain gauge on the grasper, and motor voltage. Note that the calibration curve corresponding load cell and strain gauge data may have to be recalculated to account for grasper jaw distance from each other.

### What needs to be done in this project:
 * Generate calibration curve that correlates force
with the load cell data. When this is done, put the
calibration curve in `CalculateLoadCellForce`.
 * Use this along with the calibration curve that has been calculated to correspond grasper strain gauge data with grasper force.
 * Test the calibration curve by comparing `grasper_jaw_force` and `grasper_jaw_force_actual` to see if the curve needs to be improved.
 * Add in the force controller once grasper strain gauge data has been correlated to grasper jaw force.
 * In addition to force, we want an accurate jaw position. todo.
