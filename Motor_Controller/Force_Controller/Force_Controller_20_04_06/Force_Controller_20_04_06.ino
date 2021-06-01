// This sketch acts as a force controller for the smart grasper
// control system. 

// Futek CSG110 Amplifier
// 6 April 2020
// Jack Kaplan and Matthew Arnold

// Note: Unless Otherwise specififed
// All forces measured in Newtons
// All voltages measured in Volts

// TODO:
// Determine actual force calibration curve
// Determine ideal value for kp
// Find way to set and determine the refrence force digitally 
// Optional: Find way to read the actual voltage value

// Potentiometer pin for refrence force input
#define Fr_POT A0

// output pin for command voltage
#define Vout_PIN 9

// input pin for strain gauge on grasper using futek amplifier 
#define GRASPER_STRAIN_GAUGE_PIN A1

// Maximum force output allowed to prevent damage to tissue
const int Maximum_Force_Output = 10; //TODO calculate actual value

// Maximum allowable voltage corresponding to the 
// maximum force output
const float max_vout = 2.5; //TODO calculate actual value

//Current applied voltage. Update every cycle
float v_now = 0;

void setup() {
  // put your setup code here, to run once:

  // ensure the initial output voltage = 0 V
  output_voltage(v_now);
  
  Serial.begin(9600);

  Serial.println("Initializing the strain gauge");
  pinMode(GRASPER_STRAIN_GAUGE_PIN, INPUT);

}

void loop() {
  // put your main code here, to run repeatedly:

  force_controller();

}


// Force Controller. Implemented this way to allow it to be
// combined with other Arduino code.
void force_controller(){

  // get actual force and refrence force measured in Newtons
  float Fa = get_force();
  float Fr = refrence_force();

  // get the voltage added by the error between the refrence
  // and actual forces
  float v_add = p_controller(Fa, Fr);

  // get the new v_out
  float v_out = v_add + v_now;

  // safety mechanism to ensure v_out <= max_vout
  v_out = min(max_vout, v_out);

  output_voltage(v_out);
  
}

// TODO: Determine ideal value for kp
// Proportional Controller
float p_controller(float F, float Fr){

  //Difference between output and refrence forces measured in N
  float error = Fr - F; 
  
  float kp = 0.1; // proportional error constant (V/N)
  float v = kp * error; // calculate output voltage

  return v;
}

// returns a float value of the force in Newtons
// the grasper is applying
float get_force() {
  float grasperStrainGaugeData = analogRead(GRASPER_STRAIN_GAUGE_PIN);

  //convert the raw gasper strain gauge data to the actual force
  //output in Newtons
  float force;
  force = grasper_force_calibration(grasperStrainGaugeData);  
  
  return force;

}

// TODO: Determine actual calibration curve
// Takes in the Raw Grasper Strain Gauge data and then uses
// the experimentally determined calibration curve to output the 
// actual force in Newtons
float grasper_force_calibration(float strain_data){

  //TODO: replace with actual calibration curve
  long data_max = 16777216;
  long data_min = 0;
  float force = I2F_map(strain_data, data_min, data_max, 0, 50);

  return force;
}

// TODO: find way to set and determine the refrence force digitally 
// Get the refrence force
float refrence_force() {

  // Read in refrence force from potentiometer 
  float f_min = 0;
  float f_max = 10;
  float fr_raw = analogRead(Fr_POT);
  int fr_scaled = I2F_map(fr_raw, 0, 1024, f_min, f_max);

  // make sure the refrence force does not exceed the maximum
  // allowed force output
  float fr_out = min(fr_scaled, Maximum_Force_Output);
  
  return fr_out;

}


// Takes in a floating v_out between 0 and the maximum
// output voltage (max_vout) and converts it to a PWM singal
// that it then outputs. If v_out is outside the bounds, then
// it either outputs the minimum or maximum voltage.
void output_voltage(float v_out) {

  // Checks to see if v_out between 0 and max_vout volts
  if ((v_out >= 0) and (v_out <= max_vout)) {
  }

  // If v_out negative, v_out = 0
  else if (v_out < 0) {
    //Serial.println("Error in output_voltage(v_out), v_out < 0");
    v_out = 0;
  }

  // If v_out > max_vout, v_out = max_vout
  else if (v_out > max_vout) {
    //Serial.println("Error in output_voltage(v_out), v_out > 5");
    v_out = max_vout;
  }

  // If something else goes wrong, then be safe and output 0 V
  else {
    Serial.println("Unknown Error in output_voltage()");
    v_out = 0;
  }

  // update v_now so it reflects the current v_out
  v_now = v_out;

  // Map the voltage v_out to the correct PWM 
  int v_command = F2I_map(v_out, 0, 5, 0, 255);
  analogWrite(Vout_PIN, v_command);
}



//mapping function that takes in a float and outputs an integer
int F2I_map(float x, long in_min, long in_max, long out_min, long out_max) {
  int map_out = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
  return map_out;
}


//mapping function that takes in an integer and outputs a float
float I2F_map(long x, long in_min, long in_max, float out_min, float out_max) {
  float map_out = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
  return map_out;
}

//mapping function that takes in a float and outputs an integer
float F2F_map(float x, long in_min, long in_max, long out_min, long out_max) {
  float map_out = (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min;
  return map_out;
} 
