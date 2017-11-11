#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

//#define PI (3.1415926535897932384626433832795)

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
   is_initialized_ = false;
   
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  // initial state vector
  x_ = VectorXd(5);
  x_ << 0,0,0,0,0;

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 0.0225,0,0,0,0,
        0,0.0225,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,0,0,0,1;
        
  ///* predicted sigma points matrix
  Xsig_pred_ = MatrixXd(5,5);

  ///* time when the state is true, in us
  time_us_ = 0;

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 3;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 3;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;  

  // State dimension
  n_x_ = 5;

  // Augmented state dimension
  n_aug_ = 7;

  // Sigma point spreading parameter
  lambda_ = 3 - n_aug_;

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
   
  /******************************************************************/
  /*                 Initialization Structure                       */
  /******************************************************************/
  if(!is_initialized_){
     cout<<"UKF: "<<endl;
    
    if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
      x_(0) = meas_package.raw_measurements_[0]*cos(meas_package.raw_measurements_[1]);
      x_(1) = meas_package.raw_measurements_[0]*sin(meas_package.raw_measurements_[1]);
    }
    else if (meas_package.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
      x_(0) = meas_package.raw_measurements_[0];
      x_(1) = meas_package.raw_measurements_[1];
    }
    
    previous_timestamp_ = meas_package.timestamp_;
        
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }  
  
  /******************************************************************/
  /*                     Control Structure                          */
  /******************************************************************/  
  //compute the time elapsed between the current and previous measurements
  float dt = (meas_package.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
  
  Prediction(dt);
  
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
     UpdateRadar(meas_package);
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
     UpdateLidar(meas_package);
  }  
  previous_timestamp_ = meas_package.timestamp_;
}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
  /**
  TODO:

  Complete this function! Estimate the object's location. Modify the state
  vector, x_. Predict sigma points, the state, and the state covariance matrix.
  */

  //create augmented mean vector
  VectorXd x_aug = VectorXd(n_aug_);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(n_aug_, n_aug_);

  //create sigma point matrix
  MatrixXd Xsig_aug = MatrixXd(n_aug_, 2 * n_aug_ + 1);
  
  //create matrix with predicted sigma points as columns
  MatrixXd Xsig_pred = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  //create vector for weights
  VectorXd weights = VectorXd(2 * n_aug_ +1);
  
  /******************************************************************/
  /*                Create Augmented Sigma Points                   */
  /******************************************************************/
  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  Xsig_aug.col(0)  = x_aug;
  for (int i = 0; i< n_aug_; i++)
  {
    Xsig_aug.col(i+1)       = x_aug + sqrt(lambda_+n_aug_) * L.col(i);
    Xsig_aug.col(i+1+n_aug_) = x_aug - sqrt(lambda_+n_aug_) * L.col(i);
  }
  
  /******************************************************************/
  /*                Predict Augmented Sigma Points                  */
  /******************************************************************/
  for(int i = 0; i < ((2*n_aug_)+1); i++)
  {
      //extract values for better readability
      double p_x = Xsig_aug(0, i);
      double p_y = Xsig_aug(1, i);
      double v = Xsig_aug(2, i);
      double yaw = Xsig_aug(3, i);
      double yawd = Xsig_aug(4, i);
      double nu_a = Xsig_aug(5, i);
      double nu_yawdd = Xsig_aug(6, i);

      //predicted state values
      double px_p, py_p;

      //avoid division by zero
      if(fabs(yawd) > 0.001)
      {
      px_p = p_x + (v/yawd) * (sin(yaw + (yawd*delta_t)) - sin(yaw));
      py_p = p_y + (v/yawd) * (cos(yaw) - cos(yaw + (yawd*delta_t)));
      }
      else
      {
      px_p = p_x + (v*delta_t*cos(yaw));
      py_p = p_y + (v*delta_t*sin(yaw));
      }

      double v_p = v;
      double yaw_p = yaw + (yawd*delta_t);
      double yawd_p = yawd;

      //add noise
      px_p += ((0.5*nu_a*delta_t*delta_t)*cos(yaw));
      py_p += ((0.5*nu_a*delta_t*delta_t)*sin(yaw));
      v_p += (nu_a*delta_t);

      yaw_p += (0.5*nu_yawdd*delta_t*delta_t);
      yawd_p += (nu_yawdd*delta_t);

      Xsig_pred(0, i) = px_p;
      Xsig_pred(1, i) = py_p;
      Xsig_pred(2, i) = v_p;
      Xsig_pred(3, i) = yaw_p;
      Xsig_pred(4, i) = yawd_p;  
  }
  
  
  /******************************************************************/
  /*                Predict Mean and Covariance                     */
  /******************************************************************/
  // set weights
  double weight_0 = lambda_/(lambda_ + n_aug_);
  weights(0) = weight_0;
  for (int i = 1; i < ((2 * n_aug_)+1); i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {  //iterate over sigma points
    x_ += (weights(i) * Xsig_pred.col(i));
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred.col(i) - x_;
    //angle normalization
    while (x_diff(3)> M_PI) x_diff(3)-=2.*M_PI;
    while (x_diff(3)<-M_PI) x_diff(3)+=2.*M_PI;

    P_ += (weights(i) * x_diff * x_diff.transpose()) ;
  }
}

/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use lidar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the lidar NIS.
  */
}

/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Use radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.

  You'll also need to calculate the radar NIS.
  */
}
