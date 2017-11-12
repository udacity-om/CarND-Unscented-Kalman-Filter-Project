#include "ukf.h"
#include "Eigen/Dense"
#include <iostream>

//#define PI (3.1415926535897932384626433832795)

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

static void angleNormalization(double *angle);

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
        
  // predicted sigma points matrix
  Xsig_pred_ = MatrixXd(n_x_, 2 * n_aug_ + 1);
  
  // Weights of sigma points
  weights_ = VectorXd((2 * n_aug_) +1);

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
  
  //cout<<"Before Prediction"<<endl;
  Prediction(dt);
  //cout<<"After Prediction"<<endl;
  
  if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
     //cout<<"Before UpdateRadar"<<endl;
     UpdateRadar(meas_package);
     //cout<<"After UpdateRadar"<<endl;
  }
  else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
     //cout<<"Before UpdateLidar"<<endl;
     UpdateLidar(meas_package);
     //cout<<"After UpdateLidar"<<endl;
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
  MatrixXd Xsig_aug = MatrixXd(n_aug_, (2 * n_aug_) + 1);
  
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

      Xsig_pred_(0, i) = px_p;
      Xsig_pred_(1, i) = py_p;
      Xsig_pred_(2, i) = v_p;
      Xsig_pred_(3, i) = yaw_p;
      Xsig_pred_(4, i) = yawd_p;  
  }
  
  /******************************************************************/
  /*                Predict Mean and Covariance                     */
  /******************************************************************/
  // set weights
  double weight_0 = lambda_/(lambda_ + n_aug_);
  weights_(0) = weight_0;
  for (int i = 1; i < ((2 * n_aug_)+1); i++) {  //2n+1 weights
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {  //iterate over sigma points
    x_ += (weights_(i) * Xsig_pred_.col(i));
  }

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 0; i < (2 * n_aug_) + 1; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
    
    //angle normalization
    angleNormalization(&x_diff(3));

    P_ += (weights_(i) * x_diff * x_diff.transpose()) ;
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
  MatrixXd R_ = MatrixXd(2, 2);
  MatrixXd H_ = MatrixXd(2, 5);
  //measurement covariance matrix - laser
  R_ << 0.0225, 0,
        0, 0.0225;
  // measurement matrix - laser
  H_ << 1,0,0,0,0,
        0,1,0,0,0;          
        
  VectorXd z = meas_package.raw_measurements_;        
              
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  MatrixXd Ht = H_.transpose();
  MatrixXd S = (H_ * P_ * Ht) + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;
  
  // new estimate
  x_ += (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - (K * H_)) * P_;
  
  #epsilon_ = y.trans * S.inverse() * y
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
  
  /******************************************************************/
  /*                Predict Radar Sigma Points                      */
  /******************************************************************/
  //set measurement dimension, radar can measure r, phi, r_dot 
  int n_z = 3;
  
  //create matrix for sigma points in measurement space
  MatrixXd Zsig = MatrixXd(n_z, (2 * n_aug_) + 1); 
  
  //mean predicted measurement
  VectorXd z_pred = VectorXd(n_z);
  
  //measurement covariance matrix S
  MatrixXd S = MatrixXd(n_z, n_z);
  
  //measurement noise covariance matrix R
  MatrixXd R = MatrixXd(n_z,n_z);
  
  //transform sigma points in measurement space
  for(int i = 0; i < ((2*n_aug_)+1); i++)
  {
     //extract values for better readability
     double p_x = Xsig_pred_(0, i);
     double p_y = Xsig_pred_(1, i);
     double v = Xsig_pred_(2, i);
     double yaw = Xsig_pred_(3, i);
     
     double v1 = v*cos(yaw);
     double v2 = v*sin(yaw);
     
     //measurement model
     Zsig(0,i) = sqrt((p_x*p_x) + (p_y*p_y));         //r
     Zsig(1,i) = atan2(p_y, p_x);                     //phi
     Zsig(2,i) = ((p_x*v1) + (p_y*v2))/Zsig(0,i);     //r_dot 
  }
  
  //mean predicted measurement
  z_pred.fill(0.0);
  for(int i = 0; i < ((2*n_aug_)+1); i++)
  {
     z_pred += (weights_(i)*Zsig.col(i));
  }
  
  //measurement covariance matrix See
  S.fill(0.0);
  for(int i = 0; i <  ((2*n_aug_)+1); i++) //2n+1 simga points
  {
     //residual
     VectorXd z_diff = Zsig.col(i) - z_pred;
     
     //angle normalization
     angleNormalization(&z_diff(1));
     
     S += (weights_(i)*z_diff*z_diff.transpose());
  }
  
  //add measurement noise covariance matrix
  R << (std_radr_*std_radr_), 0, 0,
       0, (std_radphi_*std_radphi_), 0,
       0, 0, (std_radrd_*std_radrd_);
  
  S += R;
  
  
  /******************************************************************/
  /*                         Update Radar                           */
  /******************************************************************/
  //create vector for incoming radar measurement
  VectorXd z = VectorXd(n_z);
  
  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  
  Tc.fill(0.0);
  for(int i = 0; i < ((2*n_aug_)+1); i++) //2n+1 sigma points
  {
     //residual
     VectorXd z_diff = Zsig.col(i) - z_pred;
     
     //angle normalization
     angleNormalization(&z_diff(1));
     
     //state difference
     VectorXd x_diff = Xsig_pred_.col(i) - x_;
     
     //angle normalization
     angleNormalization(&x_diff(3));
     
     Tc += (weights_(i)*x_diff*z_diff.transpose());     
  }
  
  //Kalman gain K
  MatrixXd K = Tc * S.inverse(); 
  
  z << meas_package.raw_measurements_(0),
       meas_package.raw_measurements_(1),
       meas_package.raw_measurements_(2);
  
  //residual
  VectorXd z_diff = z - z_pred;
  
  //angle normalization
  angleNormalization(&z_diff(1));
  
  x_ += (K*z_diff);
  P_ -= (K*S*K.transpose());  
}

static void angleNormalization(double *angle)
{
   //angle normalization
   double temp_angle = 0;
   
   temp_angle = *angle;
   
   while(temp_angle > M_PI){
      temp_angle -= (2.0*M_PI);
   }
   while(temp_angle < -M_PI){
      temp_angle += (2.0*M_PI);
   }   
   
   *angle = temp_angle;      
}
