#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  cout<<"Initializing"<<endl;
  // if this is false, laser measurements will be ignored (except during init)
  use_laser_ = true;

  // if this is false, radar measurements will be ignored (except during init)
  use_radar_ = true;

  n_x_ = 5;

  n_aug_ = n_x_ + 2;

  n_sig_ = (2 * n_aug_) + 1;
  // initial state vector
  x_ = VectorXd(n_x_);

  // initial covariance matrix
  P_ = MatrixXd(n_x_, n_x_);

  Xsig_pred_ = MatrixXd(n_x_, n_sig_);

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 5;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 1;

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

  /**
  TODO:

  Complete the initialization. See ukf.h for other member properties.

  Hint: one or more values initialized above might be wildly off...
  */
  P_ << 1,0,0,0,0,
        0,1,0,0,0,
        0,0,1,0,0,
        0,0,0,1,0,
        0,0,0,0,1;

  n_x_ = 5;
  lambda_ = 3 - n_aug_;
  NIS_laser_ = 0.0;
  NIS_radar_ = 0.0;
  laser_count_ = 0;
  radar_count_ = 0;
  weights_ = VectorXd(n_sig_);
  InitializeWeights();

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
  /**
  TODO:

  Complete this function! Make sure you switch between lidar and radar
  measurements.
  */
  cout<<"Process Measurements"<<endl;
  if(false == is_initialized_)    
  {
    cout<<"First measurement"<<endl;
    x_ = VectorXd(5);
    if( MeasurementPackage::LASER == meas_package.sensor_type_)
    {
      cout<<"First measurement is LASER"<<endl;
      x_(0) = meas_package.raw_measurements_(0);
      x_(1) = meas_package.raw_measurements_(1);
      x_(2) = 0;
      x_(3) = 0;//atan2(x_(1),x_(0));
      x_(4) = 0;
    }
    else
    {
      cout<<"First measurement is RADAR"<<endl;
      x_(0) = meas_package.raw_measurements_(0) * cos(meas_package.raw_measurements_(1));
      x_(1) = meas_package.raw_measurements_(0) * sin(meas_package.raw_measurements_(1));
      x_(2) = meas_package.raw_measurements_(2);
      x_(3) = meas_package.raw_measurements_(1);
      x_(4) = 0;
    }
    
    time_us_ = meas_package.timestamp_;    
    is_initialized_ = true;
    cout<<"First x : "<<endl;
    cout<<x_;
    cout<<endl;
    cout<<endl;
    cout<<"First P : "<<endl;
    cout<<P_;
    cout<<endl;
    cout<<endl;
    cout<<"End of Measurement processing.\n\n\n\n\n\n\n\n";
    return;
  }

  double dt = ((double)meas_package.timestamp_ - (double)time_us_)/1000000.0;
  time_us_ = meas_package.timestamp_;    

  cout<<"Going to perform prediction, time elapsed : "<<dt<<endl;
  Prediction(dt);

  if(MeasurementPackage::LASER == meas_package.sensor_type_)
  {
    cout<<"LASER update"<<endl;
    UpdateLidar(meas_package);
  }
  else
  {
    cout<<"RADAR update"<<endl;
    UpdateRadar(meas_package);
  }
  cout<<"End of Measurement processing.\n\n\n\n\n\n\n\n";

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

  /* 1. Generate Sigma points */
  /* 2. Augment Sigma points with noise */
  cout<<"Calculating augmented sigma points"<<endl;
  MatrixXd x_aug = AugmentSigmaPoints();
  cout<<"Augmented X :"<<endl;
  cout<<x_aug;
  cout<<endl;
  cout<<endl;
  /* 3. Augment Covariance Matrix with noise */
  /* 4. Predict state updated sigma points */
  cout<<"Predicting updated sigma points"<<endl;

  PredictSigmaPoints(x_aug, delta_t);
  cout<<Xsig_pred_;
  cout<<endl;
  cout<<endl;
  /* 5. Get state vector and covariance from predicted sigma points */
  cout<<"Performing state update from predicted sigma points"<<endl;
  GetStateFromSigmaPoints();
  cout<<"State : "<<endl;
  cout<<x_;
  cout<<endl;
  cout<<endl;
  cout<<"P Matrix : "<<endl;
  cout<<P_;
  cout<<endl;
  cout<<endl;
}

MatrixXd UKF::GetSigmaPoints() {
  MatrixXd sigmaPoints(n_x_, (2 * n_x_) + 1);
  lambda_ = 3 - n_x_;

  MatrixXd sqrtP = P_.llt().matrixL();

  sqrtP = sqrt(lambda_ + n_x_) * sqrtP;

  sigmaPoints.col(0) = x_;
  for(int i = 1; i < n_x_ + 1; i++)
  {
    sigmaPoints.col(i) = x_ + sqrtP.col(i-1);
    sigmaPoints.col(i + n_x_) = x_ - sqrtP.col(i-1);
  }
  return sigmaPoints;
}

MatrixXd UKF::AugmentSigmaPoints() {
  /* Local variable declarations */
  MatrixXd augSigmaPoints(n_aug_, n_sig_);
  VectorXd x_aug(n_aug_);

  MatrixXd P_aug(n_aug_, n_aug_);
  MatrixXd Q(2,2);

  /* Build the noise matrix */
  Q.fill(0.0);
  Q(0,0) = std_a_ * std_a_;
  Q(1,1) = std_yawdd_ * std_yawdd_;

  /* Get augmented Covariance matrix */
  P_aug.fill(0.0);
  P_aug.block(0,0,n_x_,n_x_) = P_;
  P_aug.block(n_x_,n_x_,2,2) = Q;

  /* Get Square root of covariance matrix */
  MatrixXd sqrtP = P_aug.llt().matrixL();
  sqrtP = sqrt(lambda_ + n_aug_) * sqrtP;

  /* Create Augmented state matrix */
  x_aug.segment(0,n_x_) = x_;
  x_aug(n_x_) = 0;
  x_aug(n_x_ + 1) = 0;

  augSigmaPoints.col(0) = x_aug;

  for(int i = 0; i < n_aug_; i++)
  {
    augSigmaPoints.col(i + 1) = x_aug + sqrtP.col(i);
    augSigmaPoints.col(i + n_aug_+ 1) = x_aug - sqrtP.col(i);
  }

  return augSigmaPoints;

}


void UKF::PredictSigmaPoints(MatrixXd x_aug, double dt) {

  for(int i = 0; i < n_sig_; i++)
  {
    double state_update_x;
    double state_update_y;
    double state_update_v;
    double state_update_rho;
    double state_update_rho_d;
    // State update Equations
    if(fabs(x_aug(4, i)) >= 0.001)
    {
      state_update_x     = x_aug(2 ,i)/x_aug(4,i) * ( sin(x_aug(3, i) + x_aug(4, i) * dt ) - sin(x_aug(3, i)) );
      state_update_y     = x_aug(2, i)/x_aug(4,i) * ( cos(x_aug(3, i)) - cos(x_aug(3, i) + x_aug(4, i) * dt ) );
      state_update_v     = 0;
      state_update_rho   = x_aug(4, i) * dt;
      state_update_rho_d = 0;
    }
    else
    {
      state_update_x     = x_aug(2, i) * cos(x_aug(3, i)) * dt;
      state_update_y     = x_aug(2, i) * sin(x_aug(3, i)) * dt;
      state_update_v     = 0;
      state_update_rho   = 0;
      state_update_rho_d = 0;
    }

    // Noise Update
    state_update_x     += 0.5 * (dt * dt) * cos(x_aug(3, i)) * x_aug(5, i);
    state_update_y     += 0.5 * (dt * dt) * sin(x_aug(3, i)) * x_aug(5, i);
    state_update_v     += dt * x_aug(5, i);
    state_update_rho   += 0.5 * (dt * dt) * x_aug(6, i);
    state_update_rho_d += dt * x_aug(6, i);
    // Final update to class variable 
    Xsig_pred_.col(i)(0) = x_aug(0,i) + state_update_x;
    Xsig_pred_.col(i)(1) = x_aug(1,i) + state_update_y;
    Xsig_pred_.col(i)(2) = x_aug(2,i) + state_update_v;
    Xsig_pred_.col(i)(3) = (x_aug(3,i) + state_update_rho);
    Xsig_pred_.col(i)(4) = x_aug(4,i) + state_update_rho_d;

    if(Xsig_pred_.col(i)(3) - NormalizeAngle(x_aug(3,i) + state_update_rho) > 1e-8)
    {
      cout<<endl<<"Predicted Sigma psi : "<<Xsig_pred_.col(i)(3)<<endl;
      cout<<"Un-normalized Sigma psi : "<<NormalizeAngle(x_aug(3,i) + state_update_rho)<<endl;
    }

  }
}

void UKF::GetStateFromSigmaPoints()
{
  //predicted state mean
  x_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points
    x_ = x_ + weights_(i) * Xsig_pred_.col(i);
  }

  x_(3) = NormalizeAngle(x_(3));

  //predicted state covariance matrix
  P_.fill(0.0);
  for (int i = 1; i < n_sig_; i++) {  //iterate over sigma points

    // state difference
    /* Trick to ensure Covariance matrix remains positive definite, since lambda is negative
     * this is not guaranteed without this trick. This trick causes some loss of accuracy.
     */
    VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);

    x_diff(3) = NormalizeAngle(x_diff(3));

    P_ = P_ + weights_(i) * x_diff * x_diff.transpose() ;
  } 
}

MatrixXd UKF::LidarGetZSigPoints(int n_z)
{
  MatrixXd Zsig(n_z, n_sig_);
  for(int i = 0; i < n_sig_; i++)
  {
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
 
    // measurement model
    Zsig(0,i) = p_x;
    Zsig(1,i) = p_y;

  }
  return Zsig;

}


void  UKF::InitializeWeights()
{
  VectorXd weights = VectorXd(n_sig_);
  double weight_0 = lambda_/(lambda_ + n_aug_);

  weights_(0) = weight_0;

  for (int i=1; i<n_sig_; i++) {  
    double weight = 0.5/(n_aug_+lambda_);
    weights_(i) = weight;
  }

}

VectorXd UKF::GetZPred(int n_z, MatrixXd Zsig, bool isRadar)
{
  VectorXd z_pred = VectorXd(n_z);
  z_pred.fill(0.0);
  for (int i=0; i < n_sig_; i++) {
      z_pred = z_pred + weights_(i) * Zsig.col(i);
  }
  if(true == isRadar)
  {
    z_pred(1) = NormalizeAngle(z_pred(1));
  }
  return z_pred;
}

void UKF::GetSMatrix(int n_z, MatrixXd R, MatrixXd Zsig, VectorXd z_pred, bool isRadar, MatrixXd &S)
{
  S.fill(0.0);
  for (int i = 0; i < n_sig_; i++) 
  {
    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    if(true == isRadar)
    {
      //angle normalization
      z_diff(1) = NormalizeAngle(z_diff(1));
    }

    S = S + weights_(i) * z_diff * z_diff.transpose();
  }

  S = S + R;
}


MatrixXd UKF::GetTMatrix(int n_z, MatrixXd Zsig, VectorXd z_pred, bool isRadar)
{
  MatrixXd Tc = MatrixXd(n_x_, n_z);
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

    //residual
    VectorXd z_diff = Zsig.col(i) - z_pred;

    if(true == isRadar)
    {
      //angle normalization
      z_diff(1) = NormalizeAngle(z_diff(1));
    }
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;

    if(true == isRadar)
    {
      //angle normalization
      x_diff(3) = NormalizeAngle(x_diff(3));
    }

    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }
  return Tc;

}

double UKF::NormalizeAngle(double input)
{
#if 0
  while(input >  M_PI) input -= (2.0 * M_PI);
  while(input < -M_PI) input += (2.0 * M_PI);
#else
  if(input < M_PI && input > -M_PI)
    return input;

  input = fmod(input, 2 * M_PI);

  if(input < -M_PI)
  {
    input += 2 * M_PI;
  }
  else if(input > M_PI)
  {
    input -= 2 * M_PI;
  }
#endif
  return input;
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
  int n_z = 2;

#if 1
  cout<<"LIDAR : Measurement points : "<<endl;
  cout<<meas_package.raw_measurements_;
  cout<<endl;
  cout<<endl;

  MatrixXd H(2,5);

  H << 1, 0, 0, 0, 0,
       0, 1, 0, 0, 0;

  VectorXd z = meas_package.raw_measurements_;

  VectorXd z_diff = z - (H * x_);
  cout<<"LIDAR : z_diff = "<<endl;
  cout<<z_diff;
  cout<<endl;
  cout<<endl;

  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_laspx_ * std_laspx_, 0                      , 
          0                      , std_laspy_ * std_laspy_;


  MatrixXd S = (H * P_ * H.transpose()) + R;
  cout<<"LIDAR : S = "<<endl;
  cout<<S;
  cout<<endl;
  cout<<endl;

  MatrixXd K = P_ * H.transpose() * S.inverse();
  cout<<"LIDAR : K = "<<endl;
  cout<<K;
  cout<<endl;
  cout<<endl;

  x_ = x_ + K * z_diff;
  x_(3) = NormalizeAngle(x_(3));

  P_ = (MatrixXd::Identity(5,5) - (K * H)) * P_;

  cout<<"LIDAR : x after update : "<<endl;
  cout<<x_;
  cout<<endl;
  cout<<endl;

  //P_ = P_ - K*S*K.transpose();
  cout<<"LIDAR : P after update : "<<endl;
  cout<<P_;
  cout<<endl;
  cout<<endl;
#else
  MatrixXd Zsig = LidarGetZSigPoints(n_z);
  cout<<"LIDAR : Zsig = "<<endl;
  cout<<Zsig;
  cout<<endl;
  cout<<endl;

  VectorXd z_pred = GetZPred(n_z, Zsig, false);
  cout<<"LIDAR : z_pred = "<<endl;
  cout<<z_pred;
  cout<<endl;
  cout<<endl;

  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_laspx_ * std_laspx_, 0, 
          0                      , std_laspy_ * std_laspy_;

  MatrixXd S(n_z, n_z);
  GetSMatrix(n_z, R, Zsig, z_pred, false, S);
  cout<<"LIDAR : S = "<<endl;
  cout<<S;
  cout<<endl;
  cout<<endl;

  //create matrix for cross correlation Tc
  MatrixXd Tc = GetTMatrix(n_z, Zsig, z_pred, false);

  cout<<"LIDAR : T = "<<endl;
  cout<<Tc;
  cout<<endl;
  cout<<endl;
  
  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  cout<<"LIDAR : K = "<<endl;
  cout<<K;
  cout<<endl;
  cout<<endl;

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  cout<<"LIDAR : z_diff = "<<endl;
  cout<<z_diff;
  cout<<endl;
  cout<<endl;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  //angle normalization
  x_(3) = NormalizeAngle(x_(3));
  cout<<"LIDAR : x after update : "<<endl;
  cout<<x_;
  cout<<endl;
  cout<<endl;

  P_ = P_ - K*S*K.transpose();
  cout<<"LIDAR : P after update : "<<endl;
  cout<<P_;
  cout<<endl;
  cout<<endl;
#endif
  
  NIS_laser_ = z_diff.transpose() * S.inverse() * z_diff;
  laser_count_++;

  cout<<"LIDAR : NIS = "<<NIS_laser_<<endl<<endl;
}



MatrixXd UKF::RadarGetZSigPoints(int n_z)
{
  MatrixXd Zsig(n_z, n_sig_);

  for(int i = 0; i < n_sig_; i++)
  {
    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // measurement model
    Zsig(0,i) = sqrt(p_x*p_x + p_y*p_y);                        //r
    if(Zsig(0,i) < 0.00001)
    {
      p_x = 0.00001;
      p_y = 0.00001;
      Zsig(0,i) = 0.00001;
      Zsig(2,i) = 0;                   //r_dot
    }
    else
    {
      Zsig(2,i) = (p_x*v1 + p_y*v2 )/Zsig(0,i);                   //r_dot
    }
    Zsig(1,i) = atan2(p_y,p_x);                               //phi

  }
  return Zsig;

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
  int n_z = 3;
  //set vector for weights

  cout<<"RADAR : Measurement points : "<<endl;
  cout<<meas_package.raw_measurements_;
  cout<<endl;
  cout<<endl;

  MatrixXd Zsig = RadarGetZSigPoints(n_z);
  cout<<"RADAR : Zsig = "<<endl;
  cout<<Zsig;
  cout<<endl;
  cout<<endl;

  VectorXd z_pred = GetZPred(n_z, Zsig, true);
  cout<<"RADAR : z_pred = "<<endl;
  cout<<z_pred;
  cout<<endl;
  cout<<endl;

  MatrixXd R = MatrixXd(n_z,n_z);
  R <<    std_radr_ * std_radr_, 0                        , 0,
          0                    , std_radphi_ * std_radphi_, 0,
          0                    , 0                        , std_radrd_ * std_radrd_;

  MatrixXd S (n_z, n_z);
  GetSMatrix(n_z, R, Zsig, z_pred, true, S);
  cout<<"RADAR : S = "<<endl;
  cout<<S;
  cout<<endl;
  cout<<endl;


  MatrixXd Tc = GetTMatrix(n_z, Zsig, z_pred, true);
  cout<<"RADAR : T = "<<endl;
  cout<<Tc;
  cout<<endl;
  cout<<endl;

  //Kalman gain K;
  MatrixXd K = Tc * S.inverse();
  cout<<"RADAR : K = "<<endl;
  cout<<K;
  cout<<endl;
  cout<<endl;

  //residual
  VectorXd z_diff = meas_package.raw_measurements_ - z_pred;
  //angle normalization
  z_diff(1) = NormalizeAngle(z_diff(1));
  cout<<"RADAR : z_diff = "<<endl;
  cout<<z_diff;
  cout<<endl;
  cout<<endl;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;

  //angle normalization
  x_(3) = NormalizeAngle(x_(3));
  cout<<"RADAR : x after update : "<<endl;
  cout<<x_;
  cout<<endl;
  cout<<endl;

  P_ = P_ - K*S*K.transpose();
  cout<<"RADAR : P after update : "<<endl;
  cout<<P_;
  cout<<endl;
  cout<<endl;

  NIS_radar_ = z_diff.transpose() * S.inverse() * z_diff;
  radar_count_++;
  cout<<"RADAR : NIS = "<<NIS_radar_<<endl<<endl;

}
