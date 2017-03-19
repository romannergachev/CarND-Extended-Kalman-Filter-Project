#include "kalman_filter.h"
#include "tools.h"

using Eigen::MatrixXd;
using Eigen::VectorXd;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}

void KalmanFilter::Update(const VectorXd &z) {
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;
  update(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  double px = x_[0];
  double py = x_[1];
  double vx = x_[2];
  double vy = x_[3];

  double rho = sqrt(px * px + py * py);
  double phi;
  double d_rho;
  if (fabs(px) > SMALL_NUMBER) {
    phi = atan(py / px);
  } else {
    phi = 0;
  }

  double sqrtP = sqrt(px * px + py * py);
  if (fabs(sqrtP) > SMALL_NUMBER) {
    d_rho = (px * vx + py * vy) / sqrtP;
  } else {
    d_rho = 0;
  }

  VectorXd z_pred(3);
  z_pred << rho, phi, d_rho;
  VectorXd y = z - z_pred;

  if (y[1] > M_PI && y[1] < -M_PI) {
    int quantity = static_cast<int>(y[1] / M_PI);
    y[1] -= quantity;
  }

  update(y);

}

void KalmanFilter::update(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd PHt = P_ * Ht;
  MatrixXd K = PHt * Si;

  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}
