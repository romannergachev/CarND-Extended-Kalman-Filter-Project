#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0,0,0,0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  // ... your code here
  if(estimations.size() == 0 || estimations.size() != ground_truth.size()) {
    return rmse;
  }

  //accumulate squared residuals
  for(int i=0; i < estimations.size(); ++i) {
    VectorXd delta = estimations[i] - ground_truth[i];
    delta = delta.array() * delta.array();
    rmse += delta;
  }

  //calculate the mean
  rmse /= estimations.size();

  //calculate the squared root
  rmse = rmse.array().sqrt();

  //return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  MatrixXd Hj(3,4);
  //recover state parameters
  double px = x_state(0);
  double py = x_state(1);
  double vx = x_state(2);
  double vy = x_state(3);

  double squares = pow(px, 2) + pow(py, 2);
  double root = sqrt(squares);
  double root3 = squares*root;

  if (squares > 0.0001) {
    Hj << px/root, py/root, 0, 0,
      (-1) * py/squares, px/squares, 0, 0,
      py * (vx*py - vy*px)/root3, px * (vy*px - vx*py)/root3, px/root, py/root;
  }


  return Hj;
}
