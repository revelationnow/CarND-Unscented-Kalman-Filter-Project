
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <stdlib.h>
#include "Eigen/Dense"
#include "ukf.h"
#include "ukf_test.h"
#include "ground_truth_package.h"
#include "measurement_package.h"


int main(int argc, char* argv[]) {
  UKF_TEST ukf_test;
  ukf_test.UKF_TEST_TEST_1_Augment_sig();
  ukf_test.UKF_TEST_TEST_2_Predict_sig();
  ukf_test.UKF_TEST_TEST_3_Get_state();
  return 0;
}

