#ifndef _CROWDNET_H_
#define _CROWDNET_H_

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core/utils/trace.hpp>
#include <opencv2/core/core.hpp>
using namespace cv;
using namespace cv::dnn;

#include <iostream>
using namespace std;

const String model_txt = "D:\\Win\\Doc\\#Project\\SRTP\\models\\deploy.prototxt";
const String model_bin = "D:\\Win\\Doc\\#Project\\SRTP\\weight\\dcc_crowdnet\\dcc_crowdnet_train_iter_140000.caffemodel";

#endif