#pragma once
#include "string"
#include "opencv2/opencv.hpp"
#include <opencv2/dnn.hpp>
using namespace cv::dnn;
using namespace std;
const string caffeplatedir = "../";
const string model_file = caffeplatedir + "/modeldef/deploy.prototxt";
const string trained_file = caffeplatedir + "/plate996.caffemodel";
const string mean_file = caffeplatedir + "/modeldef/mean.binaryproto";

class CLenetClassifier
{
public:
	static CLenetClassifier*getInstance()
	{
		static CLenetClassifier instance;
		return &instance;
	}
	std::pair<int, double>predict(const cv::Mat &img);
	bool load(cv::String modelTxt = model_file, cv::String modelBin = trained_file);
private:
	bool bloaded = false;
    Net _net;
    cv::Scalar _mean;
	CLenetClassifier() {
	}
};