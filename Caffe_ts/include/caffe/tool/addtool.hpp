#ifndef CAFFE_TOOL_ADDTOOL_HPP_
#define CAFFE_TOOL_ADDTOOL_HPP_

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/tool/adddb.hpp"

using std::pair;
using std::string;
using std::vector;

// Pair (label, confidence)
typedef pair<int, float> Pair;
typedef vector<Pair> PairVec;

// Load a sample from the file path
cv::Mat LoadSample(const string& filepath);
cv::Mat LoadSample(const string& filepath, caffe::ReadOnlyDB* db);

// Normalize a sample according to target_size. Generally,
// the last step is to subtract the mean of the matrix to be returned.
// Note that this method should return a float-type (CV_32F) matrix.
cv::Mat NormalizeSampleForTraining(const cv::Mat& src,
                                   const cv::Size& target_size);

// Normalize a sample according to target_size. Generally,
// the last step is to subtract the mean of the matrix to be returned.
// Note that this method should return a float-type (CV_32F) matrix.
cv::Mat NormalizeSampleForTest(const cv::Mat& src,
                               const cv::Size& target_size);

// Normalize a sample according to target_size. In some situations,
// you may want to extract several patches from one sample.
// Note that all the matrices returned should be float-type (CV_32F).
vector<cv::Mat> NormalizeSampleForVote(const cv::Mat& src,
                                       const cv::Size& target_size);

// Different voting methods can be implemented here.
const PairVec ClassifyByVote(const vector<PairVec>& preditions);

#endif  // CAFFE_TOOL_ADDTOOL_HPP_
