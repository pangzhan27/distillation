#ifndef CAFFE_TOOL_CNNWRAPPER_HPP_
#define CAFFE_TOOL_CNNWRAPPER_HPP_

#include <string>

#include "caffe/caffe.hpp"
#include "caffe/tool/addtool.hpp"

using std::string;
using std::vector;
using boost::shared_ptr;

namespace caffe {

class CnnWrapper {
 public:
  CnnWrapper(const string& model_file, const string& trained_file);
  CnnWrapper() { }

  const Blob<float>& get_blob(const string& blob_name) {
    return *(net_->blob_by_name(blob_name));
  }
  const Blob<float>& Extract(const cv::Mat& sample, const string& blob_name);
  const Blob<float>& Extract(const vector<cv::Mat>& samples,
                             const string& blob_name);
  const PairVec& Classify(const cv::Mat& sample);
  const vector<PairVec>& Classify(const vector<cv::Mat>& samples);

 private:
  void SetInputBlob(const vector<cv::Mat> samples);

  shared_ptr<Net<float> > net_;
  vector<PairVec> predictions_;
  int input_channels_;
  int input_height_;
  int input_width_;
};

}  // namespace caffe

#endif  // CAFFE_TOOL_CNNWRAPPER_HPP_
