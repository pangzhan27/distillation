#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/tool/cnnwrapper.hpp"

namespace caffe {

struct PairComparator {
  inline bool operator() (const Pair& p1, const Pair& p2) {
    return p1.second > p2.second;
  }
};

CnnWrapper::CnnWrapper(const string& model_file, const string& trained_file) {
#ifdef CPU_ONLY
  Caffe::set_mode(Caffe::CPU);
#else
  Caffe::set_mode(Caffe::GPU);
#endif
  // Load the network.
  net_.reset(new Net<float>(model_file, TEST));
  net_->CopyTrainedLayersFrom(trained_file);
  // Get the shape of input blob.
  Blob<float>* input_blob = net_->input_blobs()[0];
  input_channels_ = input_blob->channels();
  input_height_ = input_blob->height();
  input_width_ = input_blob->width();
}

const Blob<float>& CnnWrapper::Extract(const cv::Mat& sample,
                                       const string& blob_name) {
  vector<cv::Mat> samples;
  samples.push_back(sample);
  return Extract(samples, blob_name);
}

const Blob<float>& CnnWrapper::Extract(const vector<cv::Mat>& samples,
                                       const string& blob_name) {
  SetInputBlob(samples);
  net_->ForwardPrefilled();
  return get_blob(blob_name);
}

const PairVec& CnnWrapper::Classify(const cv::Mat& sample) {
  vector<cv::Mat> samples;
  samples.push_back(sample);
  return Classify(samples)[0];
}

const vector<PairVec>& CnnWrapper::Classify(const vector<cv::Mat>& samples) {
  CHECK_EQ(net_->output_blobs().size(), 1)
      << "Network should have exactly one output.";
  const Blob<float>* output_blob =  net_->output_blobs()[0];
  CHECK_EQ(output_blob->shape().size(), 2)
      << "network should output 1 dimensional probability distributions.";
  SetInputBlob(samples);
  net_->ForwardPrefilled();
  // Resize predictions
  const int num = output_blob->shape(0);
  const int dim = output_blob->shape(1);
  if (predictions_.size() != num) {
    predictions_.resize(num);
    for (int i = 0; i < num; ++i) {
      predictions_[i].resize(dim);
    }
  }
  // Get the prediction results
  const float* output_data = output_blob->cpu_data();
  for (int i = 0; i < num; ++i) {
    PairVec& pair_vec = predictions_[i];
    for (int j = 0; j < dim; ++j) {
      pair_vec[j].first = j;
      pair_vec[j].second = output_data[i * dim + j];
    }
    std::sort(pair_vec.begin(), pair_vec.end(), PairComparator());
  }
  return predictions_;
}

void CnnWrapper::SetInputBlob(const vector<cv::Mat> samples) {
  CHECK_EQ(net_->input_blobs().size(), 1)
      << "Network should have exactly one input.";
  // Reshape the net according to size of samples
  Blob<float>* input_blob = net_->input_blobs()[0];
  if (samples.size() != input_blob->shape(0)) {
    input_blob->Reshape(samples.size(), input_channels_, input_height_,
                        input_width_);
    net_->Reshape();
  }
  // Fill the input blob
  for (int sample_id = 0; sample_id < samples.size(); ++sample_id) {
    const cv::Mat& sample = samples[sample_id];
    CHECK_EQ(sample.depth(), CV_32F)
        << "sample should have an element type of CV_32F (float).";
    CHECK_EQ(sample.channels(), input_channels_)
        << "sample and the blob of input layer should have the same "
        << "number of channels.";
    CHECK_EQ(sample.rows, input_height_)
        << "sample and the blob of input layer should have the same height.";
    CHECK_EQ(sample.cols, input_width_)
        << "sample and the blob of input layer should have the same width.";
    float* input_data = input_blob->mutable_cpu_data() +
        sample_id * input_channels_ * input_height_ * input_width_;
    for (int i = 0; i < input_height_; ++i) {
      float* row_data = (float*) sample.ptr(i);
      for (int j = 0; j < input_width_; ++j) {
        for (int k = 0 ; k < input_channels_; ++k) {
          input_data[(k * input_height_ + i) * input_width_ + j] =
              row_data[j * input_channels_ + k];
        }
      }
    }
  }
}

} // namespace caffe
