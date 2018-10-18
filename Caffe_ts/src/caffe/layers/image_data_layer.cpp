#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/tool/addtool.hpp"

namespace caffe {

template <typename Dtype>
ImageDataLayer<Dtype>::~ImageDataLayer<Dtype>() {
  this->StopInternalThread();
  delete db_;
}

template <typename Dtype>
void ImageDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int new_height = this->layer_param_.image_data_param().new_height();
  const int new_width  = this->layer_param_.image_data_param().new_width();
  CHECK(new_height > 0 && new_width > 0) << "Current implementation requires "
      "new_height and new_width to be set at the same time.";
  const int batch_size = this->layer_param_.image_data_param().batch_size();
  CHECK_GT(batch_size, 0) << "Positive batch size required";
  string root_folder = this->layer_param_.image_data_param().root_folder();

  // Read the file with filenames and labels
  const string& source = this->layer_param_.image_data_param().source();
  LOG(INFO) << "Opening file " << source;
  std::ifstream infile(source.c_str());
  string filename;
  int label;
  while (infile >> filename >> label) {
    lines_.push_back(std::make_pair(filename, label));
  }
  LOG(INFO) << "A total of " << lines_.size() << " images.";
  lines_id_ = 0;
  // Open database
  db_ = new ReadOnlyLMDB(root_folder);

  // Check if we would need to randomly shuffle data
  if (this->layer_param_.image_data_param().shuffle()) {
    LOG(INFO) << "Shuffling data";
    const unsigned int prefetch_rng_seed = caffe_rng_rand();
    prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
    ShuffleImages();
  }

  // Check if we would need to randomly skip a few data points
  if (this->layer_param_.image_data_param().rand_skip()) {
    unsigned int skip = caffe_rng_rand() %
        this->layer_param_.image_data_param().rand_skip();
    LOG(INFO) << "Skipping first " << skip << " data points.";
    CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
    lines_id_ = skip;
  }

  // Read an image, and use it to initialize the top blob.
  cv::Mat sample = LoadSample(lines_[lines_id_].first, db_);
  sample = NormalizeSampleForTest(sample, cv::Size(new_width, new_height));
  const int top_shape_tmp[] = {batch_size, sample.channels(), sample.rows,
                               sample.cols};
  vector<int> top_shape(top_shape_tmp, top_shape_tmp + 4);
  // Reshape data
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->data_.Reshape(top_shape);
  }
  top[0]->Reshape(top_shape);
  LOG(INFO) << "output data size: " << top[0]->num() << ","
      << top[0]->channels() << "," << top[0]->height() << ","
      << top[0]->width();
  // Reshape label
  vector<int> label_shape(1, batch_size);
  top[1]->Reshape(label_shape);
  for (int i = 0; i < this->prefetch_.size(); ++i) {
    this->prefetch_[i]->label_.Reshape(label_shape);
  }
}

template <typename Dtype>
void ImageDataLayer<Dtype>::ShuffleImages() {
  caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rng_->generator());
  shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
  CPUTimer batch_timer;
  batch_timer.Start();
  double read_time = 0;
  double trans_time = 0;
  CPUTimer timer;
  CHECK(batch->data_.count());
  ImageDataParameter image_data_param = this->layer_param_.image_data_param();
  const int batch_size = image_data_param.batch_size();
  const int new_height = image_data_param.new_height();
  const int new_width = image_data_param.new_width();
  string root_folder = image_data_param.root_folder();

  Dtype* prefetch_data = batch->data_.mutable_cpu_data();
  Dtype* prefetch_label = batch->label_.mutable_cpu_data();

  // datum scales
  const int lines_size = lines_.size();
  const cv::Size target_size(new_width, new_height);
  for (int item_id = 0; item_id < batch_size; ++item_id) {
    // Read a sample
    timer.Start();
    CHECK_GT(lines_size, lines_id_);
    cv::Mat sample = LoadSample(lines_[lines_id_].first, db_);
    read_time += timer.MicroSeconds();
    // Apply transformations
    timer.Start();
    if (this->phase_ == TRAIN) {
      sample = NormalizeSampleForTraining(sample, target_size);
    } else if (this->phase_ == TEST) {
      sample = NormalizeSampleForTest(sample, target_size);
    } else {
      LOG(FATAL) << "Unknown phase.";
    }
    trans_time += timer.MicroSeconds();
    // Set the data
    const int rows = sample.rows;
    const int cols = sample.cols;
    const int channels = sample.channels();
    Dtype* dst_data = prefetch_data + item_id * channels * rows * cols;
    for (int i = 0; i < rows; ++i) {
      const float* src_data = (float*) sample.ptr(i);
      for (int j = 0; j < cols; ++j) {
        for (int k = 0; k < channels; ++k) {
          dst_data[(k * rows + i) * cols + j] = src_data[j * channels + k];
        }
      }
    }
    // Set the label
    prefetch_label[item_id] = lines_[lines_id_].second;
    // go to the next iter
    lines_id_++;
    if (lines_id_ >= lines_size) {
      // We have reached the end. Restart from the first.
      DLOG(INFO) << "Restarting data prefetching from start.";
      lines_id_ = 0;
      if (this->layer_param_.image_data_param().shuffle()) {
        ShuffleImages();
      }
    }
  }
  batch_timer.Stop();
  DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
  DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
  DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageDataLayer);
REGISTER_LAYER_CLASS(ImageData);

}  // namespace caffe
#endif  // USE_OPENCV
