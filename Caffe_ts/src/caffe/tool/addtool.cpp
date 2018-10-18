#include <glog/logging.h>
#include <iostream>
#include "caffe/proto/caffe.pb.h"
#include "caffe/tool/addtool.hpp"

cv::Mat LoadSample(const string& filepath) {
  cv::Mat image = cv::imread(filepath, 0);
  CHECK(!image.empty()) << "failed to load file: " << filepath;
  return image;
}

cv::Mat LoadSample(const string& filepath, caffe::ReadOnlyDB* db) {
  caffe::Datum datum;
  datum.ParseFromString(db->GetValue(filepath));
  CHECK_EQ(datum.channels(), 1);
  cv::Mat image(datum.height(), datum.width(), CV_8UC1);
  const uchar* datum_data = (uchar*)(&datum.data()[0]);
  for (int i = 0; i < datum.height(); ++i) {
    uchar* row_data = (uchar*) image.ptr(i);
    for (int j = 0; j < datum.width(); ++j) {
      row_data[j] = datum_data[i * datum.width() + j];
    }
  }
  return image;
}

cv::Mat NormalizeSampleForTraining(const cv::Mat& src,
                                   const cv::Size& target_size) {
  CHECK_EQ(src.channels(), 1) << "Gray scale image required.";
  // reverse gray scale
  cv::Mat image = 255 - src;
  // resize the image (keeping aspect ratio)
  int width = target_size.width;
  int height = width * src.rows / src.cols;
  if (height > target_size.height) {
    height = target_size.height;
    width = height * src.cols / src.rows;
  }
  if (width < 1) width = 1;
  if (height < 1) height = 1;
  cv::Mat resized_image;
  cv::resize(image, resized_image, cv::Size(width, height));
  // padding
  int offset_x = (target_size.width - width) / 2;
  int offset_y = (target_size.height - height) / 2;
  cv::Rect roi(offset_x, offset_y, width, height);
  cv::Mat padded_image(target_size, CV_8UC1);
  padded_image.setTo(0);
  resized_image.copyTo(padded_image(roi));
  // Mean subtraction
  cv::Mat float_image;
  double mean = cv::mean(padded_image).val[0];
  padded_image.convertTo(float_image, CV_32F, 1.0, -mean);
  return float_image;
}

cv::Mat NormalizeSampleForTest(const cv::Mat& src,
                               const cv::Size& target_size) {
  return NormalizeSampleForTraining(src, target_size);
}

vector<cv::Mat> NormalizeSampleForVote(const cv::Mat& src,
                                       const cv::Size& target_size) {
  cv::Mat sample = NormalizeSampleForTest(src, target_size);
  return vector<cv::Mat>(1, sample);
}

const PairVec ClassifyByVote(const vector<PairVec>& preditions) {
  return preditions[0];
}
