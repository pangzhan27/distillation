#include <ctime>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tool/addtool.hpp"
#include "caffe/tool/cnnwrapper.hpp"

using caffe::CnnWrapper;
using caffe::Caffe;

void PrintUsage(char* argv0) {
  std::cout << "Usage: " << argv0 << " deploy.prototxt weights.caffemodel "
            << "sample/folder/ sample-list.txt 61x21 GPU0 result.txt"
            << std::endl;
}

int GetDeviceId(const string& gpu_info) {
  CHECK_EQ(gpu_info.length(), 4);
  std::stringstream ss(gpu_info.substr(3));
  int gpu_id;
  ss >> gpu_id;
  return gpu_id;
}

// change the size from a string form "61x21" to a Size form Size(61, 21)
cv::Size GetSizeFromString(const string& size_str) {
  std::stringstream ss(size_str);
  int width, height;
  char x;
  ss >> width >> x >> height;
  return cv::Size(width, height);
}

int main(int argc, char* argv[]) {
  if (argc != 8) {
    PrintUsage(argv[0]);
    return -1;
  }
  ::google::InitGoogleLogging(argv[0]);

  // Set GPU ID
  Caffe::SetDevice(GetDeviceId(string(argv[6])));
  Caffe::set_mode(Caffe::GPU);

  // Do classification
  const cv::Size size = GetSizeFromString(argv[5]);
  CnnWrapper cnn(argv[1], argv[2]);
  int confuse_matrix[2][2] = {0};
  const int batch_size = 128;
  vector<cv::Mat> batch_sample;
  vector<int> batch_label;
  string name;
  int label;
  std::ifstream infile(argv[4]);
  int total_count = 0;
  CHECK(infile.is_open()) << "cannot open file: " << argv[4];
  while (infile >> name >> label) {
    cv::Mat sample = LoadSample(string(argv[3]) + name);
    CHECK(!sample.empty()) << "cannot load sample: " << name;
    batch_sample.push_back(NormalizeSampleForTest(sample, size));
    batch_label.push_back(label);
    if (batch_sample.size() == batch_size) {
      const vector<PairVec>& predictions = cnn.Classify(batch_sample);
      for (int i = 0; i < batch_size; ++i) {
        int gtid = batch_label[i];
        int pdid = predictions[i][0].first;
        confuse_matrix[gtid][pdid] += 1;
      }
      batch_sample.clear();
      batch_label.clear();
    }
    if (++total_count % 10000 == 0) {
      std::cout << "have finished " << total_count << " images." << std::endl;
    }
  }
  CHECK_EQ(batch_sample.size(), 0);
  CHECK_EQ(batch_label.size(), 0);
  infile.close();

  std::ofstream outfile(argv[7], std::ios::app);
  CHECK(outfile.is_open()) << "cannot open file: " << argv[7];
  const int cm0 = confuse_matrix[0][0] + confuse_matrix[0][1];
  const int cm1 = confuse_matrix[1][0] + confuse_matrix[1][1];
  double v00 = double(confuse_matrix[0][0]) / cm0;
  double v01 = double(confuse_matrix[0][1]) / cm0;
  double v10 = double(confuse_matrix[1][0]) / cm1;
  double v11 = double(confuse_matrix[1][1]) / cm1;
  double v = double(confuse_matrix[0][0] + confuse_matrix[1][1]) / (cm0 + cm1);
  outfile << argv[2] << "\t" << v00 << "\t" << v01 << "\t"
          << v10 << "\t" << v11 << "\t" << v << std::endl;
  outfile.close();
  return 0;
}
