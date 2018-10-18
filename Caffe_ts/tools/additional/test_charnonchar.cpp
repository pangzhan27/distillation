#include <cmath>
#include <vector>
#include <fstream>
#include <glog/logging.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tool/addtool.hpp"
#include "caffe/tool/cnnwrapper.hpp"

using std::vector;
using caffe::Blob;
using caffe::Caffe;

class TextLineProcessor {
 public:
  TextLineProcessor(const string& model,
                    const string& weights,
                    const cv::Size& input_size)
    : cnn_(model, weights), cnn_input_size_(input_size) {
    memset(confusion_matrix_, 0, sizeof(confusion_matrix_));
  }
  ~TextLineProcessor() { }
  void Process(const string& source, const string& save_dir);

 private:
  int InferNumImages(int cc_count);
  string CreateSavePath(const string& save_dir, const string& sample_path);
  void ProcessTextLine(const string& sample_path, const string& save_dir);
  void SaveResult(const string& save_path, const int cc_count,
                  const Blob<float>& blob);

  caffe::CnnWrapper cnn_;
  cv::Size cnn_input_size_;
  int confusion_matrix_[2][2];
};

void TextLineProcessor::Process(const string& source, const string& save_dir) {
  int count = 0;
  std::ifstream fin(source.c_str());
  CHECK(fin.is_open()) << "failed to open file: " + source;
  string path;
  while (fin >> path) {
    ProcessTextLine(path, save_dir);
    std::cout << count++ << " - finished processing: " << path << std::endl;
  }
  fin.close();
  std::cout << "\n\n------ result -------\n";
  std::cout << "\t\tnonchar(pd)\tchar(pd)\n";
  std::cout << "nonchar(gt)\t\t" << confusion_matrix_[0][0] << "\t\t"
            << confusion_matrix_[0][1] << "\n";
  std::cout << "char(gt)\t\t" << confusion_matrix_[1][0] << "\t\t"
            << confusion_matrix_[1][1] << "\n";
  std::cout << "---------- end -----------";
}

int TextLineProcessor::InferNumImages(int cc_count) {
  CHECK(cc_count > 0) << "invalid cc_count.";
  int num_images = 0;
  if (cc_count == 1) {
    num_images = 1;
  } else if (cc_count == 2) {
    num_images = 3;
  } else if (cc_count == 3) {
    num_images = 6;
  } else {
    num_images = cc_count * 4 - 6;
  }
  return num_images;
}

string TextLineProcessor::CreateSavePath(const string& save_dir,
                                         const string& sample_path) {
  size_t start = sample_path.find_last_of('/') + 1;
  size_t end = sample_path.find_last_of('.');
  return save_dir + "/" + sample_path.substr(start, end - start) + ".conf";
}

void TextLineProcessor::ProcessTextLine(const string& sample_path,
                                        const string& save_dir) {
  std::ifstream fin(sample_path.c_str());
  CHECK(fin.is_open()) << "failed to open file: " << sample_path;
  vector<cv::Mat> batch_samples;
  vector<int> batch_labels;
  int cc_count, label;
  string path;
  fin >> cc_count;
  while (fin >> path >> label) {
    cv::Mat image = cv::imread(path, 0);
    CHECK(!image.empty()) << "failed to load image: " << path;
    image = NormalizeSampleForTest(image, cnn_input_size_);
    batch_samples.push_back(image);
    batch_labels.push_back(label);
  }
  fin.close();
  CHECK_EQ(InferNumImages(cc_count), batch_samples.size())
      << "wrong file: " << sample_path;
  const Blob<float>& prob = cnn_.Extract(batch_samples, "prob");
  const string save_path = CreateSavePath(save_dir, sample_path);
  SaveResult(save_path, cc_count, prob);
  const float* data = prob.cpu_data();
  for (int i = 0; i < batch_labels.size(); ++i) {
    int pid = data[2*i] < data[2*i+1];
    confusion_matrix_[batch_labels[i]][pid] += 1;
  }
}

void TextLineProcessor::SaveResult(const string& save_path,
                                   const int cc_count,
                                   const Blob<float>& blob) {
  CHECK(blob.shape().size() == 2);
  CHECK(blob.shape(1) == 2);
  float* result = new float[cc_count * 4];
  memset(result, -1, sizeof(float) * cc_count * 4);
  const float* data = blob.cpu_data();
  for (int i = 0; i < cc_count; ++i) {
    result[i * 4] = data[(i * 4) * 2 + 1];
    if (i < cc_count - 1) {
      result[i * 4 + 1] = data[(i * 4 + 1) * 2 + 1];
    }
    if (i < cc_count - 2) {
      result[i * 4 + 2] = data[(i * 4 + 2) * 2 + 1];
    }
    if (i < cc_count - 3) {
      result[i * 4 + 3] = data[(i * 4 + 3) * 2 + 1];
    }
  }
  std::ofstream out(save_path.c_str(), std::ios::binary);
  CHECK(out.is_open()) << "failed to open file: " << save_path;
  int num_floats = cc_count * 4;
  out.write((char*)(&num_floats), sizeof(int));
  out.write((char*)(result), sizeof(float)*num_floats);
  out.close();
  delete[] result;
}

void PrintUsage(char* argv0) {
  std::cout << "Usage: " << argv0 << " deploy.prototxt weights.caffemodel "
            << "sample-list.txt save/dir/ 50x50 GPU1" << std::endl;
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
  if (argc != 7) {
    PrintUsage(argv[0]);
    return -1;
  }
  ::google::InitGoogleLogging(argv[0]);

  // Set GPU ID
  Caffe::SetDevice(GetDeviceId(string(argv[6])));
  Caffe::set_mode(Caffe::GPU);

  const cv::Size size = GetSizeFromString(argv[5]);
  TextLineProcessor processor(argv[1], argv[2], size);
  processor.Process(argv[3], argv[4]);
  std::cout << "\ndone!" << std::endl;
  return 0;
}
