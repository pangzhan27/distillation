#include <ctime>
#include <string>
#include <fstream>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "caffe/tool/addtool.hpp"
#include "caffe/tool/cnnwrapper.hpp"

using caffe::CnnWrapper;

void PrintUsage(char* argv0) {
  std::cout << "Usage (Single Mode): \n" << argv0
            << " deploy.prototxt caffe.model image.jpg 61x21" << std::endl;
  std::cout << "Usage (Batch Mode): \n" << argv0
            << " deploy.prototxt caffe.model sample/folder/ list.txt 61x21"
            << std::endl;
}

// change the size from a string form "61x21" to a Size form Size(61, 21)
cv::Size GetSizeFromString(const string& size_str) {
  std::stringstream ss(size_str);
  int width, height;
  char x;
  ss >> width >> x >> height;
  return cv::Size(width, height);
}

void TestSingle(const string& model_file, const string& weight_file,
                const string& sample_file, const cv::Size& input_size) {
  // Create cnn model
  CnnWrapper cnn(model_file, weight_file);
  // Load sample by sample file path
  cv::Mat sample = LoadSample(sample_file);
  // Extract patches from sample and normalize
  vector<cv::Mat> patches = NormalizeSampleForVote(sample, input_size);
  // Classify patches
  const vector<PairVec>& predictions = cnn.Classify(patches);
  // Get the final result by voting
  const PairVec predition = ClassifyByVote(predictions);
  // Print the top 5 results
  const int n = std::min<int>(predition.size(), 5);
  std::cout << "result of sample " << sample_file << std::endl;
  std::cout << "label:\tprobability" << std::endl;
  for (int i = 0; i < n; ++i) {
    int label = predition[i].first;
    float probability = predition[i].second;
    std::cout << label << ":\t" << probability << std::endl;
  }
}

void TestBatch(const string& model_file, const string& weight_file,
               const string& sample_dir, const string& list_file,
               const cv::Size& input_size) {
  int sample_count = 0;
  int correct_count = 0;
  double elapsed_time = 0.0;
  // Create cnn model
  CnnWrapper cnn(model_file, weight_file);
  // open the list file
  std::ifstream infile(list_file.c_str());
  CHECK(infile.is_open()) << "failed to open file: " << list_file;
  // process list file line by line
  string name;
  int label;
  double start = std::clock();
  while (infile >> name >> label) {
    // Load sample by sample file path
    cv::Mat sample = LoadSample(sample_dir + name);
    // Extract patches from sample and normalize
    vector<cv::Mat> patches = NormalizeSampleForVote(sample, input_size);
    // Classify patches
    const vector<PairVec>& predictions = cnn.Classify(patches);
    // Get the final result by voting
    const PairVec predition = ClassifyByVote(predictions);
    sample_count += 1;
    correct_count += (predition[0].first == label);
    if(predition[0].first==12)
    {
        correct_count += (predition[1].first == label);
    }
    const int batch_size = 1000;
    if (sample_count % batch_size == 0) {
      double elapsed_time_batch = std::clock() - start;
      double accuracy = double(correct_count) / sample_count;
      double average_time = elapsed_time_batch / batch_size;
      std::cout << "Accuracy: " << correct_count << "/" << sample_count
                << " = " << accuracy << ", average time: " << average_time
                << " ms" << std::endl;
      elapsed_time += elapsed_time_batch;
      start = std::clock();
    }
  }
  infile.close();

  double accuracy = double(correct_count) / sample_count;
  double average_time = elapsed_time / sample_count;
  std::cout << "\nFinal result: \n";
  std::cout << "Accuracy: " << correct_count << "/" << sample_count
            << " = " << accuracy << ", average time: " << average_time
            << " ms" << std::endl;
}

int main(int argc, char* argv[]) {
  if (argc == 5) {
    TestSingle(argv[1], argv[2], argv[3], GetSizeFromString(argv[4]));
  } else if (argc == 6) {
    TestBatch(argv[1], argv[2], argv[3], argv[4], GetSizeFromString(argv[5]));
  } else {
    PrintUsage(argv[0]);
  }
  return 0;
}
