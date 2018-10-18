#include <string>
#include <iostream>

#include "caffe/tool/adddb.hpp"

using std::string;

int main(int argc, char* argv[]) {
  caffe::GlobalInit(&argc, &argv);
  google::LogToStderr();
  if (argc != 3) {
    std::cout << "Usage: " << argv[0] << " dbtpye dbname" << std::endl;
    std::cout << "\tdbtype can be leveldb or lmdb" << std::endl;
    std::cout << "\tdbname is the path of database location." << std::endl;
    return 1;
  }
  caffe::ReadOnlyDB* db = NULL;
  if (string(argv[1]) == "leveldb") {
    db = new caffe::ReadOnlyLevelDB(argv[2]);
  } else if (string(argv[1]) == "lmdb") {
    db = new caffe::ReadOnlyLMDB(argv[2]);
  } else {
    LOG(FATAL) << "Unknown DB type " << argv[1];
  }
  std::cout << "Read performance (Sequential): "
            << db->BandwidthSequential() << "ms/MS." << std::endl;
  std::cout << "Read performance (random): "
            << db->BandwidthRandom() << "ms/MS." << std::endl;
  delete db;
  return 0;
}
