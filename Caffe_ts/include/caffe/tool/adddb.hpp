#ifndef CAFFE_TOOL_ADDDB_HPP_
#define CAFFE_TOOL_ADDDB_HPP_

#include <string>
#include <lmdb.h>
#include <leveldb/db.h>
#include "caffe/common.hpp"

namespace caffe {

using std::string;

class ReadOnlyDB {
 public:
  ReadOnlyDB() { }
  virtual ~ReadOnlyDB() { }
  virtual string GetValue(const string& key) = 0;
  virtual void SaveKeysToFile(const string& filename) = 0;
  virtual double BandwidthSequential() = 0;
  virtual double BandwidthRandom() = 0;
  DISABLE_COPY_AND_ASSIGN(ReadOnlyDB);
};

class ReadOnlyLevelDB : public ReadOnlyDB {
 public:
  ReadOnlyLevelDB(const string& source);
  virtual ~ReadOnlyLevelDB() { delete db_; }
  virtual string GetValue(const string& key);
  virtual void SaveKeysToFile(const string& filename);
  virtual double BandwidthSequential();
  virtual double BandwidthRandom();
 private:
  leveldb::DB* db_;
  DISABLE_COPY_AND_ASSIGN(ReadOnlyLevelDB);
};

class ReadOnlyLMDB : public ReadOnlyDB {
 public:
  ReadOnlyLMDB(const string& source);
  virtual ~ReadOnlyLMDB();
  virtual string GetValue(const string& key);
  virtual void SaveKeysToFile(const string& filename);
  virtual double BandwidthSequential();
  virtual double BandwidthRandom();
 private:
  MDB_env* mdb_env_;
  MDB_txn* mdb_txn_;
  MDB_dbi mdb_dbi_;
  DISABLE_COPY_AND_ASSIGN(ReadOnlyLMDB);
};

}  // namespace caffe

#endif  // CAFFE_TOOL_ADDDB_HPP_
