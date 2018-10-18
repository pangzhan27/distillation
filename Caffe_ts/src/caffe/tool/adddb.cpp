#include <vector>
#include <fstream>

#include "caffe/tool/adddb.hpp"
#include "caffe/util/benchmark.hpp"

namespace caffe {

ReadOnlyLevelDB::ReadOnlyLevelDB(const string& source) {
  leveldb::Options options;
  options.block_size = 65536;
  options.max_open_files = 100;
  options.error_if_exists = false;
  options.create_if_missing = false;
  leveldb::Status status = leveldb::DB::Open(options, source, &db_);
  CHECK(status.ok()) << "Failed to open leveldb " << source
                     << std::endl << status.ToString();
  LOG(INFO) << "Opened leveldb " << source;
}

string ReadOnlyLevelDB::GetValue(const string& key) {
  string value;
  leveldb::Status status = db_->Get(leveldb::ReadOptions(), key, &value);
  CHECK(status.ok()) << "Failed to read value of key " << key
                     << std::endl << status.ToString();
  return value;
}

void ReadOnlyLevelDB::SaveKeysToFile(const string& filename) {
  std::ofstream outfile(filename.c_str());
  CHECK(outfile.is_open()) << "failed to open file: " << filename;
  leveldb::Iterator* it = db_->NewIterator(leveldb::ReadOptions());
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    outfile << it->key().ToString() << std::endl;
  }
  delete it;
  outfile.close();
}

double ReadOnlyLevelDB::BandwidthSequential() {
  vector<string> keys;
  leveldb::Iterator* it = db_->NewIterator(leveldb::ReadOptions());
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    keys.push_back(it->key().ToString());
  }
  LOG(INFO) << "A total of " << keys.size() << " keys.";
  delete it;
  string value;
  CPUTimer timer;
  timer.Start();
  for (int i = 0; i < keys.size(); ++i) {
    leveldb::Status status;
    status = db_->Get(leveldb::ReadOptions(), keys[i], &value);
    CHECK(status.ok()) << "Failed to read value of key " << keys[i]
                       << std::endl << status.ToString();
  }
  return timer.MilliSeconds() * 1000000.0 / keys.size();
}

double ReadOnlyLevelDB::BandwidthRandom() {
  vector<string> keys;
  leveldb::Iterator* it = db_->NewIterator(leveldb::ReadOptions());
  for (it->SeekToFirst(); it->Valid(); it->Next()) {
    keys.push_back(it->key().ToString());
  }
  LOG(INFO) << "A total of " << keys.size() << " keys.";
  delete it;
  srand(time(NULL));
  std::random_shuffle(keys.begin(), keys.end());
  string value;
  CPUTimer timer;
  timer.Start();
  for (int i = 0; i < keys.size(); ++i) {
    leveldb::Status status;
    status = db_->Get(leveldb::ReadOptions(), keys[i], &value);
    CHECK(status.ok()) << "Failed to read value of key " << keys[i]
                       << std::endl << status.ToString();
  }
  return timer.MilliSeconds() * 1000000.0 / keys.size();
}

////////////////////////// LMDB ////////////////////////

#define MDB_CHECK(status)                                          \
  do {                                                             \
    int mdb_status = status;                                       \
    CHECK_EQ(mdb_status, MDB_SUCCESS) << mdb_strerror(mdb_status); \
  } while (0)

ReadOnlyLMDB::ReadOnlyLMDB(const string& source) {
  const size_t LMDB_MAP_SIZE = 1099511627776;  // 1 TB
  MDB_CHECK(mdb_env_create(&mdb_env_));
  MDB_CHECK(mdb_env_set_mapsize(mdb_env_, LMDB_MAP_SIZE));
  int flags = MDB_RDONLY | MDB_NOTLS;
  MDB_CHECK(mdb_env_open(mdb_env_, source.c_str(), flags, 0664));
  LOG(INFO) << "Opened lmdb " << source;
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn_));
  MDB_CHECK(mdb_dbi_open(mdb_txn_, NULL, 0, &mdb_dbi_));
}

ReadOnlyLMDB::~ReadOnlyLMDB() {
  mdb_dbi_close(mdb_env_, mdb_dbi_);
  mdb_txn_abort(mdb_txn_);
  mdb_env_close(mdb_env_);
}

string ReadOnlyLMDB::GetValue(const string& key) {
  MDB_val mdb_key, mdb_value;
  mdb_key.mv_data = const_cast<char*>(&(key[0]));
  mdb_key.mv_size = key.size();
  MDB_CHECK(mdb_get(mdb_txn_, mdb_dbi_, &mdb_key, &mdb_value));
  return string(static_cast<char*>(mdb_value.mv_data), mdb_value.mv_size);
}

void ReadOnlyLMDB::SaveKeysToFile(const string& filename) {
  std::ofstream outfile(filename.c_str());
  CHECK(outfile.is_open()) << "failed to open file: " << filename;
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn));
  MDB_CHECK(mdb_dbi_open(mdb_txn, NULL, 0, &mdb_dbi_));
  MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
  MDB_val key, value;
  while (mdb_cursor_get(mdb_cursor, &key, &value, MDB_NEXT) == 0) {
    outfile << static_cast<char*>(key.mv_data) << std::endl;
  }
  mdb_cursor_close(mdb_cursor);
  mdb_txn_abort(mdb_txn);
  outfile.close();
}

double ReadOnlyLMDB::BandwidthSequential() {
  vector<string> keys;
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn));
  MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
  MDB_val mdb_key, mdb_value;
  while (mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT) == 0) {
    string str(static_cast<char*>(mdb_key.mv_data), mdb_key.mv_size);
    keys.push_back(str);
  }
  LOG(INFO) << "A total of " << keys.size() << " keys.";
  mdb_cursor_close(mdb_cursor);
  mdb_txn_abort(mdb_txn);
  CPUTimer timer;
  timer.Start();
  for (int i = 0; i < keys.size(); ++i) {
    mdb_key.mv_data = const_cast<char*>(&(keys[i][0]));
    mdb_key.mv_size = keys[i].size();
    MDB_CHECK(mdb_get(mdb_txn_, mdb_dbi_, &mdb_key, &mdb_value));
  }
  timer.Stop();
  return timer.MilliSeconds() * 1000000.0 / keys.size();
}

double ReadOnlyLMDB::BandwidthRandom() {
  vector<string> keys;
  MDB_txn* mdb_txn;
  MDB_cursor* mdb_cursor;
  MDB_CHECK(mdb_txn_begin(mdb_env_, NULL, MDB_RDONLY, &mdb_txn));
  MDB_CHECK(mdb_cursor_open(mdb_txn, mdb_dbi_, &mdb_cursor));
  MDB_val mdb_key, mdb_value;
  while (mdb_cursor_get(mdb_cursor, &mdb_key, &mdb_value, MDB_NEXT) == 0) {
    string str(static_cast<char*>(mdb_key.mv_data), mdb_key.mv_size);
    keys.push_back(str);
  }
  LOG(INFO) << "A total of " << keys.size() << " keys.";
  mdb_cursor_close(mdb_cursor);
  mdb_txn_abort(mdb_txn);
  srand(time(NULL));
  std::random_shuffle(keys.begin(), keys.end());
  CPUTimer timer;
  timer.Start();
  for (int i = 0; i < keys.size(); ++i) {
    mdb_key.mv_data = const_cast<char*>(&(keys[i][0]));
    mdb_key.mv_size = keys[i].size();
    MDB_CHECK(mdb_get(mdb_txn_, mdb_dbi_, &mdb_key, &mdb_value));
  }
  timer.Stop();
  return timer.MilliSeconds() * 1000000.0 / keys.size();
}

}  // namespace caffe
