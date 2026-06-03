#ifndef SLAM_UTIL_SQLITE3_H
#define SLAM_UTIL_SQLITE3_H

#include <vector>
#include <string>

typedef struct sqlite3 sqlite3;
typedef struct sqlite3_stmt sqlite3_stmt;

namespace cv::slam {
namespace util {
namespace sqlite3_util {

bool create_table(sqlite3* db,
                  const std::string& name,
                  const std::vector<std::pair<std::string, std::string>>& columns);
bool drop_table(sqlite3* db,
                const std::string& name);
bool begin(sqlite3* db);
bool next(sqlite3* db, sqlite3_stmt* stmt);
bool commit(sqlite3* db);
sqlite3_stmt* create_select_stmt(sqlite3* db, const std::string& table_name);
sqlite3_stmt* create_insert_stmt(sqlite3* db,
                                 const std::string& name,
                                 const std::vector<std::pair<std::string, std::string>>& columns);

} // namespace sqlite3_util
} // namespace util
} // namespace cv::slam

#endif // SLAM_UTIL_SQLITE3_H
