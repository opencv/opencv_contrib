#include "util/sqlite3.hpp"
#include <sqlite3.h>
#include <opencv2/core/utils/logger.hpp>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace util {
namespace sqlite3_util {

bool create_table(sqlite3* db,
                  const std::string& name,
                  const std::vector<std::pair<std::string, std::string>>& columns) {
    int ret = SQLITE_ERROR;
    std::string create_stmt_str = "CREATE TABLE " + name + "(id INTEGER PRIMARY KEY";
    for (const auto& column : columns) {
        create_stmt_str += ", " + column.first + " " + column.second;
    }
    create_stmt_str += ");";
    ret = sqlite3_exec(db, create_stmt_str.c_str(), nullptr, nullptr, nullptr);
    if (ret != SQLITE_OK) {
        CV_LOG_ERROR(&g_log_tag, "SQLite error (create_table): " << sqlite3_errmsg(db));
    }
    return ret == SQLITE_OK;
}

bool begin(sqlite3* db) {
    int ret = SQLITE_ERROR;
    ret = sqlite3_exec(db, "BEGIN;", nullptr, nullptr, nullptr);
    if (ret != SQLITE_OK) {
        CV_LOG_ERROR(&g_log_tag, "SQLite error (begin): " << sqlite3_errmsg(db));
    }
    return ret == SQLITE_OK;
}

bool next(sqlite3* db, sqlite3_stmt* stmt) {
    int ret = sqlite3_step(stmt);
    if (ret != SQLITE_DONE) {
        CV_LOG_ERROR(&g_log_tag, "SQLite step is not done: " << sqlite3_errmsg(db));
        return false;
    }

    sqlite3_reset(stmt);
    sqlite3_clear_bindings(stmt);
    return true;
}

bool commit(sqlite3* db) {
    int ret = SQLITE_ERROR;
    ret = sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);
    if (ret != SQLITE_OK) {
        CV_LOG_ERROR(&g_log_tag, "SQLite error (commit): " << sqlite3_errmsg(db));
    }
    return ret == SQLITE_OK;
}

bool drop_table(sqlite3* db,
                const std::string& name) {
    const std::string stmt_str = "DROP TABLE IF EXISTS " + name + ";";
    int ret = sqlite3_exec(db, stmt_str.c_str(), nullptr, nullptr, nullptr);
    return ret == SQLITE_OK;
}

sqlite3_stmt* create_select_stmt(sqlite3* db, const std::string& table_name) {
    sqlite3_stmt* stmt;
    const std::string stmt_str = "SELECT * FROM " + table_name + ";";
    int ret = sqlite3_prepare_v2(db, stmt_str.c_str(), -1, &stmt, nullptr);
    if (ret != SQLITE_OK) {
        CV_LOG_ERROR(&g_log_tag, "SQLite error: " << sqlite3_errmsg(db));
        return nullptr;
    }
    return stmt;
}

sqlite3_stmt* create_insert_stmt(sqlite3* db,
                                 const std::string& name,
                                 const std::vector<std::pair<std::string, std::string>>& columns) {
    sqlite3_stmt* stmt = nullptr;
    std::string insert_stmt_str = "INSERT INTO " + name + "(id";
    for (const auto& column : columns) {
        insert_stmt_str += ", " + column.first;
    }
    insert_stmt_str += ") VALUES(?";
    for (size_t i = 0; i < columns.size(); ++i) {
        insert_stmt_str += ", ?";
    }
    insert_stmt_str += ")";
    int ret = sqlite3_prepare_v2(db, insert_stmt_str.c_str(), -1, &stmt, nullptr);
    if (!stmt || ret != SQLITE_OK) {
        CV_LOG_ERROR(&g_log_tag, "SQLite error (prepare): " << sqlite3_errmsg(db));
        return nullptr;
    }
    return stmt;
}
} // namespace sqlite3_util
} // namespace util
} // namespace cv::slam
