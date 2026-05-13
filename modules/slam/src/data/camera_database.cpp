#include "camera/base.hpp"
#include "camera/perspective.hpp"
#include "camera/fisheye.hpp"
#include "camera/equirectangular.hpp"
#include "camera/radial_division.hpp"
#include "data/camera_database.hpp"

#include <opencv2/core/utils/logger.hpp>
#include <nlohmann/json.hpp>
#ifdef USE_SQLITE3
#include <sqlite3.h>
#endif

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace data {

camera_database::camera_database() {
    CV_LOG_DEBUG(&g_log_tag, "CONSTRUCT: data::camera_database");
}

camera_database::~camera_database() {
    for (const auto& name_camera : cameras_) {
        const auto& camera_name = name_camera.first;
        delete cameras_.at(camera_name);
        cameras_.at(camera_name) = nullptr;
    }
    cameras_.clear();

    CV_LOG_DEBUG(&g_log_tag, "DESTRUCT: data::camera_database");
}

void camera_database::add_camera(camera::base* camera) {
    std::lock_guard<std::mutex> lock(mtx_database_);
    assert(cameras_.count(camera->name_) == 0);
    cameras_.emplace(camera->name_, camera);
}

camera::base* camera_database::get_camera(const std::string& camera_name) const {
    std::lock_guard<std::mutex> lock(mtx_database_);
    if (cameras_.count(camera_name)) {
        return cameras_.at(camera_name);
    }
    else {
        return nullptr;
    }
}

void camera_database::from_json(const nlohmann::json& json_cameras) {
    std::lock_guard<std::mutex> lock(mtx_database_);

    CV_LOG_INFO(&g_log_tag, "decoding " << json_cameras.size() << " camera(s) to load");
    for (const auto& json_id_camera : json_cameras.items()) {
        const auto& camera_name = json_id_camera.key();
        const auto& json_camera = json_id_camera.value();

        if (cameras_.count(camera_name)) {
            CV_LOG_INFO(&g_log_tag, "A camera with the same name (\"" << camera_name << "\") already existed in database.");
            continue;
        }

        CV_LOG_INFO(&g_log_tag, "load a camera \"" << camera_name << "\" from JSON");
        camera::base* camera = nullptr;
        const auto setup_type = camera::base::load_setup_type(json_camera.at("setup_type").get<std::string>());
        const auto model_type = camera::base::load_model_type(json_camera.at("model_type").get<std::string>());
        const auto color_order = camera::base::load_color_order(json_camera.at("color_order").get<std::string>());

        switch (model_type) {
            case camera::model_type_t::Perspective: {
                camera = new camera::perspective(camera_name, setup_type, color_order,
                                                 json_camera.at("cols").get<unsigned int>(),
                                                 json_camera.at("rows").get<unsigned int>(),
                                                 json_camera.at("fps").get<double>(),
                                                 json_camera.at("fx").get<double>(),
                                                 json_camera.at("fy").get<double>(),
                                                 json_camera.at("cx").get<double>(),
                                                 json_camera.at("cy").get<double>(),
                                                 json_camera.at("k1").get<double>(),
                                                 json_camera.at("k2").get<double>(),
                                                 json_camera.at("p1").get<double>(),
                                                 json_camera.at("p2").get<double>(),
                                                 json_camera.at("k3").get<double>(),
                                                 json_camera.at("focal_x_baseline").get<double>());
                break;
            }
            case camera::model_type_t::Fisheye: {
                camera = new camera::fisheye(camera_name, setup_type, color_order,
                                             json_camera.at("cols").get<unsigned int>(),
                                             json_camera.at("rows").get<unsigned int>(),
                                             json_camera.at("fps").get<double>(),
                                             json_camera.at("fx").get<double>(),
                                             json_camera.at("fy").get<double>(),
                                             json_camera.at("cx").get<double>(),
                                             json_camera.at("cy").get<double>(),
                                             json_camera.at("k1").get<double>(),
                                             json_camera.at("k2").get<double>(),
                                             json_camera.at("k3").get<double>(),
                                             json_camera.at("k4").get<double>(),
                                             json_camera.at("focal_x_baseline").get<double>());
                break;
            }
            case camera::model_type_t::Equirectangular: {
                camera = new camera::equirectangular(camera_name, color_order,
                                                     json_camera.at("cols").get<unsigned int>(),
                                                     json_camera.at("rows").get<unsigned int>(),
                                                     json_camera.at("fps").get<double>());
                break;
            }
            case camera::model_type_t::RadialDivision: {
                camera = new camera::radial_division(camera_name, setup_type, color_order,
                                                     json_camera.at("cols").get<unsigned int>(),
                                                     json_camera.at("rows").get<unsigned int>(),
                                                     json_camera.at("fps").get<double>(),
                                                     json_camera.at("fx").get<double>(),
                                                     json_camera.at("fy").get<double>(),
                                                     json_camera.at("cx").get<double>(),
                                                     json_camera.at("cy").get<double>(),
                                                     json_camera.at("distortion").get<double>(),
                                                     json_camera.at("focal_x_baseline").get<double>());
                break;
            }
        }

        assert(!cameras_.count(camera_name));
        cameras_[camera_name] = camera;
    }
}

nlohmann::json camera_database::to_json() const {
    std::lock_guard<std::mutex> lock(mtx_database_);

    CV_LOG_INFO(&g_log_tag, "encoding " << cameras_.size() << " camera(s) to store");
    std::map<std::string, nlohmann::json> cameras;
    for (const auto& name_camera : cameras_) {
        const auto& camera_name = name_camera.first;
        const auto camera = name_camera.second;
        cameras[camera_name] = camera->to_json();
    }
    return cameras;
}

#ifdef USE_SQLITE3
bool camera_database::from_db(sqlite3* db) {
    std::lock_guard<std::mutex> lock(mtx_database_);

    sqlite3_stmt* stmt;
    int ret = sqlite3_prepare_v2(db, "SELECT * FROM cameras;", -1, &stmt, nullptr);
    if (ret != SQLITE_OK) {
        CV_LOG_ERROR(&g_log_tag, "SQLite error: " << sqlite3_errmsg(db));
        return false;
    }

    while ((ret = sqlite3_step(stmt)) == SQLITE_ROW) {
        auto p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, 1));
        std::string camera_name(p, p + sqlite3_column_bytes(stmt, 1));

        if (cameras_.count(camera_name)) {
            CV_LOG_INFO(&g_log_tag, "A camera with the same name (\"" << camera_name << "\") already existed in database.");
            continue;
        }

        p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, 2));
        auto setup_type = camera::base::load_setup_type(std::string(p, p + sqlite3_column_bytes(stmt, 2)));
        p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, 3));
        auto model_type = camera::base::load_model_type(std::string(p, p + sqlite3_column_bytes(stmt, 3)));
        p = reinterpret_cast<const char*>(sqlite3_column_blob(stmt, 4));
        auto color_order = camera::base::load_color_order(std::string(p, p + sqlite3_column_bytes(stmt, 4)));
        auto cols = sqlite3_column_int(stmt, 5);
        auto rows = sqlite3_column_int(stmt, 6);
        auto fps = sqlite3_column_double(stmt, 7);

        CV_LOG_INFO(&g_log_tag, "load a camera \"" << camera_name << "\" from database");
        camera::base* camera = nullptr;

        switch (model_type) {
            case camera::model_type_t::Perspective: {
                auto fx = sqlite3_column_double(stmt, 8);
                auto fy = sqlite3_column_double(stmt, 9);
                auto cx = sqlite3_column_double(stmt, 10);
                auto cy = sqlite3_column_double(stmt, 11);
                auto k1 = sqlite3_column_double(stmt, 12);
                auto k2 = sqlite3_column_double(stmt, 13);
                auto p1 = sqlite3_column_double(stmt, 14);
                auto p2 = sqlite3_column_double(stmt, 15);
                auto k3 = sqlite3_column_double(stmt, 16);
                auto focal_x_baseline = sqlite3_column_double(stmt, 18);
                camera = new camera::perspective(camera_name, setup_type, color_order,
                                                 cols, rows, fps,
                                                 fx, fy, cx, cy,
                                                 k1, k2, p1, p2, k3, focal_x_baseline);
                break;
            }
            case camera::model_type_t::Fisheye: {
                auto fx = sqlite3_column_double(stmt, 8);
                auto fy = sqlite3_column_double(stmt, 9);
                auto cx = sqlite3_column_double(stmt, 10);
                auto cy = sqlite3_column_double(stmt, 11);
                auto k1 = sqlite3_column_double(stmt, 12);
                auto k2 = sqlite3_column_double(stmt, 13);
                auto k3 = sqlite3_column_double(stmt, 16);
                auto k4 = sqlite3_column_double(stmt, 17);
                auto focal_x_baseline = sqlite3_column_double(stmt, 18);
                camera = new camera::fisheye(camera_name, setup_type, color_order,
                                             cols, rows, fps,
                                             fx, fy, cx, cy,
                                             k1, k2, k3, k4, focal_x_baseline);
                break;
            }
            case camera::model_type_t::Equirectangular: {
                camera = new camera::equirectangular(camera_name, color_order,
                                                     cols, rows, fps);
                break;
            }
            case camera::model_type_t::RadialDivision: {
                auto fx = sqlite3_column_double(stmt, 8);
                auto fy = sqlite3_column_double(stmt, 9);
                auto cx = sqlite3_column_double(stmt, 10);
                auto cy = sqlite3_column_double(stmt, 11);
                auto focal_x_baseline = sqlite3_column_double(stmt, 18);
                auto distortion = sqlite3_column_double(stmt, 19);
                camera = new camera::radial_division(camera_name, setup_type, color_order,
                                                     cols, rows, fps,
                                                     fx, fy, cx, cy,
                                                     distortion, focal_x_baseline);
                break;
            }
        }

        assert(!cameras_.count(camera_name));
        cameras_[camera_name] = camera;
    }
    sqlite3_finalize(stmt);
    return ret == SQLITE_DONE;
}

bool camera_database::to_db(sqlite3* db) const {
    std::lock_guard<std::mutex> lock(mtx_database_);
    std::vector<std::pair<std::string, std::string>> columns{
        {"name", "BLOB"},
        {"setup_type", "BLOB"},
        {"model_type", "BLOB"},
        {"color_type", "BLOB"},
        {"cols", "INTEGER"},
        {"rows", "INTEGER"},
        {"fps", "REAL"},
        {"fx", "REAL"},
        {"fy", "REAL"},
        {"cx", "REAL"},
        {"cy", "REAL"},
        {"k1", "REAL"},
        {"k2", "REAL"},
        {"p1", "REAL"},
        {"p2", "REAL"},
        {"k3", "REAL"},
        {"k4", "REAL"},
        {"focal_x_baseline", "REAL"},
        {"distortion", "REAL"}};

    int ret = sqlite3_exec(db, "DROP TABLE IF EXISTS cameras;", nullptr, nullptr, nullptr);
    if (ret == SQLITE_OK) {
        std::string stmt_str = "CREATE TABLE cameras(id INTEGER PRIMARY KEY";
        for (const auto& column : columns) {
            stmt_str += ", " + column.first + " " + column.second;
        }
        stmt_str += ");";
        ret = sqlite3_exec(db, stmt_str.c_str(), nullptr, nullptr, nullptr);
    }
    if (ret == SQLITE_OK) {
        ret = sqlite3_exec(db, "BEGIN;", nullptr, nullptr, nullptr);
    }
    sqlite3_stmt* stmt = nullptr;
    if (ret == SQLITE_OK) {
        std::string stmt_str = "INSERT INTO cameras(id";
        for (const auto& column : columns) {
            stmt_str += ", " + column.first;
        }
        stmt_str += ") VALUES(?";
        for (size_t i = 0; i < columns.size(); ++i) {
            stmt_str += ", ?";
        }
        stmt_str += ")";
        ret = sqlite3_prepare_v2(db, stmt_str.c_str(), -1, &stmt, nullptr);
    }
    if (ret != SQLITE_OK) {
        CV_LOG_ERROR(&g_log_tag, "SQLite error (prepare): " << sqlite3_errmsg(db));
        return false;
    }
    unsigned int camera_id = 0;
    for (const auto& name_camera : cameras_) {
        const auto camera = name_camera.second;
        if (ret == SQLITE_OK || ret == SQLITE_DONE) {
            ret = sqlite3_bind_int64(stmt, 1, camera_id);
            camera_id++;
        }
        if (ret == SQLITE_OK) {
            ret = sqlite3_bind_blob(stmt, 2, camera->name_.c_str(), camera->name_.size(), SQLITE_TRANSIENT);
        }
        if (ret == SQLITE_OK) {
            const std::string setup_type = camera->get_setup_type_string();
            ret = sqlite3_bind_blob(stmt, 3, setup_type.c_str(), setup_type.size(), SQLITE_TRANSIENT);
        }
        if (ret == SQLITE_OK) {
            const std::string model_type = camera->get_model_type_string();
            ret = sqlite3_bind_blob(stmt, 4, model_type.c_str(), model_type.size(), SQLITE_TRANSIENT);
        }
        if (ret == SQLITE_OK) {
            const std::string color_type = camera->get_color_order_string();
            ret = sqlite3_bind_blob(stmt, 5, color_type.c_str(), color_type.size(), SQLITE_TRANSIENT);
        }
        if (ret == SQLITE_OK) {
            ret = sqlite3_bind_int(stmt, 6, camera->cols_);
        }
        if (ret == SQLITE_OK) {
            ret = sqlite3_bind_int(stmt, 7, camera->rows_);
        }
        if (ret == SQLITE_OK) {
            ret = sqlite3_bind_double(stmt, 8, camera->fps_);
        }
        switch (camera->model_type_) {
            case camera::model_type_t::Perspective: {
                auto c = static_cast<camera::perspective*>(camera);
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 9, c->fx_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 10, c->fy_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 11, c->cx_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 12, c->cy_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 13, c->k1_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 14, c->k2_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 15, c->p1_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 16, c->p2_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 17, c->k3_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 19, c->focal_x_baseline_);
                }
                break;
            }
            case camera::model_type_t::Fisheye: {
                auto c = static_cast<camera::fisheye*>(camera);
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 9, c->fx_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 10, c->fy_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 11, c->cx_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 12, c->cy_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 13, c->k1_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 14, c->k2_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 17, c->k3_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 18, c->k4_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 19, c->focal_x_baseline_);
                }
                break;
            }
            case camera::model_type_t::Equirectangular: {
                break;
            }
            case camera::model_type_t::RadialDivision: {
                auto c = static_cast<camera::radial_division*>(camera);
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 9, c->fx_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 10, c->fy_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 11, c->cx_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 12, c->cy_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 19, c->focal_x_baseline_);
                }
                if (ret == SQLITE_OK) {
                    ret = sqlite3_bind_double(stmt, 20, c->distortion_);
                }
                break;
            }
        }
        if (ret != SQLITE_OK) {
            CV_LOG_ERROR(&g_log_tag, "SQLite error (bind): " << sqlite3_errmsg(db));
            return false;
        }
        if (ret == SQLITE_OK) {
            ret = sqlite3_step(stmt);
        }
        if (ret != SQLITE_DONE) {
            CV_LOG_ERROR(&g_log_tag, "SQLite step is not done: " << sqlite3_errmsg(db));
            return false;
        }
        else {
            sqlite3_reset(stmt);
            sqlite3_clear_bindings(stmt);
        }
    }
    sqlite3_finalize(stmt);
    ret = sqlite3_exec(db, "COMMIT;", nullptr, nullptr, nullptr);
    if (ret != SQLITE_OK) {
        CV_LOG_ERROR(&g_log_tag, "SQLite error (commit): " << sqlite3_errmsg(db));
        return false;
    }
    else {
        return true;
    }
}

} // namespace data
} // namespace cv::slam

#endif