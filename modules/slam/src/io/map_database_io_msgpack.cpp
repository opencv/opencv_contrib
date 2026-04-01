#include "data/frame.hpp"
#include "data/keyframe.hpp"
#include "data/landmark.hpp"
#include "data/camera_database.hpp"
#include "data/orb_params_database.hpp"
#include "data/bow_database.hpp"
#include "data/map_database.hpp"
#include "io/map_database_io_msgpack.hpp"

#include <opencv2/core/utils/logger.hpp>
#include <nlohmann/json.hpp>

#include <fstream>

namespace cv::slam {

static cv::utils::logging::LogTag g_log_tag("cv_slam", cv::utils::logging::LOG_LEVEL_INFO);
namespace io {

bool map_database_io_msgpack::save(const std::string& path,
                                   const data::camera_database* const cam_db,
                                   const data::orb_params_database* const orb_params_db,
                                   const data::map_database* const map_db) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);

    assert(cam_db && orb_params_db && map_db);
    const auto cameras = cam_db->to_json();
    const auto orb_params = orb_params_db->to_json();
    nlohmann::json keyfrms;
    nlohmann::json landmarks;
    map_db->to_json(keyfrms, landmarks);

    nlohmann::json json{{"cameras", cameras},
                        {"orb_params", orb_params},
                        {"keyframes", keyfrms},
                        {"landmarks", landmarks},
                        {"keyframe_next_id", static_cast<unsigned int>(map_db->next_keyframe_id_)},
                        {"landmark_next_id", static_cast<unsigned int>(map_db->next_landmark_id_)}};

    std::ofstream ofs(path, std::ios::out | std::ios::binary);

    if (ofs.is_open()) {
        CV_LOG_INFO(&g_log_tag, "save the MessagePack file of database to " << path);
        const auto msgpack = nlohmann::json::to_msgpack(json);
        ofs.write(reinterpret_cast<const char*>(msgpack.data()), msgpack.size() * sizeof(uint8_t));
        ofs.close();
        return true;
    }
    else {
        CV_LOG_FATAL(&g_log_tag, "cannot create a file at " << path);
        return false;
    }
}

bool map_database_io_msgpack::load(const std::string& path,
                                   data::camera_database* cam_db,
                                   data::orb_params_database* orb_params_db,
                                   data::map_database* map_db,
                                   data::bow_database* bow_db,
                                   data::bow_vocabulary* bow_vocab) {
    std::lock_guard<std::mutex> lock(data::map_database::mtx_database_);
    assert(cam_db && orb_params_db && map_db && bow_db && bow_vocab);

    // load binary bytes

    std::ifstream ifs(path, std::ios::in | std::ios::binary);
    if (!ifs.is_open()) {
        CV_LOG_FATAL(&g_log_tag, "cannot load the file at " << path);
        return false;
    }

    CV_LOG_INFO(&g_log_tag, "load the MessagePack file of database from " << path);
    std::vector<uint8_t> msgpack;
    while (true) {
        uint8_t buffer;
        ifs.read(reinterpret_cast<char*>(&buffer), sizeof(uint8_t));
        if (ifs.eof()) {
            break;
        }
        msgpack.push_back(buffer);
    }
    ifs.close();

    // parse into JSON

    const auto json = nlohmann::json::from_msgpack(msgpack);

    // load database
    const auto json_cameras = json.at("cameras");
    cam_db->from_json(json_cameras);
    const auto json_orb_params = json.at("orb_params");
    orb_params_db->from_json(json_orb_params);
    const auto json_keyfrms = json.at("keyframes");
    const auto json_landmarks = json.at("landmarks");
    map_db->from_json(cam_db, orb_params_db, bow_vocab, json_keyfrms, json_landmarks);
    // load next ID
    map_db->next_keyframe_id_ += json.at("keyframe_next_id").get<unsigned int>();
    map_db->next_landmark_id_ += json.at("landmark_next_id").get<unsigned int>();

    // update bow database
    const auto keyfrms = map_db->get_all_keyframes();
    for (const auto& keyfrm : keyfrms) {
        bow_db->add_keyframe(keyfrm);
    }
    return true;
}

} // namespace io
} // namespace cv::slam
