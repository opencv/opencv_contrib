#ifndef SLAM_DATA_ORB_PARAMS_DATABASE_H
#define SLAM_DATA_ORB_PARAMS_DATABASE_H

#include <mutex>
#include <unordered_map>

#include <nlohmann/json_fwd.hpp>

namespace cv::slam {

namespace feature {
struct orb_params;
} // namespace feature

namespace data {

class orb_params_database {
public:
    explicit orb_params_database();

    ~orb_params_database();

    void add_orb_params(feature::orb_params* orb_params);

    feature::orb_params* get_orb_params(const std::string& orb_params_name) const;

    void from_json(const nlohmann::json& json_orb_params);

    nlohmann::json to_json() const;

private:
    //-----------------------------------------
    //! mutex to access the database
    mutable std::mutex mtx_database_;
    //! database (key: orb_params name, value: pointer of feature::orb_params)
    std::unordered_map<std::string, feature::orb_params*> orb_params_database_;
};

} // namespace data
} // namespace cv::slam

#endif // SLAM_DATA_ORB_PARAMS_DATABASE_H
