#ifndef SLAM_IO_MAP_DATABASE_IO_MSGPACK_H
#define SLAM_IO_MAP_DATABASE_IO_MSGPACK_H

#include "io/map_database_io_base.hpp"
#include "data/bow_vocabulary.hpp"

#include <string>

namespace cv::slam {

namespace data {
class camera_database;
class bow_database;
class map_database;
} // namespace data

namespace io {

class map_database_io_msgpack : public map_database_io_base {
public:
    /**
     * Constructor
     */
    map_database_io_msgpack() = default;

    /**
     * Destructor
     */
    virtual ~map_database_io_msgpack() = default;

    /**
     * Save the map database as JSON
     */
    bool save(const std::string& path,
              const data::camera_database* const cam_db,
              const data::orb_params_database* const orb_params_db,
              const data::map_database* const map_db) override;

    /**
     * Load the map database from MessagePack
     */
    bool load(const std::string& path,
              data::camera_database* cam_db,
              data::orb_params_database* orb_params_db,
              data::map_database* map_db,
              data::bow_database* bow_db,
              data::bow_vocabulary* bow_vocab) override;
};

} // namespace io
} // namespace cv::slam

#endif // SLAM_IO_MAP_DATABASE_IO_MSGPACK_H
