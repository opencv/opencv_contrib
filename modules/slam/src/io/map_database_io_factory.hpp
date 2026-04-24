#ifndef SLAM_IO_MAP_DATABASE_IO_FACTORY_H
#define SLAM_IO_MAP_DATABASE_IO_FACTORY_H

#include "data/bow_vocabulary.hpp"
#include "io/map_database_io_base.hpp"
#include "io/map_database_io_msgpack.hpp"
#include "io/map_database_io_sqlite3.hpp"

#include <string>

namespace cv::slam {

namespace data {
class camera_database;
class bow_database;
class map_database;
} // namespace data

namespace io {

class map_database_io_factory {
public:
    static std::shared_ptr<map_database_io_base> create(const std::string& map_format) {
        std::shared_ptr<map_database_io_base> map_database_io;
        if (map_format == "sqlite3") {
            map_database_io = std::make_shared<io::map_database_io_sqlite3>();
        }
        else if (map_format == "msgpack") {
            map_database_io = std::make_shared<io::map_database_io_msgpack>();
        }
        else {
            throw std::runtime_error("Invalid map format: " + map_format);
        }
        return map_database_io;
    }
};

} // namespace io
} // namespace cv::slam

#endif // SLAM_IO_MAP_DATABASE_IO_FACTORY_H
