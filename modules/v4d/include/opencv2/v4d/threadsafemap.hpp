// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright Amir Hassan (kallaballa) <amir@viel-zu.org>


#ifndef MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_THREADSAFEMAP_HPP_
#define MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_THREADSAFEMAP_HPP_

#include <any>
#include <concepts>
#include <mutex>
#include <unordered_map>
#include <shared_mutex>

#include <any>
#include <concepts>
#include <mutex>
#include <unordered_map>
#include <shared_mutex>

namespace cv {
namespace v4d {

// A concept to check if a type is hashable
template<typename T>
concept Hashable = requires(T a) {
    { std::hash<T>{}(a) } -> std::convertible_to<std::size_t>;
};

// A concept to check if a type can be stored in an std::unordered_map as value
template<typename T>
concept Mappable = requires(T a) {
    { std::any_cast<T>(std::any{}) } -> std::same_as<T>;
};

// A class that can set and get values in a thread-safe manner (per-key locking)
template<Hashable K>
class ThreadSafeMap {
private:
    // A map from keys to values
    std::unordered_map<K, std::any> map;

    // A map from keys to mutexes
    std::unordered_map<K, std::shared_mutex> mutexes;

    // A mutex to lock the map
    std::shared_mutex map_mutex;

public:
    // A method to set a value for a given key
    template<Mappable V>
    void set(K key, V value) {
        // Lock the map mutex for writing
        std::unique_lock<std::shared_mutex> map_lock(map_mutex);

        // Check if the key exists in the map
        if (map.find(key) == map.end()) {
            // If the key does not exist, insert it into the map and the mutexes
            map[key] = value;
            mutexes[key];
        } else {
            // If the key exists, lock the mutex for the key for writing
            std::unique_lock<std::shared_mutex> key_lock(mutexes[key]);

            // Set the value for the key
            map[key] = value;
        }
    }

    // A method to get a value for a given key
    template<Mappable V>
    V get(K key) {
        // Lock the map mutex for reading
        std::shared_lock<std::shared_mutex> map_lock(map_mutex);

        // Check if the key exists in the map
        if (map.find(key) == map.end()) {
            CV_Error(Error::StsError, "Key not found in map");
        }

        // Lock the mutex for the key for reading
        std::shared_lock<std::shared_mutex> key_lock(mutexes[key]);

        // Get the value for the key
        return std::any_cast<V>(map[key]);
    }

    template<typename F> void on(K key, F func) {
        // Lock the map mutex for reading
        std::shared_lock<std::shared_mutex> map_lock(map_mutex);

        // Check if the key exists in the map
        if (map.find(key) == map.end()) {
            CV_Error(Error::StsError, "Key not found in map");
        }

        // Lock the mutex for the key for writing
        std::unique_lock<std::shared_mutex> key_lock(mutexes[key]);

        // Get the value for the key
        std::any value = map[key];

        // Apply the functor to the value
        func(value);

        // Set the value for the key
        map[key] = value;
    }

    // A method to get a pointer to the value for a given key
    // Note: This function is not thread-safe
    template<Mappable V>
    V* ptr(K key) {
        return std::any_cast<V>(&map[key]);
    }
};

}
}



#endif /* MODULES_V4D_INCLUDE_OPENCV2_V4D_DETAIL_THREADSAFEMAP_HPP_ */
