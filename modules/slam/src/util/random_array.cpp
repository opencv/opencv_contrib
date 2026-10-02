#include "util/random_array.hpp"

#include <random>
#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>

namespace cv::slam {
namespace util {

std::mt19937 create_random_engine(bool use_fixed_seed) {
    std::mt19937 random_engine;
    if (use_fixed_seed) {
        return std::mt19937();
    }
    else {
        std::random_device random_device;
        std::vector<std::uint_least32_t> v(10);
        std::generate(v.begin(), v.end(), std::ref(random_device));
        std::seed_seq seed(v.begin(), v.end());
        return std::mt19937(seed);
    }
}

template<typename T>
std::vector<T> create_random_array(const size_t size, const T rand_min, const T rand_max,
                                   std::mt19937& random_engine) {
    assert(rand_min <= rand_max);
    assert(size <= static_cast<size_t>(rand_max - rand_min + 1));

    std::uniform_int_distribution<T> uniform_int_distribution(rand_min, rand_max);

    // Create a random sequence slightly larger than the 'size' (because duplication possibly exists)
    const auto make_size = static_cast<size_t>(size * 1.2);

    // Iterate until the output vector reaches the 'size'
    std::vector<T> v;
    v.reserve(size);
    while (v.size() != size) {
        // Add random integer values
        while (v.size() < make_size) {
            v.push_back(uniform_int_distribution(random_engine));
        }

        // Sort to remove duplicates so that the last iterator of the deduplicated sequence goes into 'unique_end'
        std::sort(v.begin(), v.end());
        auto unique_end = std::unique(v.begin(), v.end());

        // If the vector size is too large, change it to an iterator up to the 'size'
        if (size < static_cast<size_t>(std::distance(v.begin(), unique_end))) {
            unique_end = std::next(v.begin(), size);
        }

        // Delete the portion from the duplication to the end
        v.erase(unique_end, v.end());
    }

    // Shuffle because they are in ascending order
    std::shuffle(v.begin(), v.end(), random_engine);

    return v;
}

// template specialization
template std::vector<int> create_random_array(size_t, int, int, std::mt19937&);
template std::vector<unsigned int> create_random_array(size_t, unsigned int, unsigned int, std::mt19937&);

} // namespace util
} // namespace cv::slam
