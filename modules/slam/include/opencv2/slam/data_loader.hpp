#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace cv{
namespace vo{

CV_EXPORTS bool ensureDirectoryExists(const std::string &dir);

/**
 * @brief Simple helper to iterate image sequences on disk.
 *
 * DataLoader enumerates image files in a directory and provides simple
 * sequential access for samples and demos. This class does not perform
 * any image decoding policy beyond forwarding files to OpenCV's IO.
 */
class CV_EXPORTS DataLoader {
public:
    /**
     * @brief Construct a DataLoader for the given image directory.
     * @param imageDir Directory containing image files (absolute or relative path).
     */
    DataLoader(const std::string &imageDir);

    /**
     * @brief Load the next image in the sequence.
     * @param image Output image (decoded by OpenCV).
     * @param imagePath Output path of the loaded image file.
     * @return True if an image was loaded; false when the sequence has ended.
     */
    bool getNextImage(Mat &image, std::string &imagePath);

    /**
     * @brief Reset the internal iterator to the beginning of the sequence.
     */
    void reset();

    /**
     * @brief Check whether more images are available.
     * @return True when there are remaining images to read.
     */
    bool hasNext() const;

    /**
     * @brief Get total number of images discovered in the directory.
     * @return Number of image files.
     */
    size_t size() const;

    /**
     * @brief Try to load camera intrinsics from a YAML file.
     * @param yamlPath Path to the YAML file containing camera parameters.
     * @return True on success, false otherwise.
     */
    bool loadIntrinsics(const std::string &yamlPath);

    /** @brief Focal length in x. */
    double fx() const { return fx_; }
    /** @brief Focal length in y. */
    double fy() const { return fy_; }
    /** @brief Principal point x-coordinate. */
    double cx() const { return cx_; }
    /** @brief Principal point y-coordinate. */
    double cy() const { return cy_; }

private:
    std::vector<std::string> imageFiles;
    size_t currentIndex;

    // Camera intrinsics (fallback values when not loaded)
    double fx_, fy_, cx_, cy_;
};

} // namespace vo
} // namespace cv
