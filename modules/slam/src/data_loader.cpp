#include "opencv2/slam/data_loader.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <opencv2/core/utils/logger.hpp>
#include <fstream>
#include <sstream>


namespace cv {
namespace vo {

bool ensureDirectoryExists(const std::string &dir){
    if(dir.empty()) return false;
    try {
        if(cv::utils::fs::exists(dir)) return cv::utils::fs::isDirectory(dir);
        return cv::utils::fs::createDirectories(dir);
    } catch(const std::exception &e) {
        CV_LOG_ERROR(NULL, cv::format("ensureDirectoryExists: %s", e.what()));
        return false;
    }
}

DataLoader::DataLoader(const std::string &imageDir)
    : currentIndex(0), fx_(700.0), fy_(700.0), cx_(0.5), cy_(0.5)
{
    try {
        if(!cv::utils::fs::exists(imageDir) || !cv::utils::fs::isDirectory(imageDir)){
            CV_LOG_ERROR(NULL, cv::format("DataLoader: imageDir does not exist or is not a directory: %s", imageDir.c_str()));
            return;
        }
    } catch(const std::exception &e){
        CV_LOG_ERROR(NULL, cv::format("DataLoader: filesystem error checking imageDir: %s", e.what()));
        return;
    }

    // Use glob to list files, catching any possible OpenCV exceptions
    try {
        glob(imageDir + "/*", imageFiles, false);
    } catch(const Exception &e){
        CV_LOG_ERROR(NULL, cv::format("DataLoader: glob failed for '%s': %s", imageDir.c_str(), e.what()));
        imageFiles.clear();
        return;
    }

    if(imageFiles.empty()){
        CV_LOG_ERROR(NULL, cv::format("DataLoader: no image files found in %s", imageDir.c_str()));
        return;
    }

    try {
        std::string yaml1 = cv::utils::fs::join(imageDir, "sensor.yaml");
        std::string yaml2 = cv::utils::fs::join(cv::utils::fs::getParent(imageDir), "sensor.yaml");
        if(!loadIntrinsics(yaml1)){
            loadIntrinsics(yaml2); // best-effort
        }
    } catch(const std::exception &){
    }
}

bool DataLoader::loadIntrinsics(const std::string &yamlPath){
    std::ifstream ifs(yamlPath);
    if(!ifs.is_open()) return false;
    std::string line;
    while(std::getline(ifs, line)){
        auto pos = line.find("intrinsics:");
        if(pos != std::string::npos){
            size_t lb = line.find('[', pos);
            size_t rb = line.find(']', pos);
            std::string nums;
            if(lb != std::string::npos && rb != std::string::npos && rb > lb){
                nums = line.substr(lb+1, rb-lb-1);
            } else {
                std::string rest;
                while(std::getline(ifs, rest)){
                    nums += rest + " ";
                    if(rest.find(']') != std::string::npos) break;
                }
            }
            for(char &c: nums) if(c == ',' || c == '[' || c == ']') c = ' ';
            std::stringstream ss(nums);
            std::vector<double> vals;
            double v;
            while(ss >> v) vals.push_back(v);
            if(vals.size() >= 4){
                fx_ = vals[0]; fy_ = vals[1]; cx_ = vals[2]; cy_ = vals[3];
                CV_LOG_DEBUG(NULL, cv::format("DataLoader: loaded intrinsics from %s", yamlPath.c_str()));
                return true;
            }
        }
    }
    return false;
}

bool DataLoader::hasNext() const { return currentIndex < imageFiles.size(); }

size_t DataLoader::size() const { return imageFiles.size(); }

void DataLoader::reset(){ currentIndex = 0; }

bool DataLoader::getNextImage(Mat &image, std::string &imagePath){
    if(currentIndex >= imageFiles.size()) return false;
    imagePath = imageFiles[currentIndex];
    image = imread(imagePath, IMREAD_UNCHANGED);
    if(image.empty()){
        CV_LOG_WARNING(NULL, cv::format("DataLoader: couldn't read %s, skipping", imagePath.c_str()));
        currentIndex++;
        return getNextImage(image, imagePath); // try next
    }
    currentIndex++;
    return true;
}

} // namespace vo
} // namespace cv