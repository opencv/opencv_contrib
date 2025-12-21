#include "opencv2/slam/data_loader.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/utils/filesystem.hpp>
#include <iostream>
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
        std::cerr << "ensureDirectoryExists: " << e.what() << std::endl;
        return false;
    }
}

DataLoader::DataLoader(const std::string &imageDir)
    : currentIndex(0), fx_(700.0), fy_(700.0), cx_(0.5), cy_(0.5)
{
    try {
        if(!cv::utils::fs::exists(imageDir) || !cv::utils::fs::isDirectory(imageDir)){
            std::cerr << "DataLoader: imageDir does not exist or is not a directory: " << imageDir << std::endl;
            return;
        }
    } catch(const std::exception &e){
        std::cerr << "DataLoader: filesystem error checking imageDir: " << e.what() << std::endl;
        return;
    }

    // Use glob to list files, catching any possible OpenCV exceptions
    try {
        glob(imageDir + "/*", imageFiles, false);
    } catch(const Exception &e){
        std::cerr << "DataLoader: glob failed for '" << imageDir << "': " << e.what() << std::endl;
        imageFiles.clear();
        return;
    }

    if(imageFiles.empty()){
        std::cerr << "DataLoader: no image files found in " << imageDir << std::endl;
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
                std::cerr << "DataLoader: loaded intrinsics from " << yamlPath << std::endl;
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
        std::cerr << "DataLoader: couldn't read " << imagePath << ", skipping" << std::endl;
        currentIndex++;
        return getNextImage(image, imagePath); // try next
    }
    currentIndex++;
    return true;
}

} // namespace vo
} // namespace cv