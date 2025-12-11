#include "opencv2/slam/data_loader.hpp"
#include <opencv2/imgcodecs.hpp>
#if defined(__has_include)
#  if __has_include(<filesystem>)
#    include <filesystem>
#    define HAVE_STD_FILESYSTEM 1
    namespace fs = std::filesystem;
#  elif __has_include(<experimental/filesystem>)
#    include <experimental/filesystem>
#    define HAVE_STD_FILESYSTEM 1
    namespace fs = std::experimental::filesystem;
#  else
#    define HAVE_STD_FILESYSTEM 0
#  endif
#else
#  define HAVE_STD_FILESYSTEM 0
#endif
#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>
#include <sys/stat.h>
#ifdef _WIN32
#  include <direct.h>
#  ifndef S_ISDIR
#    define S_ISDIR(mode) (((mode) & S_IFDIR) != 0)
#  endif
#  ifndef mkdir
#    define mkdir(path, mode) _mkdir(path)
#  endif
#endif
#include <iostream>
#include <fstream>
#include <sstream>


namespace cv {
namespace vo {

bool ensureDirectoryExists(const std::string &dir){
    if(dir.empty()) return false;
#if HAVE_STD_FILESYSTEM
    try{
        fs::path p(dir);
        return fs::create_directories(p) || fs::exists(p);
    }catch(...){ return false; }
#else
    std::string tmp = dir;
    if(tmp.empty()) return false;
    while(tmp.size() > 1 && tmp.back() == '/') tmp.pop_back();
    std::string cur;
    if(!tmp.empty() && tmp[0] == '/') cur = "/";
    size_t pos = 0;
    while(pos < tmp.size()){
        size_t next = tmp.find('/', pos);
        std::string part = (next == std::string::npos) ? tmp.substr(pos) : tmp.substr(pos, next-pos);
        if(!part.empty()){
            if(cur.size() > 1 && cur.back() != '/') cur += '/';
            cur += part;
            if(mkdir(cur.c_str(), 0755) != 0){
                if(errno == EEXIST){ /* ok */ }
                else {
                    struct stat st;
                    if(stat(cur.c_str(), &st) == 0 && S_ISDIR(st.st_mode)){
                        // ok
                    } else return false;
                }
            }
        }
        if(next == std::string::npos) break;
        pos = next + 1;
    }
    return true;
#endif
}

DataLoader::DataLoader(const std::string &imageDir)
    : currentIndex(0), fx_(700.0), fy_(700.0), cx_(0.5), cy_(0.5)
{
    // Check whether the directory exists. Prefer std::filesystem when available,
    // otherwise fall back to POSIX stat.
#if HAVE_STD_FILESYSTEM
    try {
        fs::path p(imageDir);
        if(!fs::exists(p) || !fs::is_directory(p)){
            std::cerr << "DataLoader: imageDir does not exist or is not a directory: " << imageDir << std::endl;
            return;
        }
    } catch(const std::exception &e){
        std::cerr << "DataLoader: filesystem error checking imageDir: " << e.what() << std::endl;
        // fallthrough to try glob
    }
#else
    struct stat sb;
    if(stat(imageDir.c_str(), &sb) != 0 || !S_ISDIR(sb.st_mode)){
        std::cerr << "DataLoader: imageDir does not exist or is not a directory: " << imageDir << std::endl;
        return;
    }
#endif

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

    // Try to find sensor.yaml in imageDir or its parent directory.
    // Use filesystem paths when available; otherwise build paths with simple string ops.
#if HAVE_STD_FILESYSTEM
    fs::path p(imageDir);
    std::string yaml1 = (p / "sensor.yaml").string();
    std::string yaml2 = (p.parent_path() / "sensor.yaml").string();
    if(!loadIntrinsics(yaml1)){
        loadIntrinsics(yaml2); // best-effort
    }
#else
    auto make_parent_yaml = [](const std::string &dir)->std::string{
        std::string tmp = dir;
        if(!tmp.empty() && tmp.back() == '/') tmp.pop_back();
        auto pos = tmp.find_last_of('/');
        if(pos == std::string::npos) return std::string("sensor.yaml");
        return tmp.substr(0, pos) + "/sensor.yaml";
    };
    std::string yaml1 = imageDir + "/sensor.yaml";
    std::string yaml2 = make_parent_yaml(imageDir);
    if(!loadIntrinsics(yaml1)){
        loadIntrinsics(yaml2); // best-effort
    }
#endif
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