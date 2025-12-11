#pragma once
#include <opencv2/core.hpp>
#include <string>
#include <vector>

namespace cv{
namespace vo{

bool ensureDirectoryExists(const std::string &dir);

class DataLoader {
public:
    // 构造：传入图像目录（可以是相对或绝对路径）
    DataLoader(const std::string &imageDir);

    // 获取下一张图像，成功返回 true 并填充 image 与 imagePath；到末尾返回 false
    bool getNextImage(Mat &image, std::string &imagePath);

    // 重置到序列开始
    void reset();

    // 是否还有图像
    bool hasNext() const;

    // 图像总数
    size_t size() const;

    // 尝试加载并返回相机内参（fx, fy, cx, cy），返回是否成功
    bool loadIntrinsics(const std::string &yamlPath);

    // 内参访问
    double fx() const { return fx_; }
    double fy() const { return fy_; }
    double cx() const { return cx_; }
    double cy() const { return cy_; }

private:
    std::vector<std::string> imageFiles;
    size_t currentIndex;

    // 相机内参（若未加载则为回退值）
    double fx_, fy_, cx_, cy_;
};

} // namespace vo
} // namespace cv
