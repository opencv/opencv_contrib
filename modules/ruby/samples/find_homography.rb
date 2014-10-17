require 'opencv'
require 'pp'

include OpenCV

points1 = std::Vector.new(cv::Point2f)
points2 = std::Vector.new(cv::Point2f)

points1.push_back(cv::Point2f.new(1,1))
points1.push_back(cv::Point2f.new(3,1))
points1.push_back(cv::Point2f.new(4,2))
points1.push_back(cv::Point2f.new(10,1))

points1.each do |pt|
    points2.push_back(pt*0.2)
end

pp cv::find_homography(points1,points2)
