require 'minitest/autorun'
require 'opencv'
require 'pp'
include OpenCV


MiniTest.autorun
describe Cv do
    describe "triangulate_points" do
        it "must triangulate the given points" do

            # generate camera matrix
            p1 = cv::Mat.eye(3,4,CV_32FC1)
            p2 = cv::Mat.eye(3,4,CV_32FC1)
            p2[0,3] = 0.1

            # generate 2 points
            points1 = Vector.new(cv::Point2f)
            points2 = Vector.new(cv::Point2f)

            points1 << cv::Point2f.new(0.5,0.1)
            points2 << cv::Point2f.new(0.7,0.1)

            points1 << cv::Point2f.new(0.4,0.1)
            points2 << cv::Point2f.new(0.2,0.1)

            points3d = cv::Mat.new
            cv::triangulate_points(p1,p2,cv::Mat.new(points1).t,cv::Mat.new(points2).t,points3d)

            assert_equal 4, points3d.rows
            assert_equal 2, points3d.cols
        end
    end
end
