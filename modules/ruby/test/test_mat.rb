require 'minitest/autorun'
require 'opencv'
require 'pp'
include OpenCV

MiniTest.autorun
describe Cv::Mat do
    before do
    end

    after do
    end

    describe "initialize" do
        it "can be created with the right type and size" do
            mat = cv::Mat.new(3,4,CV_64FC1)
            assert mat
            assert_equal 3, mat.rows
            assert_equal 4, mat.cols
            assert_equal CV_64FC1, mat.type
        end

        it "can be created from a ruby array" do
            mat = cv::Mat.new([1,2,3])
            assert mat
            assert_equal [1,2,3], mat.t.to_a.flatten
        end

        it "can be created from multiple ruby arrays" do
            mat = cv::Mat.new([1,2,3],[3,3,3],[5,6,7])
            assert_equal [[1,2,3],[3,3,3],[5,6,7]], mat.to_a

            mat = cv::Mat.new([[1,2,3],[3,3,3],[5,6,7]])
            assert_equal [[1,2,3],[3,3,3],[5,6,7]], mat.to_a
        end

        it "can be created from std::vector" do
            vec = Std::Vector.new(cv::Point)
            vec << cv::Point.new(2,3)
            vec << cv::Point.new(4,5)
            mat = cv::Mat.new(vec)
            assert_equal [[2,3],[4,5]], mat.to_a

            vec = Std::Vector.new(cv::Point2f)
            vec << cv::Point2f.new(2,3)
            vec << cv::Point2f.new(4,5)
            mat = cv::Mat.new(vec)
            assert_equal [[2,3],[4,5]], mat.to_a
        end
    end

    describe "each_row" do
        it "iterators over each row" do
            mat = cv::Mat.new([1,2,3],[3,3,3],[5,6,7])
            result = []
            mat.each_row do |r|
                result << r
            end
            assert_equal [1,2,3], result[0].to_a.flatten!
            assert_equal [3,3,3], result[1].to_a.flatten!
            assert_equal [5,6,7], result[2].to_a.flatten!
        end
        it "returns an enumerator if no block is given" do
            mat = cv::Mat.new([1,2,3],[3,3,3],[5,6,7])
            result = mat.each_row.to_a
            assert_equal [1,2,3], result[0].to_a.flatten!
            assert_equal [3,3,3], result[1].to_a.flatten!
            assert_equal [5,6,7], result[2].to_a.flatten!
        end
    end

    describe "each_row_with_index" do
        it "iterators over each row" do
            mat = cv::Mat.new([1,2,3],[3,3,3],[5,6,7])
            result = []
            result_i = []
            mat.each_row_with_index do |r,i|
                result << r
                result_i << i
            end
            assert_equal [1,2,3], result[0].to_a.flatten!
            assert_equal [3,3,3], result[1].to_a.flatten!
            assert_equal [5,6,7], result[2].to_a.flatten!
            assert_equal [0,1,2], result_i
        end
    end

    describe "each_col" do
        it "iterators over each col" do
            mat = cv::Mat.new([1,2,3],[3,3,3],[5,6,7])
            result = []
            mat.each_col do |r|
                result << r
            end
            assert_equal [1,3,5], result[0].to_a.flatten!
            assert_equal [2,3,6], result[1].to_a.flatten!
            assert_equal [3,3,7], result[2].to_a.flatten!
        end
        it "returns an enumerator if no block is given" do
            mat = cv::Mat.new([1,2,3],[3,3,3],[5,6,7])
            result = mat.each_col.to_a
            assert_equal [1,3,5], result[0].to_a.flatten!
            assert_equal [2,3,6], result[1].to_a.flatten!
            assert_equal [3,3,7], result[2].to_a.flatten!
        end
    end

    describe "each_col_with_index" do
        it "iterators over each col" do
            mat = cv::Mat.new([1,2,3],[3,3,3],[5,6,7])
            result = []
            result_i = []
            mat.each_col_with_index do |c,i|
                result << c
                result_i << i
            end
            assert_equal [1,3,5], result[0].to_a.flatten!
            assert_equal [2,3,6], result[1].to_a.flatten!
            assert_equal [3,3,7], result[2].to_a.flatten!
            assert_equal [0,1,2], result_i
        end
    end


    describe "[]" do
        it "can return a specific value" do
            mat = cv::Mat.new([1,2,3],[3,3,3],[5,6,7])
            assert_equal 6, mat[2,1]
        end

        it "can access different channels" do
            mats = std::Vector.new(cv::Mat)
            mats.push_back(cv::Mat::zeros(10,10,CV_8UC1))
            mats.push_back(cv::Mat::ones(10,10,CV_8UC1))
            mat = cv::Mat.new
            cv::merge(mats,mat)

            assert_equal 0, mat[9,9]
            assert_equal 1, mat[8,8,1]
        end
    end

    describe "[]=" do
        it "can cange a specific value" do
            mat = cv::Mat.new([1,2,3],[3,3,3],[5,6,7])
            assert_equal [[1,2,3],[3,3,3],[5,6,7]], mat.to_a
            mat[1,2] = 4
            assert_equal [[1,2,3],[3,3,4],[5,6,7]], mat.to_a
        end

        it "can change different channels" do
            mats = std::Vector.new(cv::Mat)
            mat = cv::Mat::zeros(10,10,CV_32FC3)

            mat[9,9,0] = 123
            mat[9,9,1] = 130
            mat[9,9,2] = 140

            cv::split(mat,mats)
            assert_equal 123.0, mats[0][9,9]
            assert_equal 130.0, mats[1][9,9]
            assert_equal 140.0, mats[2][9,9]
        end
    end
end
