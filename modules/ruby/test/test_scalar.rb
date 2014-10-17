require 'minitest/autorun'
require 'opencv'
require 'pp'
include OpenCV

MiniTest.autorun
describe Cv::Scalar do
    describe "initialize" do
        it "can be created from 4 values" do
            scalar = cv::Scalar.new(1,2,3,4)
            assert_equal [1,2,3,4], scalar.to_a
        end
    end

    describe "[]" do
        it "must access a certain field" do
            scalar = cv::Scalar.new(1,2,3,4)
            assert_equal 1, scalar[0]
            assert_equal 2, scalar[1]
            assert_equal 3, scalar[2]
            assert_equal 4, scalar[3]
        end
    end

    describe "[]=" do
        it "must seta certain field" do
            scalar = cv::Scalar.new(1,2,3,4)
            assert_equal 2, scalar[1]

            scalar[1] = 222
            assert_equal 222, scalar[1]
        end
    end
end
