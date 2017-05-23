require 'minitest/autorun'
require 'opencv'
require 'pp'
include OpenCV

MiniTest.autorun
describe Std::Vector::Cv_Point do
    describe "initialize" do
        it "can be created without arguments" do
            vec = Std::Vector::Cv_Point.new
            assert vec
            vec = Std::Vector.new(Cv::Point)
            assert vec
        end

        it "can be created with an element as argument" do
            vec = Std::Vector.new(Cv::Point.new)
            assert vec
        end

        it "can be created with elements as argument" do
            vec = Std::Vector.new(Cv::Point.new,Cv::Point.new)
            assert vec
            assert_equal 2, vec.size
            vec = Std::Vector.new([Cv::Point.new,Cv::Point.new])
            assert vec
            assert_equal 2, vec.size
        end
    end

    describe "at" do
        it "must return a certain element" do
            vec = Std::Vector.new(Cv::Point.new(0,0),Cv::Point.new(1,2))
            assert_equal cv::Point.new(1,2), vec.at(1)
        end

        it "must raise RangeError when out of range " do
            vec = Std::Vector.new(Cv::Point.new(0,0),Cv::Point.new(1,2))
            assert_raises RangeError do
                vec.at(-1)
            end
            assert_raises RangeError do
                vec.at(2)
            end
            vec.clear
            assert_raises RangeError do
                vec.at(0)
            end
        end
    end

    describe "[]" do
        it "must return a certain element" do
            vec = Std::Vector.new(Cv::Point.new(0,0),Cv::Point.new(1,2))
            assert_equal cv::Point.new(1,2), vec.at(1)
        end

        it "must raise RangeError when out of range " do
            vec = Std::Vector.new(Cv::Point.new(0,0),Cv::Point.new(1,2))
            assert_raises RangeError do
                vec[-1]
            end
            assert_raises RangeError do
                vec[2]
            end
        end
    end
end
