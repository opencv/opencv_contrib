require 'minitest/autorun'
require 'opencv'
require 'pp'
include OpenCV

MiniTest.autorun

describe Cv::Vec2d do
    describe "initialize" do
        it "can be created from 2 values" do
            vec = cv::Vec2d.new(12,2)
            assert_equal [12,2], vec.to_a
        end
    end

    describe "[]" do
        it "must access a given field" do
            vec = cv::Vec2d.new(12,2)
            assert_equal 2, vec[1]
        end
    end

    describe "[]=" do
        it "must set a given field" do
            vec = cv::Vec2d.new(12,2)
            vec[0] = 0
            assert_equal 0, vec[0]
        end
    end
end

describe Cv::Vec2f do
    describe "initialize" do
        it "can be created from 2 values" do
            vec = cv::Vec2f.new(12,2)
            assert_equal [12,2], vec.to_a
        end
    end

    describe "[]" do
        it "must access a given field" do
            vec = cv::Vec2f.new(12,2)
            assert_equal 2, vec[1]
        end
    end

    describe "[]=" do
        it "must set a given field" do
            vec = cv::Vec2f.new(12,2)
            vec[0] = 0
            assert_equal 0, vec[0]
        end
    end
end

describe Cv::Vec2i do
    describe "initialize" do
        it "can be created from 2 values" do
            vec = cv::Vec2i.new(12,2)
            assert_equal [12,2], vec.to_a
        end
    end

    describe "[]" do
        it "must access a given field" do
            vec = cv::Vec2i.new(12,2)
            assert_equal 2, vec[1]
        end
    end

    describe "[]=" do
        it "must set a given field" do
            vec = cv::Vec2i.new(12,2)
            vec[0] = 0
            assert_equal 0, vec[0]
        end
    end
end

describe Cv::Vec3d do
    describe "initialize" do
        it "can be created from 3 values" do
            vec = cv::Vec3d.new(12,2,3)
            assert_equal [12,2,3], vec.to_a
        end
    end
end

describe Cv::Vec3f do
    describe "initialize" do
        it "can be created from 3 values" do
            vec = cv::Vec3f.new(12,2,3)
            assert_equal [12,2,3], vec.to_a
        end
    end
end

describe Cv::Vec3i do
    describe "initialize" do
        it "can be created from 3 values" do
            vec = cv::Vec3i.new(12,2,3)
            assert_equal [12,2,3], vec.to_a
        end
    end
end

describe Cv::Vec4d do
    describe "initialize" do
        it "can be created from 4 values" do
            vec = cv::Vec4d.new(12,2,3,4)
            assert_equal [12,2,3,4], vec.to_a
        end
    end
end

describe Cv::Vec4f do
    describe "initialize" do
        it "can be created from 4 values" do
            vec = cv::Vec4f.new(12,2,3,4)
            assert_equal [12,2,3,4], vec.to_a
        end
    end
end

describe Cv::Vec4i do
    describe "initialize" do
        it "can be created from 4 values" do
            vec = cv::Vec4i.new(12,2,3,4)
            assert_equal [12,2,3,4], vec.to_a
        end
    end
end

describe Cv::Vec6d do
    describe "initialize" do
        it "can be created from 6 values" do
            vec = cv::Vec6d.new(12,2,3,4,5,6)
            assert_equal [12,2,3,4,5,6], vec.to_a
        end
    end
end

describe Cv::Vec6i do
    describe "initialize" do
        it "can be created from 6 values" do
            vec = cv::Vec6i.new(12,2,3,4,5,6)
            assert_equal [12,2,3,4,5,6], vec.to_a
        end
    end
end
