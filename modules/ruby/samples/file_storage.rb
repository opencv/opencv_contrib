require 'opencv'
require 'pp'
include OpenCV

#writing
fs = cv::FileStorage.new("test.yml",cv::FileStorage::WRITE)
fs << "mat" << cv::Mat.ones(10,10,CV_32FC1)
fs << "int" << 10
fs << "seq" << "[" << 10 << 2 << "]"
fs << "map1" << "{" <<"element1" << 2 << "}"
fs.release

# reading
fs = cv::FileStorage.new("test.yml",cv::FileStorage::READ)
fs.each do |e|
    puts "got node #{e.name}"
end

pp "seq: #{fs["seq"].to_array_of_int}"
pp "mat: #{fs["mat"].to_mat}"
pp "int: #{fs["int"].to_int}"
pp "map.element1: #{fs["map1"]["element1"].to_int}"
pp "map.element1: #{fs.map1.element1.to_int}"

fs.release
