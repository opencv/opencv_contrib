# writes a video as single images"

require 'opencv'
include OpenCV

video_file = cv::VideoCapture.new(ARGV[0])
frame = cv::Mat.new
id = 0
puts "writing video as single images"
while video_file.read(frame)
    cv::imwrite "name_#{"%05d" % id}.png",frame
    id+=1
end
puts "all done"
