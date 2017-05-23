$LOAD_PATH << File.join(File.dirname(__FILE__),'lib')

require 'rbind'
require 'pp'
require 'yaml'

require File.join(File.dirname(__FILE__),'templates.rb')

if ARGV.empty?
    raise "No headers are given. Expected DST_FOLDER LIST_HEADERS"
end
opencv_headers = ARGV
output_folder = opencv_headers.shift

rbind = Rbind::Rbind.new("OpenCV")
rbind.includes = opencv_headers

# add some templates and alias
rbind.parser.type_alias["const_c_string"] = rbind.c_string.to_const
rbind.add_std_types
rbind.parser.add_type OpenCVPtr2.new

# add Vec types
2.upto(6) do |idx|
    next if idx == 5
    rbind.parser.add_type Vec.new("cv::Vec#{idx}d",rbind.double,idx)
    rbind.parser.add_type Vec.new("cv::Vec#{idx}f",rbind.float,idx)
    rbind.parser.add_type Vec.new("cv::Vec#{idx}i",rbind.int,idx)
end

# forward declaration to avoid missing types during parsing
rbind.cv.add_type(Rbind::RClass.new("ShapeTransformer"))
rbind.cv.add_type(Rbind::RClass.new("Feature2D"))
rbind.cv.type_alias["FeatureDetector"] = rbind.cv.Feature2D
rbind.cv.type_alias["DescriptorExtractor"] = rbind.cv.Feature2D

# add missing enum values
rbind.cv.add_type(Rbind::RClass.new("Stitcher"))
rbind.cv.Stitcher.add_type(Rbind::REnum.new("Status"))
rbind.cv.Stitcher.Status.values = {:OK => 0, :ERR_NEED_MORE_IMGS => 1,:ERR_HOMOGRAPHY_EST_FAIL => 2,:ERR_CAMERA_PARAMS_ADJUST_FAIL => 3}

# parsing
rbind.parse File.join(File.dirname(__FILE__),"pre_opencv.txt")
rbind.use_namespace rbind.cv
rbind.use_namespace rbind.std
rbind.cv.type_alias["string"] = rbind.cv.String

# parse headers and delete all headers which
# have no content
rbind.includes.delete_if do |h|
    0 == rbind.parse_header(h)
end

rbind.parse File.join(File.dirname(__FILE__),"post_opencv.txt")
rbind.cv.ml.StatModel.getParams.ignore = true if rbind.cv.type?(:ml) # shadowed base method

# add some std::vector types
[:Point2d,:Point3f,:Point3d,:Vec4i,:uint32_t,:uint64_t,
       :int8_t,:int64_t,:Scalar,"std::vector<Point2d>"].each do |t|
    rbind.parser.type("std::vector<#{t}>")
end

# add some extra documentation
# rbind.parser.doc = "ROpenCV API Documentation for OpenCV #{opencv_version}"
# rbind.parser.cv.doc = "ROpenCV API Documentation for OpenCV #{opencv_version}"

# replace default parameter values which are template
# functions and not available on the ruby side
Rbind::GeneratorRuby.on_normalize_default_value do |parameter|
    if parameter.default_value =~ /.*makePtr<(.*)>\(\)/
        "cv::Ptr<#{$1}>(new #{$1})"
    end
end

# generate files
rbind.generator_ruby.library_name = "opencv_ruby"
rbind.generator_ruby.file_prefix = "opencv"
rbind.generate(output_folder,File.join(File.dirname(__FILE__),"..","lib","opencv"))

