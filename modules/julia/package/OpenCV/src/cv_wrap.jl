function Base.getproperty(m::KeyPoint, s::Symbol)
    if s==:pt
        return cpp_to_julia(jlopencv_KeyPoint_get_pt(m))
    end
    if s==:size
        return cpp_to_julia(jlopencv_KeyPoint_get_size(m))
    end
    if s==:angle
        return cpp_to_julia(jlopencv_KeyPoint_get_angle(m))
    end
    if s==:response
        return cpp_to_julia(jlopencv_KeyPoint_get_response(m))
    end
    if s==:octave
        return cpp_to_julia(jlopencv_KeyPoint_get_octave(m))
    end
    if s==:class_id
        return cpp_to_julia(jlopencv_KeyPoint_get_class_id(m))
    end
    return Base.getfield(m, s)
end
function Base.setproperty!(m::KeyPoint, s::Symbol, v)
    return Base.setfield(m, s, v)
end

function KeyPoint(x::Float32, y::Float32, _size::Float32, _angle::Float32, _response::Float32, _octave::Int32, _class_id::Int32)
	return cpp_to_julia(jlopencv_cv_cv_KeyPoint_cv_KeyPoint_KeyPoint(julia_to_cpp(x),julia_to_cpp(y),julia_to_cpp(_size),julia_to_cpp(_angle),julia_to_cpp(_response),julia_to_cpp(_octave),julia_to_cpp(_class_id)))
end
KeyPoint(x::Float32, y::Float32, _size::Float32; _angle::Float32 = Float32(-1), _response::Float32 = Float32(0), _octave::Int32 = Int32(0), _class_id::Int32 = Int32(-1)) = KeyPoint(x, y, _size, _angle, _response, _octave, _class_id)

function VideoCapture(filename::String, apiPreference::Int32)
	return cpp_to_julia(jlopencv_cv_cv_VideoCapture_cv_VideoCapture_VideoCapture(julia_to_cpp(filename),julia_to_cpp(apiPreference)))
end
VideoCapture(filename::String; apiPreference::Int32 = Int32(CAP_ANY)) = VideoCapture(filename, apiPreference)

function VideoCapture(index::Int32, apiPreference::Int32)
	return cpp_to_julia(jlopencv_cv_cv_VideoCapture_cv_VideoCapture_VideoCapture(julia_to_cpp(index),julia_to_cpp(apiPreference)))
end
VideoCapture(index::Int32; apiPreference::Int32 = Int32(CAP_ANY)) = VideoCapture(index, apiPreference)

function CascadeClassifier(filename::String)
	return cpp_to_julia(jlopencv_cv_cv_CascadeClassifier_cv_CascadeClassifier_CascadeClassifier(julia_to_cpp(filename)))
end


function detect(cobj::cv_Ptr{T}, image::InputArray, mask::InputArray) where {T <: Feature2D}
	return cpp_to_julia(jlopencv_cv_cv_Feature2D_cv_Feature2D_detect(julia_to_cpp(cobj),julia_to_cpp(image),julia_to_cpp(mask)))
end
detect(cobj::cv_Ptr{T}, image::InputArray; mask::InputArray = (CxxMat())) where {T <: Feature2D} = detect(cobj, image, mask)


function detectMultiScale(cobj::CascadeClassifier, image::InputArray, scaleFactor::Float64, minNeighbors::Int32, flags::Int32, minSize::Size{Int32}, maxSize::Size{Int32})
	return cpp_to_julia(jlopencv_cv_cv_CascadeClassifier_cv_CascadeClassifier_detectMultiScale(julia_to_cpp(cobj),julia_to_cpp(image),julia_to_cpp(scaleFactor),julia_to_cpp(minNeighbors),julia_to_cpp(flags),julia_to_cpp(minSize),julia_to_cpp(maxSize)))
end
detectMultiScale(cobj::CascadeClassifier, image::InputArray; scaleFactor::Float64 = Float64(1.1), minNeighbors::Int32 = Int32(3), flags::Int32 = Int32(0), minSize::Size{Int32} = (Size{Int32}(0,0)), maxSize::Size{Int32} = (Size{Int32}(0,0))) = detectMultiScale(cobj, image, scaleFactor, minNeighbors, flags, minSize, maxSize)


function empty(cobj::CascadeClassifier)
	return cpp_to_julia(jlopencv_cv_cv_CascadeClassifier_cv_CascadeClassifier_empty(julia_to_cpp(cobj)))
end

function read(cobj::VideoCapture, image::InputArray)
	return cpp_to_julia(jlopencv_cv_cv_VideoCapture_cv_VideoCapture_read(julia_to_cpp(cobj),julia_to_cpp(image)))
end
read(cobj::VideoCapture; image::InputArray = (CxxMat())) = read(cobj, image)

function release(cobj::VideoCapture)
	return cpp_to_julia(jlopencv_cv_cv_VideoCapture_cv_VideoCapture_release(julia_to_cpp(cobj)))
end

function SimpleBlobDetector_create(parameters::SimpleBlobDetector_Params)
	return cpp_to_julia(jlopencv_cv_cv_SimpleBlobDetector_create(julia_to_cpp(parameters)))
end
SimpleBlobDetector_create(; parameters::SimpleBlobDetector_Params = (SimpleBlobDetector_Params())) = SimpleBlobDetector_create(parameters)

function imread(filename::String, flags::Int32)
	return cpp_to_julia(jlopencv_cv_cv_imread(julia_to_cpp(filename),julia_to_cpp(flags)))
end
imread(filename::String; flags::Int32 = Int32(IMREAD_COLOR)) = imread(filename, flags)

function imshow(winname::String, mat::InputArray)
	return cpp_to_julia(jlopencv_cv_cv_imshow(julia_to_cpp(winname),julia_to_cpp(mat)))
end

function namedWindow(winname::String, flags::Int32)
	return cpp_to_julia(jlopencv_cv_cv_namedWindow(julia_to_cpp(winname),julia_to_cpp(flags)))
end
namedWindow(winname::String; flags::Int32 = Int32(WINDOW_AUTOSIZE)) = namedWindow(winname, flags)

function waitKey(delay::Int32)
	return cpp_to_julia(jlopencv_cv_cv_waitKey(julia_to_cpp(delay)))
end
waitKey(; delay::Int32 = Int32(0)) = waitKey(delay)


function rectangle(img::InputArray, pt1::Point{Int32}, pt2::Point{Int32}, color::Scalar, thickness::Int32, lineType::Int32, shift::Int32)
	return cpp_to_julia(jlopencv_cv_cv_rectangle(julia_to_cpp(img),julia_to_cpp(pt1),julia_to_cpp(pt2),julia_to_cpp(color),julia_to_cpp(thickness),julia_to_cpp(lineType),julia_to_cpp(shift)))
end
rectangle(img::InputArray, pt1::Point{Int32}, pt2::Point{Int32}, color::Scalar; thickness::Int32 = Int32(1), lineType::Int32 = Int32(LINE_8), shift::Int32 = Int32(0)) = rectangle(img, pt1, pt2, color, thickness, lineType, shift)

function cvtColor(src::InputArray, code::Int32, dst::InputArray, dstCn::Int32)
	return cpp_to_julia(jlopencv_cv_cv_cvtColor(julia_to_cpp(src),julia_to_cpp(code),julia_to_cpp(dst),julia_to_cpp(dstCn)))
end
cvtColor(src::InputArray, code::Int32; dst::InputArray = (CxxMat()), dstCn::Int32 = Int32(0)) = cvtColor(src, code, dst, dstCn)

function equalizeHist(src::InputArray, dst::InputArray)
	return cpp_to_julia(jlopencv_cv_cv_equalizeHist(julia_to_cpp(src),julia_to_cpp(dst)))
end
equalizeHist(src::InputArray; dst::InputArray = (CxxMat())) = equalizeHist(src, dst)


function destroyAllWindows()
	return cpp_to_julia(jlopencv_cv_cv_destroyAllWindows())
end
