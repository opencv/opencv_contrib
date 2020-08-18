
function createButton(bar_name::String, on_change, userdata, type::Int32 = 0, initial_button_state::Bool = false)
    func =  (x)->on_change(x, userdata)
    CxxWrap.gcprotect(userdata)
    CxxWrap.gcprotect(func)
    CxxWrap.gcprotect(on_change)
    return jl_cpp_cv2.createButton(bar_name,func, type, initial_button_state)
end

function setMouseCallback(winname::String, onMouse, userdata)
    func =  (event, x, y, flags)->onMouse(event, x, y, flags, userdata)
    CxxWrap.gcprotect(userdata)
    CxxWrap.gcprotect(func)
    CxxWrap.gcprotect(onMouse)
    return jl_cpp_cv2.setMouseCallback(winname,func)
end

function createTrackbar(trackbarname::String, winname::String, value::Ref{Int32}, count::Int32, onChange, userdata)
    func =  (x)->onChange(x, userdata)
    CxxWrap.gcprotect(userdata)
    CxxWrap.gcprotect(func)
    CxxWrap.gcprotect(onChange)
    return jl_cpp_cv2.createTrackbar(trackbarname, winname, value, count, func)
end

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

function SimpleBlobDetector_create(parameters::SimpleBlobDetector_Params)
	return cpp_to_julia(jlopencv_cv_cv_SimpleBlobDetector_create(julia_to_cpp(parameters)))
end
SimpleBlobDetector_create(; parameters::SimpleBlobDetector_Params = (SimpleBlobDetector_Params())) = SimpleBlobDetector_create(parameters)
