This is a compilation of the general functions implemented in this module.

```
function KeyPoint(x::Float32, y::Float32, _size::Float32, _angle::Float32, _response::Float32, _octave::Int32, _class_id::Int32)

KeyPoint(x::Float32, y::Float32, _size::Float32; _angle::Float32 = Float32(-1), _response::Float32 = Float32(0), _octave::Int32 = Int32(0), _class_id::Int32 = Int32(-1)) = KeyPoint(x, y, _size, _angle, _response, _octave, _class_id)

function VideoCapture(filename::String, apiPreference::Int32)

VideoCapture(filename::String; apiPreference::Int32 = Int32(CAP_ANY)) = VideoCapture(filename, apiPreference)

function VideoCapture(index::Int32, apiPreference::Int32)

VideoCapture(index::Int32; apiPreference::Int32 = Int32(CAP_ANY)) = VideoCapture(index, apiPreference)

function CascadeClassifier(filename::String)

function detect(cobj::cv_Ptr{T}, image::InputArray, mask::InputArray) where {T <: Feature2D}

detect(cobj::cv_Ptr{T}, image::InputArray; mask::InputArray = (CxxMat())) where {T <: Feature2D} = detect(cobj, image, mask)

function detectMultiScale(cobj::CascadeClassifier, image::InputArray, scaleFactor::Float64, minNeighbors::Int32, flags::Int32, minSize::Size{Int32}, maxSize::Size{Int32})

detectMultiScale(cobj::CascadeClassifier, image::InputArray; scaleFactor::Float64 = Float64(1.1), minNeighbors::Int32 = Int32(3), flags::Int32 = Int32(0), minSize::Size{Int32} = (Size{Int32}(0,0)), maxSize::Size{Int32} = (Size{Int32}(0,0))) = detectMultiScale(cobj, image, scaleFactor, minNeighbors, flags, minSize, maxSize)

function empty(cobj::CascadeClassifier)

function read(cobj::VideoCapture, image::InputArray)

read(cobj::VideoCapture; image::InputArray = (CxxMat())) = read(cobj, image)

function release(cobj::VideoCapture)

function SimpleBlobDetector_create(parameters::SimpleBlobDetector_Params)

SimpleBlobDetector_create(; parameters::SimpleBlobDetector_Params = (SimpleBlobDetector_Params())) = SimpleBlobDetector_create(parameters)

function imread(filename::String, flags::Int32)

imread(filename::String; flags::Int32 = Int32(IMREAD_COLOR)) = imread(filename, flags)

function imshow(winname::String, mat::InputArray)

function namedWindow(winname::String, flags::Int32)

namedWindow(winname::String; flags::Int32 = Int32(WINDOW_AUTOSIZE)) = namedWindow(winname, flags)

function waitKey(delay::Int32)

waitKey(; delay::Int32 = Int32(0)) = waitKey(delay)

function rectangle(img::InputArray, pt1::Point{Int32}, pt2::Point{Int32}, color::Scalar, thickness::Int32, lineType::Int32, shift::Int32)

rectangle(img::InputArray, pt1::Point{Int32}, pt2::Point{Int32}, color::Scalar; thickness::Int32 = Int32(1), lineType::Int32 = Int32(LINE_8), shift::Int32 = Int32(0)) = rectangle(img, pt1, pt2, color, thickness, lineType, shift)

function cvtColor(src::InputArray, code::Int32, dst::InputArray, dstCn::Int32)

cvtColor(src::InputArray, code::Int32; dst::InputArray = (CxxMat()), dstCn::Int32 = Int32(0)) = cvtColor(src, code, dst, dstCn)

function equalizeHist(src::InputArray, dst::InputArray)

equalizeHist(src::InputArray; dst::InputArray = (CxxMat())) = equalizeHist(src, dst)

function destroyAllWindows()

function getTextSize(text::String, fontFace::Int32, fontScale::Float64, thickness::Int32)

function putText(img::InputArray, text::String, org::Point{Int32}, fontFace::Int32, fontScale::Float64, color::Scalar, thickness::Int32, lineType::Int32, bottomLeftOrigin::Bool)

putText(img::InputArray, text::String, org::Point{Int32}, fontFace::Int32, fontScale::Float64, color::Scalar; thickness::Int32 = Int32(1), lineType::Int32 = Int32(LINE_8), bottomLeftOrigin::Bool = (false)) = putText(img, text, org, fontFace, fontScale, color, thickness, lineType, bottomLeftOrigin)

```

This is a list of functions implemented from the `dnn` module.

```
function finalize(cobj::cv_Ptr{T}, inputs::Array{InputArray, 1}, outputs::Array{InputArray, 1}) where {T <: dnn_Layer}

finalize(cobj::cv_Ptr{T}, inputs::Array{InputArray, 1}; outputs::Array{InputArray, 1} = (Array{InputArray, 1}())) where {T <: dnn_Layer} = finalize(cobj, inputs, outputs)

function outputNameToIndex(cobj::cv_Ptr{T}, outputName::String) where {T <: dnn_Layer}

function empty(cobj::dnn_Net)

function dump(cobj::dnn_Net)

function dumpToFile(cobj::dnn_Net, path::String)

function getLayerId(cobj::dnn_Net, layer::String)

function getLayerNames(cobj::dnn_Net)

function getLayer(cobj::dnn_Net, layerId::dnn_LayerId)

function connect(cobj::dnn_Net, outPin::String, inpPin::String)

function setInputsNames(cobj::dnn_Net, inputBlobNames::Array{String, 1})

function setInputShape(cobj::dnn_Net, inputName::String, shape::Array{Int32, 1})

function forward(cobj::dnn_Net, outputName::String)

forward(cobj::dnn_Net; outputName::String = (String())) = forward(cobj, outputName)
function forward(cobj::dnn_Net, outputBlobs::Array{InputArray, 1}, outputName::String)

function forward(cobj::dnn_Net, outBlobNames::Array{String, 1}, outputBlobs::Array{InputArray, 1})

forward(cobj::dnn_Net, outBlobNames::Array{String, 1}; outputBlobs::Array{InputArray, 1} = (Array{InputArray, 1}())) = forward(cobj, outBlobNames, outputBlobs)
function forwardAsync(cobj::dnn_Net, outputName::String)

forwardAsync(cobj::dnn_Net; outputName::String = (String())) = forwardAsync(cobj, outputName)
function setHalideScheduler(cobj::dnn_Net, scheduler::String)

function setPreferableBack(cobj::dnn_Net, backId::Int32)

function setPreferableTarget(cobj::dnn_Net, targetId::Int32)

function setInput(cobj::dnn_Net, blob::InputArray, name::String, scalefactor::Float64, mean::Scalar)

setInput(cobj::dnn_Net, blob::InputArray; name::String = (""), scalefactor::Float64 = Float64(1.0), mean::Scalar = ()) = setInput(cobj, blob, name, scalefactor, mean)
function setParam(cobj::dnn_Net, layer::dnn_LayerId, numParam::Int32, blob::InputArray)

function getParam(cobj::dnn_Net, layer::dnn_LayerId, numParam::Int32)

getParam(cobj::dnn_Net, layer::dnn_LayerId; numParam::Int32 = Int32(0)) = getParam(cobj, layer, numParam)
function getUnconnectedOutLayers(cobj::dnn_Net)

function getUnconnectedOutLayersNames(cobj::dnn_Net)

function getLayersShapes(cobj::dnn_Net, netInputShapes::Array{Array{Int32, 1}, 1})

function getLayersShapes(cobj::dnn_Net, netInputShape::Array{Int32, 1})

function getFLOPS(cobj::dnn_Net, netInputShapes::Array{Array{Int32, 1}, 1})

function getFLOPS(cobj::dnn_Net, netInputShape::Array{Int32, 1})

function getFLOPS(cobj::dnn_Net, layerId::Int32, netInputShapes::Array{Array{Int32, 1}, 1})

function getFLOPS(cobj::dnn_Net, layerId::Int32, netInputShape::Array{Int32, 1})

function getLayerTypes(cobj::dnn_Net)

function getLayersCount(cobj::dnn_Net, layerType::String)

function getMemoryConsumption(cobj::dnn_Net, netInputShape::Array{Int32, 1})

function getMemoryConsumption(cobj::dnn_Net, layerId::Int32, netInputShapes::Array{Array{Int32, 1}, 1})

function getMemoryConsumption(cobj::dnn_Net, layerId::Int32, netInputShape::Array{Int32, 1})

function enableFusion(cobj::dnn_Net, fusion::Bool)

function getPerfProfile(cobj::dnn_Net)

function setInputSize(cobj::dnn_Model, size::Size)

function setInputSize(cobj::dnn_Model, width::Int32, height::Int32)

function setInputMean(cobj::dnn_Model, mean::Scalar)

function setInputScale(cobj::dnn_Model, scale::Float64)

function setInputCrop(cobj::dnn_Model, crop::Bool)

function setInputSwapRB(cobj::dnn_Model, swapRB::Bool)

function setInputParams(cobj::dnn_Model, scale::Float64, size::Size, mean::Scalar, swapRB::Bool, crop::Bool)

setInputParams(cobj::dnn_Model; scale::Float64 = Float64(1.0), size::Size = (SizeOP()), mean::Scalar = (), swapRB::Bool = (false), crop::Bool = (false)) = setInputParams(cobj, scale, size, mean, swapRB, crop)
function predict(cobj::dnn_Model, frame::InputArray, outs::Array{InputArray, 1})

predict(cobj::dnn_Model, frame::InputArray; outs::Array{InputArray, 1} = (Array{InputArray, 1}())) = predict(cobj, frame, outs)
function dnn_Model(model::String, config::String)

dnn_Model(model::String; config::String = ("")) = dnn_Model(model, config)
function dnn_Model(network::dnn_Net)

function classify(cobj::dnn_ClassificationModel, frame::InputArray)

function dnn_ClassificationModel(model::String, config::String)

dnn_ClassificationModel(model::String; config::String = ("")) = dnn_ClassificationModel(model, config)
function dnn_ClassificationModel(network::dnn_Net)

function estimate(cobj::dnn_KeypointsModel, frame::InputArray, thresh::Float32)

estimate(cobj::dnn_KeypointsModel, frame::InputArray; thresh::Float32 = Float32(0.5)) = estimate(cobj, frame, thresh)
function dnn_KeypointsModel(model::String, config::String)

dnn_KeypointsModel(model::String; config::String = ("")) = dnn_KeypointsModel(model, config)
function dnn_KeypointsModel(network::dnn_Net)

function segment(cobj::dnn_SegmentationModel, frame::InputArray, mask::InputArray)

segment(cobj::dnn_SegmentationModel, frame::InputArray; mask::InputArray = (CxxMat())) = segment(cobj, frame, mask)
function dnn_SegmentationModel(model::String, config::String)

dnn_SegmentationModel(model::String; config::String = ("")) = dnn_SegmentationModel(model, config)
function dnn_SegmentationModel(network::dnn_Net)

function detect(cobj::dnn_DetectionModel, frame::InputArray, confThreshold::Float32, nmsThreshold::Float32)

detect(cobj::dnn_DetectionModel, frame::InputArray; confThreshold::Float32 = Float32(0.5), nmsThreshold::Float32 = Float32(0.0)) = detect(cobj, frame, confThreshold, nmsThreshold)
function dnn_DetectionModel(model::String, config::String)

dnn_DetectionModel(model::String; config::String = ("")) = dnn_DetectionModel(model, config)
function dnn_DetectionModel(network::dnn_Net)

function Net_readFromModelOptimizer(xml::String, bin::String)

function Net_readFromModelOptimizer(bufferModelConfig::Array{UInt8, 1}, bufferWeights::Array{UInt8, 1})

function readNetFromDarknet(cfgFile::String, darknetModel::String)

readNetFromDarknet(cfgFile::String; darknetModel::String = (String())) = readNetFromDarknet(cfgFile, darknetModel)
function readNetFromDarknet(bufferCfg::Array{UInt8, 1}, bufferModel::Array{UInt8, 1})

readNetFromDarknet(bufferCfg::Array{UInt8, 1}; bufferModel::Array{UInt8, 1} = (stdggvectoriUInt8kOP())) = readNetFromDarknet(bufferCfg, bufferModel)
function readNetFromCaffe(prototxt::String, caffeModel::String)

readNetFromCaffe(prototxt::String; caffeModel::String = (String())) = readNetFromCaffe(prototxt, caffeModel)
function readNetFromCaffe(bufferProto::Array{UInt8, 1}, bufferModel::Array{UInt8, 1})

readNetFromCaffe(bufferProto::Array{UInt8, 1}; bufferModel::Array{UInt8, 1} = (stdggvectoriUInt8kOP())) = readNetFromCaffe(bufferProto, bufferModel)
function readNetFromTensorflow(model::String, config::String)

readNetFromTensorflow(model::String; config::String = (String())) = readNetFromTensorflow(model, config)
function readNetFromTensorflow(bufferModel::Array{UInt8, 1}, bufferConfig::Array{UInt8, 1})

readNetFromTensorflow(bufferModel::Array{UInt8, 1}; bufferConfig::Array{UInt8, 1} = (stdggvectoriUInt8kOP())) = readNetFromTensorflow(bufferModel, bufferConfig)
function readNetFromTorch(model::String, isBinary::Bool, evaluate::Bool)

readNetFromTorch(model::String; isBinary::Bool = (true), evaluate::Bool = (true)) = readNetFromTorch(model, isBinary, evaluate)
function readNet(model::String, config::String, framework::String)

readNet(model::String; config::String = (""), framework::String = ("")) = readNet(model, config, framework)
function readNet(framework::String, bufferModel::Array{UInt8, 1}, bufferConfig::Array{UInt8, 1})

readNet(framework::String, bufferModel::Array{UInt8, 1}; bufferConfig::Array{UInt8, 1} = (stdggvectoriUInt8kOP())) = readNet(framework, bufferModel, bufferConfig)
function readTorchBlob(filename::String, isBinary::Bool)

readTorchBlob(filename::String; isBinary::Bool = (true)) = readTorchBlob(filename, isBinary)
function readNetFromModelOptimizer(xml::String, bin::String)

function readNetFromModelOptimizer(bufferModelConfig::Array{UInt8, 1}, bufferWeights::Array{UInt8, 1})

function readNetFromONNX(onnxFile::String)

function readNetFromONNX(buffer::Array{UInt8, 1})

function readTensorFromONNX(path::String)

function blobFromImage(image::InputArray, scalefactor::Float64, size::Size, mean::Scalar, swapRB::Bool, crop::Bool, ddepth::Int32)

blobFromImage(image::InputArray; scalefactor::Float64 = Float64(1.0), size::Size = (SizeOP()), mean::Scalar = (), swapRB::Bool = (false), crop::Bool = (false), ddepth::Int32 = Int32(CV_32F)) = blobFromImage(image, scalefactor, size, mean, swapRB, crop, ddepth)
function blobFromImages(images::Array{InputArray, 1}, scalefactor::Float64, size::Size, mean::Scalar, swapRB::Bool, crop::Bool, ddepth::Int32)

blobFromImages(images::Array{InputArray, 1}; scalefactor::Float64 = Float64(1.0), size::Size = (SizeOP()), mean::Scalar = (), swapRB::Bool = (false), crop::Bool = (false), ddepth::Int32 = Int32(CV_32F)) = blobFromImages(images, scalefactor, size, mean, swapRB, crop, ddepth)
function imagesFromBlob(blob_::InputArray, images_::Array{InputArray, 1})

imagesFromBlob(blob_::InputArray; images_::Array{InputArray, 1} = (Array{InputArray, 1}())) = imagesFromBlob(blob_, images_)
function shrinkCaffeModel(src::String, dst::String, layersTypes::Array{String, 1})

shrinkCaffeModel(src::String, dst::String; layersTypes::Array{String, 1} = (stdggvectoriStringkOP())) = shrinkCaffeModel(src, dst, layersTypes)
function writeTextGraph(model::String, output::String)

function NMSBoxes(bboxes::Array{Rect{Float64}, 1}, scores::Array{Float32, 1}, score_threshold::Float32, nms_threshold::Float32, eta::Float32, top_k::Int32)

NMSBoxes(bboxes::Array{Rect{Float64}, 1}, scores::Array{Float32, 1}, score_threshold::Float32, nms_threshold::Float32; eta::Float32 = Float32(1.0), top_k::Int32 = Int32(0)) = NMSBoxes(bboxes, scores, score_threshold, nms_threshold, eta, top_k)
function NMSBoxesRotated(bboxes::Array{RotatedRect, 1}, scores::Array{Float32, 1}, score_threshold::Float32, nms_threshold::Float32, eta::Float32, top_k::Int32)

NMSBoxesRotated(bboxes::Array{RotatedRect, 1}, scores::Array{Float32, 1}, score_threshold::Float32, nms_threshold::Float32; eta::Float32 = Float32(1.0), top_k::Int32 = Int32(0)) = NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold, eta, top_k)
```