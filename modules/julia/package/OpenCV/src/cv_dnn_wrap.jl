
function Base.getproperty(m::dnn_Layer, s::Symbol)
    if s==:blobs
        return cpp_to_julia(jlopencv_Layer_get_blobs(m))
    end
    if s==:name
        return cpp_to_julia(jlopencv_Layer_get_name(m))
    end
    if s==:type
        return cpp_to_julia(jlopencv_Layer_get_type(m))
    end
    if s==:preferableTarget
        return cpp_to_julia(jlopencv_Layer_get_preferableTarget(m))
    end
    return Base.getfield(m, s)
end

function Base.setproperty!(m::dnn_Layer, s::Symbol, v)
    if s==:name
        jlopencv_Layer_set_name(m, julia_to_cpp(v))
    end
    if s==:type
        jlopencv_Layer_set_type(m, julia_to_cpp(v))
    end
    if s==:preferableTarget
        jlopencv_Layer_set_preferableTarget(m, julia_to_cpp(v))
    end
    return Base.setfield!(m, s, v)
end

function finalize(cobj::cv_Ptr{T}, inputs::Array{InputArray, 1}, outputs::Array{InputArray, 1}) where {T <: dnn_Layer}
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Layer_cv_dnn_Layer_finalize(julia_to_cpp(cobj),julia_to_cpp(inputs),julia_to_cpp(outputs)))
end
finalize(cobj::cv_Ptr{T}, inputs::Array{InputArray, 1}; outputs::Array{InputArray, 1} = (Array{InputArray, 1}())) where {T <: dnn_Layer} = finalize(cobj, inputs, outputs)

# function run(cobj::cv_Ptr{T}, inputs::Array{InputArray, 1}, internals::Array{InputArray, 1}, outputs::Array{InputArray, 1}) where {T <: dnn_Layer}
# 	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Layer_cv_dnn_Layer_run(julia_to_cpp(cobj),julia_to_cpp(inputs),julia_to_cpp(internals),julia_to_cpp(outputs)))
# end
# run(cobj::cv_Ptr{T}, inputs::Array{InputArray, 1}, internals::Array{InputArray, 1}; outputs::Array{InputArray, 1} = (Array{InputArray, 1}())) where {T <: dnn_Layer} = run(cobj, inputs, internals, outputs)

function outputNameToIndex(cobj::cv_Ptr{T}, outputName::String) where {T <: dnn_Layer}
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Layer_cv_dnn_Layer_outputNameToIndex(julia_to_cpp(cobj),julia_to_cpp(outputName)))
end
function Base.getproperty(m::dnn_Net, s::Symbol)
    return Base.getfield(m, s)
end
function Base.setproperty!(m::dnn_Net, s::Symbol, v)
    return Base.setfield!(m, s, v)
end

function empty(cobj::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_empty(julia_to_cpp(cobj)))
end

function dump(cobj::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_dump(julia_to_cpp(cobj)))
end

function dumpToFile(cobj::dnn_Net, path::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_dumpToFile(julia_to_cpp(cobj),julia_to_cpp(path)))
end

function getLayerId(cobj::dnn_Net, layer::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getLayerId(julia_to_cpp(cobj),julia_to_cpp(layer)))
end

function getLayerNames(cobj::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getLayerNames(julia_to_cpp(cobj)))
end

function getLayer(cobj::dnn_Net, layerId::dnn_LayerId)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getLayer(julia_to_cpp(cobj),julia_to_cpp(layerId)))
end

function connect(cobj::dnn_Net, outPin::String, inpPin::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_connect(julia_to_cpp(cobj),julia_to_cpp(outPin),julia_to_cpp(inpPin)))
end

function setInputsNames(cobj::dnn_Net, inputBlobNames::Array{String, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_setInputsNames(julia_to_cpp(cobj),julia_to_cpp(inputBlobNames)))
end

function setInputShape(cobj::dnn_Net, inputName::String, shape::Array{Int32, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_setInputShape(julia_to_cpp(cobj),julia_to_cpp(inputName),julia_to_cpp(shape)))
end

function forward(cobj::dnn_Net, outputName::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_forward(julia_to_cpp(cobj),julia_to_cpp(outputName)))
end
forward(cobj::dnn_Net; outputName::String = (String())) = forward(cobj, outputName)

function forward(cobj::dnn_Net, outputBlobs::Array{InputArray, 1}, outputName::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_forward(julia_to_cpp(cobj),julia_to_cpp(outputBlobs),julia_to_cpp(outputName)))
end

function forward(cobj::dnn_Net, outBlobNames::Array{String, 1}, outputBlobs::Array{InputArray, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_forward(julia_to_cpp(cobj),julia_to_cpp(outBlobNames),julia_to_cpp(outputBlobs)))
end
forward(cobj::dnn_Net, outBlobNames::Array{String, 1}; outputBlobs::Array{InputArray, 1} = (Array{InputArray, 1}())) = forward(cobj, outBlobNames, outputBlobs)

function forwardAsync(cobj::dnn_Net, outputName::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_forwardAsync(julia_to_cpp(cobj),julia_to_cpp(outputName)))
end
forwardAsync(cobj::dnn_Net; outputName::String = (String())) = forwardAsync(cobj, outputName)

function setHalideScheduler(cobj::dnn_Net, scheduler::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_setHalideScheduler(julia_to_cpp(cobj),julia_to_cpp(scheduler)))
end

function setPreferableBackend(cobj::dnn_Net, backendId::Int32)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_setPreferableBackend(julia_to_cpp(cobj),julia_to_cpp(backendId)))
end

function setPreferableTarget(cobj::dnn_Net, targetId::Int32)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_setPreferableTarget(julia_to_cpp(cobj),julia_to_cpp(targetId)))
end

function setInput(cobj::dnn_Net, blob::InputArray, name::String, scalefactor::Float64, mean::Scalar)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_setInput(julia_to_cpp(cobj),julia_to_cpp(blob),julia_to_cpp(name),julia_to_cpp(scalefactor),julia_to_cpp(mean)))
end
setInput(cobj::dnn_Net, blob::InputArray; name::String = (""), scalefactor::Float64 = Float64(1.0), mean::Scalar = ()) = setInput(cobj, blob, name, scalefactor, mean)

function setParam(cobj::dnn_Net, layer::dnn_LayerId, numParam::Int32, blob::InputArray)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_setParam(julia_to_cpp(cobj),julia_to_cpp(layer),julia_to_cpp(numParam),julia_to_cpp(blob)))
end

function getParam(cobj::dnn_Net, layer::dnn_LayerId, numParam::Int32)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getParam(julia_to_cpp(cobj),julia_to_cpp(layer),julia_to_cpp(numParam)))
end
getParam(cobj::dnn_Net, layer::dnn_LayerId; numParam::Int32 = Int32(0)) = getParam(cobj, layer, numParam)

function getUnconnectedOutLayers(cobj::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getUnconnectedOutLayers(julia_to_cpp(cobj)))
end

function getUnconnectedOutLayersNames(cobj::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getUnconnectedOutLayersNames(julia_to_cpp(cobj)))
end

function getLayersShapes(cobj::dnn_Net, netInputShapes::Array{Array{Int32, 1}, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getLayersShapes(julia_to_cpp(cobj),julia_to_cpp(netInputShapes)))
end

function getLayersShapes(cobj::dnn_Net, netInputShape::Array{Int32, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getLayersShapes(julia_to_cpp(cobj),julia_to_cpp(netInputShape)))
end

function getFLOPS(cobj::dnn_Net, netInputShapes::Array{Array{Int32, 1}, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getFLOPS(julia_to_cpp(cobj),julia_to_cpp(netInputShapes)))
end

function getFLOPS(cobj::dnn_Net, netInputShape::Array{Int32, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getFLOPS(julia_to_cpp(cobj),julia_to_cpp(netInputShape)))
end

function getFLOPS(cobj::dnn_Net, layerId::Int32, netInputShapes::Array{Array{Int32, 1}, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getFLOPS(julia_to_cpp(cobj),julia_to_cpp(layerId),julia_to_cpp(netInputShapes)))
end

function getFLOPS(cobj::dnn_Net, layerId::Int32, netInputShape::Array{Int32, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getFLOPS(julia_to_cpp(cobj),julia_to_cpp(layerId),julia_to_cpp(netInputShape)))
end

function getLayerTypes(cobj::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getLayerTypes(julia_to_cpp(cobj)))
end

function getLayersCount(cobj::dnn_Net, layerType::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getLayersCount(julia_to_cpp(cobj),julia_to_cpp(layerType)))
end

function getMemoryConsumption(cobj::dnn_Net, netInputShape::Array{Int32, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getMemoryConsumption(julia_to_cpp(cobj),julia_to_cpp(netInputShape)))
end

function getMemoryConsumption(cobj::dnn_Net, layerId::Int32, netInputShapes::Array{Array{Int32, 1}, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getMemoryConsumption(julia_to_cpp(cobj),julia_to_cpp(layerId),julia_to_cpp(netInputShapes)))
end

function getMemoryConsumption(cobj::dnn_Net, layerId::Int32, netInputShape::Array{Int32, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getMemoryConsumption(julia_to_cpp(cobj),julia_to_cpp(layerId),julia_to_cpp(netInputShape)))
end

function enableFusion(cobj::dnn_Net, fusion::Bool)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_enableFusion(julia_to_cpp(cobj),julia_to_cpp(fusion)))
end

function getPerfProfile(cobj::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_cv_dnn_Net_getPerfProfile(julia_to_cpp(cobj)))
end

function Base.getproperty(m::dnn_Model, s::Symbol)
    return Base.getfield(m, s)
end
function Base.setproperty!(m::dnn_Model, s::Symbol, v)
    return Base.setfield!(m, s, v)
end

function setInputSize(cobj::dnn_Model, size::Size)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Model_cv_dnn_Model_setInputSize(julia_to_cpp(cobj),julia_to_cpp(size)))
end

function setInputSize(cobj::dnn_Model, width::Int32, height::Int32)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Model_cv_dnn_Model_setInputSize(julia_to_cpp(cobj),julia_to_cpp(width),julia_to_cpp(height)))
end

function setInputMean(cobj::dnn_Model, mean::Scalar)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Model_cv_dnn_Model_setInputMean(julia_to_cpp(cobj),julia_to_cpp(mean)))
end

function setInputScale(cobj::dnn_Model, scale::Float64)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Model_cv_dnn_Model_setInputScale(julia_to_cpp(cobj),julia_to_cpp(scale)))
end

function setInputCrop(cobj::dnn_Model, crop::Bool)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Model_cv_dnn_Model_setInputCrop(julia_to_cpp(cobj),julia_to_cpp(crop)))
end

function setInputSwapRB(cobj::dnn_Model, swapRB::Bool)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Model_cv_dnn_Model_setInputSwapRB(julia_to_cpp(cobj),julia_to_cpp(swapRB)))
end

function setInputParams(cobj::dnn_Model, scale::Float64, size::Size, mean::Scalar, swapRB::Bool, crop::Bool)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Model_cv_dnn_Model_setInputParams(julia_to_cpp(cobj),julia_to_cpp(scale),julia_to_cpp(size),julia_to_cpp(mean),julia_to_cpp(swapRB),julia_to_cpp(crop)))
end
setInputParams(cobj::dnn_Model; scale::Float64 = Float64(1.0), size::Size = (SizeOP()), mean::Scalar = (), swapRB::Bool = (false), crop::Bool = (false)) = setInputParams(cobj, scale, size, mean, swapRB, crop)

function predict(cobj::dnn_Model, frame::InputArray, outs::Array{InputArray, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Model_cv_dnn_Model_predict(julia_to_cpp(cobj),julia_to_cpp(frame),julia_to_cpp(outs)))
end
predict(cobj::dnn_Model, frame::InputArray; outs::Array{InputArray, 1} = (Array{InputArray, 1}())) = predict(cobj, frame, outs)

function dnn_Model(model::String, config::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Model_cv_dnn_Model_Model(julia_to_cpp(model),julia_to_cpp(config)))
end
dnn_Model(model::String; config::String = ("")) = dnn_Model(model, config)

function dnn_Model(network::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Model_cv_dnn_Model_Model(julia_to_cpp(network)))
end
function Base.getproperty(m::dnn_ClassificationModel, s::Symbol)
    return Base.getfield(m, s)
end
function Base.setproperty!(m::dnn_ClassificationModel, s::Symbol, v)
    return Base.setfield!(m, s, v)
end

function classify(cobj::dnn_ClassificationModel, frame::InputArray)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_ClassificationModel_cv_dnn_ClassificationModel_classify(julia_to_cpp(cobj),julia_to_cpp(frame)))
end

function dnn_ClassificationModel(model::String, config::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_ClassificationModel_cv_dnn_ClassificationModel_ClassificationModel(julia_to_cpp(model),julia_to_cpp(config)))
end
dnn_ClassificationModel(model::String; config::String = ("")) = dnn_ClassificationModel(model, config)

function dnn_ClassificationModel(network::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_ClassificationModel_cv_dnn_ClassificationModel_ClassificationModel(julia_to_cpp(network)))
end
function Base.getproperty(m::dnn_KeypointsModel, s::Symbol)
    return Base.getfield(m, s)
end
function Base.setproperty!(m::dnn_KeypointsModel, s::Symbol, v)
    return Base.setfield!(m, s, v)
end

function estimate(cobj::dnn_KeypointsModel, frame::InputArray, thresh::Float32)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_KeypointsModel_cv_dnn_KeypointsModel_estimate(julia_to_cpp(cobj),julia_to_cpp(frame),julia_to_cpp(thresh)))
end
estimate(cobj::dnn_KeypointsModel, frame::InputArray; thresh::Float32 = Float32(0.5)) = estimate(cobj, frame, thresh)

function dnn_KeypointsModel(model::String, config::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_KeypointsModel_cv_dnn_KeypointsModel_KeypointsModel(julia_to_cpp(model),julia_to_cpp(config)))
end
dnn_KeypointsModel(model::String; config::String = ("")) = dnn_KeypointsModel(model, config)

function dnn_KeypointsModel(network::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_KeypointsModel_cv_dnn_KeypointsModel_KeypointsModel(julia_to_cpp(network)))
end
function Base.getproperty(m::dnn_SegmentationModel, s::Symbol)
    return Base.getfield(m, s)
end
function Base.setproperty!(m::dnn_SegmentationModel, s::Symbol, v)
    return Base.setfield!(m, s, v)
end

function segment(cobj::dnn_SegmentationModel, frame::InputArray, mask::InputArray)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_SegmentationModel_cv_dnn_SegmentationModel_segment(julia_to_cpp(cobj),julia_to_cpp(frame),julia_to_cpp(mask)))
end
segment(cobj::dnn_SegmentationModel, frame::InputArray; mask::InputArray = (CxxMat())) = segment(cobj, frame, mask)

function dnn_SegmentationModel(model::String, config::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_SegmentationModel_cv_dnn_SegmentationModel_SegmentationModel(julia_to_cpp(model),julia_to_cpp(config)))
end
dnn_SegmentationModel(model::String; config::String = ("")) = dnn_SegmentationModel(model, config)

function dnn_SegmentationModel(network::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_SegmentationModel_cv_dnn_SegmentationModel_SegmentationModel(julia_to_cpp(network)))
end
function Base.getproperty(m::dnn_DetectionModel, s::Symbol)
    return Base.getfield(m, s)
end
function Base.setproperty!(m::dnn_DetectionModel, s::Symbol, v)
    return Base.setfield!(m, s, v)
end

function detect(cobj::dnn_DetectionModel, frame::InputArray, confThreshold::Float32, nmsThreshold::Float32)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_DetectionModel_cv_dnn_DetectionModel_detect(julia_to_cpp(cobj),julia_to_cpp(frame),julia_to_cpp(confThreshold),julia_to_cpp(nmsThreshold)))
end
detect(cobj::dnn_DetectionModel, frame::InputArray; confThreshold::Float32 = Float32(0.5), nmsThreshold::Float32 = Float32(0.0)) = detect(cobj, frame, confThreshold, nmsThreshold)

function dnn_DetectionModel(model::String, config::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_DetectionModel_cv_dnn_DetectionModel_DetectionModel(julia_to_cpp(model),julia_to_cpp(config)))
end
dnn_DetectionModel(model::String; config::String = ("")) = dnn_DetectionModel(model, config)

function dnn_DetectionModel(network::dnn_Net)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_DetectionModel_cv_dnn_DetectionModel_DetectionModel(julia_to_cpp(network)))
end

# function getAvailableTargets(be::dnn_Backend)
# 	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_getAvailableTargets(julia_to_cpp(be)))
# end

function Net_readFromModelOptimizer(xml::String, bin::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_readFromModelOptimizer(julia_to_cpp(xml),julia_to_cpp(bin)))
end

function Net_readFromModelOptimizer(bufferModelConfig::Array{UInt8, 1}, bufferWeights::Array{UInt8, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_Net_readFromModelOptimizer(julia_to_cpp(bufferModelConfig),julia_to_cpp(bufferWeights)))
end

function readNetFromDarknet(cfgFile::String, darknetModel::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromDarknet(julia_to_cpp(cfgFile),julia_to_cpp(darknetModel)))
end
readNetFromDarknet(cfgFile::String; darknetModel::String = (String())) = readNetFromDarknet(cfgFile, darknetModel)

function readNetFromDarknet(bufferCfg::Array{UInt8, 1}, bufferModel::Array{UInt8, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromDarknet(julia_to_cpp(bufferCfg),julia_to_cpp(bufferModel)))
end
readNetFromDarknet(bufferCfg::Array{UInt8, 1}; bufferModel::Array{UInt8, 1} = (stdggvectoriUInt8kOP())) = readNetFromDarknet(bufferCfg, bufferModel)

function readNetFromCaffe(prototxt::String, caffeModel::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromCaffe(julia_to_cpp(prototxt),julia_to_cpp(caffeModel)))
end
readNetFromCaffe(prototxt::String; caffeModel::String = (String())) = readNetFromCaffe(prototxt, caffeModel)

function readNetFromCaffe(bufferProto::Array{UInt8, 1}, bufferModel::Array{UInt8, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromCaffe(julia_to_cpp(bufferProto),julia_to_cpp(bufferModel)))
end
readNetFromCaffe(bufferProto::Array{UInt8, 1}; bufferModel::Array{UInt8, 1} = (stdggvectoriUInt8kOP())) = readNetFromCaffe(bufferProto, bufferModel)

function readNetFromTensorflow(model::String, config::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromTensorflow(julia_to_cpp(model),julia_to_cpp(config)))
end
readNetFromTensorflow(model::String; config::String = (String())) = readNetFromTensorflow(model, config)

function readNetFromTensorflow(bufferModel::Array{UInt8, 1}, bufferConfig::Array{UInt8, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromTensorflow(julia_to_cpp(bufferModel),julia_to_cpp(bufferConfig)))
end
readNetFromTensorflow(bufferModel::Array{UInt8, 1}; bufferConfig::Array{UInt8, 1} = (stdggvectoriUInt8kOP())) = readNetFromTensorflow(bufferModel, bufferConfig)

function readNetFromTorch(model::String, isBinary::Bool, evaluate::Bool)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromTorch(julia_to_cpp(model),julia_to_cpp(isBinary),julia_to_cpp(evaluate)))
end
readNetFromTorch(model::String; isBinary::Bool = (true), evaluate::Bool = (true)) = readNetFromTorch(model, isBinary, evaluate)

function readNet(model::String, config::String, framework::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNet(julia_to_cpp(model),julia_to_cpp(config),julia_to_cpp(framework)))
end
readNet(model::String; config::String = (""), framework::String = ("")) = readNet(model, config, framework)

function readNet(framework::String, bufferModel::Array{UInt8, 1}, bufferConfig::Array{UInt8, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNet(julia_to_cpp(framework),julia_to_cpp(bufferModel),julia_to_cpp(bufferConfig)))
end
readNet(framework::String, bufferModel::Array{UInt8, 1}; bufferConfig::Array{UInt8, 1} = (stdggvectoriUInt8kOP())) = readNet(framework, bufferModel, bufferConfig)

function readTorchBlob(filename::String, isBinary::Bool)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readTorchBlob(julia_to_cpp(filename),julia_to_cpp(isBinary)))
end
readTorchBlob(filename::String; isBinary::Bool = (true)) = readTorchBlob(filename, isBinary)

function readNetFromModelOptimizer(xml::String, bin::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromModelOptimizer(julia_to_cpp(xml),julia_to_cpp(bin)))
end

function readNetFromModelOptimizer(bufferModelConfig::Array{UInt8, 1}, bufferWeights::Array{UInt8, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromModelOptimizer(julia_to_cpp(bufferModelConfig),julia_to_cpp(bufferWeights)))
end

function readNetFromONNX(onnxFile::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromONNX(julia_to_cpp(onnxFile)))
end

function readNetFromONNX(buffer::Array{UInt8, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readNetFromONNX(julia_to_cpp(buffer)))
end

function readTensorFromONNX(path::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_readTensorFromONNX(julia_to_cpp(path)))
end

function blobFromImage(image::InputArray, scalefactor::Float64, size::Size, mean::Scalar, swapRB::Bool, crop::Bool, ddepth::Int32)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_blobFromImage(julia_to_cpp(image),julia_to_cpp(scalefactor),julia_to_cpp(size),julia_to_cpp(mean),julia_to_cpp(swapRB),julia_to_cpp(crop),julia_to_cpp(ddepth)))
end
blobFromImage(image::InputArray; scalefactor::Float64 = Float64(1.0), size::Size = (SizeOP()), mean::Scalar = (), swapRB::Bool = (false), crop::Bool = (false), ddepth::Int32 = Int32(CV_32F)) = blobFromImage(image, scalefactor, size, mean, swapRB, crop, ddepth)

function blobFromImages(images::Array{InputArray, 1}, scalefactor::Float64, size::Size, mean::Scalar, swapRB::Bool, crop::Bool, ddepth::Int32)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_blobFromImages(julia_to_cpp(images),julia_to_cpp(scalefactor),julia_to_cpp(size),julia_to_cpp(mean),julia_to_cpp(swapRB),julia_to_cpp(crop),julia_to_cpp(ddepth)))
end
blobFromImages(images::Array{InputArray, 1}; scalefactor::Float64 = Float64(1.0), size::Size = (SizeOP()), mean::Scalar = (), swapRB::Bool = (false), crop::Bool = (false), ddepth::Int32 = Int32(CV_32F)) = blobFromImages(images, scalefactor, size, mean, swapRB, crop, ddepth)

function imagesFromBlob(blob_::InputArray, images_::Array{InputArray, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_imagesFromBlob(julia_to_cpp(blob_),julia_to_cpp(images_)))
end
imagesFromBlob(blob_::InputArray; images_::Array{InputArray, 1} = (Array{InputArray, 1}())) = imagesFromBlob(blob_, images_)

function shrinkCaffeModel(src::String, dst::String, layersTypes::Array{String, 1})
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_shrinkCaffeModel(julia_to_cpp(src),julia_to_cpp(dst),julia_to_cpp(layersTypes)))
end
shrinkCaffeModel(src::String, dst::String; layersTypes::Array{String, 1} = (stdggvectoriStringkOP())) = shrinkCaffeModel(src, dst, layersTypes)

function writeTextGraph(model::String, output::String)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_writeTextGraph(julia_to_cpp(model),julia_to_cpp(output)))
end

function NMSBoxes(bboxes::Array{Rect{Float64}, 1}, scores::Array{Float32, 1}, score_threshold::Float32, nms_threshold::Float32, eta::Float32, top_k::Int32)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_NMSBoxes(julia_to_cpp(bboxes),julia_to_cpp(scores),julia_to_cpp(score_threshold),julia_to_cpp(nms_threshold),julia_to_cpp(eta),julia_to_cpp(top_k)))
end
NMSBoxes(bboxes::Array{Rect{Float64}, 1}, scores::Array{Float32, 1}, score_threshold::Float32, nms_threshold::Float32; eta::Float32 = Float32(1.0), top_k::Int32 = Int32(0)) = NMSBoxes(bboxes, scores, score_threshold, nms_threshold, eta, top_k)

function NMSBoxesRotated(bboxes::Array{RotatedRect, 1}, scores::Array{Float32, 1}, score_threshold::Float32, nms_threshold::Float32, eta::Float32, top_k::Int32)
	return cpp_to_julia(jlopencv_cv_dnn_cv_dnn_NMSBoxes(julia_to_cpp(bboxes),julia_to_cpp(scores),julia_to_cpp(score_threshold),julia_to_cpp(nms_threshold),julia_to_cpp(eta),julia_to_cpp(top_k)))
end
NMSBoxesRotated(bboxes::Array{RotatedRect, 1}, scores::Array{Float32, 1}, score_threshold::Float32, nms_threshold::Float32; eta::Float32 = Float32(1.0), top_k::Int32 = Int32(0)) = NMSBoxesRotated(bboxes, scores, score_threshold, nms_threshold, eta, top_k)