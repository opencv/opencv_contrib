# using StaticArrays

include("typestructs.jl")
include("Vec.jl")
const dtypes = Union{UInt8, Int8, UInt16, Int16, Int32, Float32, Float64}
size_t = UInt64

using CxxWrap
@wrapmodule(joinpath(@__DIR__,"lib","libopencv_julia"), :cv_wrap)
function __init__()
    @initcxx

    if jlopencv_core_get_sizet()==4
        size_t = UInt32
    end
end
const Scalar = Union{Tuple{}, Tuple{Number}, Tuple{Number, Number}, Tuple{Number, Number, Number}, NTuple{4, Number}}

include("Mat.jl")

const InputArray = Union{AbstractArray{T, 3} where {T <: dtypes}, CxxMat}

include("mat_conversion.jl")
include("types_conversion.jl")

function cpp_to_julia(var)
    return var
end
function julia_to_cpp(var)
    return var
end

function cpp_to_julia(var::Tuple)
    ret_arr = Array{Any, 1}()
    for it in var
        push!(ret_arr, cpp_to_julia(it))
    end
    return tuple(ret_arr...)
end

function cpp_to_julia(var::CxxBool)
    return Bool(var)
end

function julia_to_cpp(var::Bool)
    return CxxBool(var)
end


include("cv_cxx_wrap.jl")

include("cv_manual_wrap.jl")