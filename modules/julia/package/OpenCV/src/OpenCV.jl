module OpenCV

include("typestructs.jl")
include("Vec.jl")
const dtypes = Union{UInt8, Int8, UInt16, Int16, Int32, Float32, Float64}
const size_t = UInt

using CxxWrap
@wrapmodule(joinpath(@__DIR__,"lib","libopencv_julia"), :cv_wrap)
function __init__()
    @initcxx
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


# TODO: Vector to array conversion. There's some problems with the copy constructor.

# function julia_to_cpp(var::Array{T, 1}) where {T}
#     ret = CxxWrap.StdLib.StdVector{T}()
#     for x in var
#         push!(ret, julia_to_cpp(x)) # When converting an array keep expected type as final type.
#     end
#     return ret
# end

# function cpp_to_julia(var::CxxWrap.StdLib.StdVector{T}) where {T}
#     ret = Array{T, 1}()
#     for x in var
#         push!(ret, cpp_to_julia(x))
#     end
#     return ret
# end

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

# using StaticArrays

include("cv_wrap.jl")
end
