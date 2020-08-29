#Adapted from IndirectArray

struct Vec{T, N} <: AbstractArray{T,1}
    cpp_object
    data::AbstractArray{T, 1}
    cpp_allocated::Bool
    @inline function Vec{T, N}(obj) where {T, N}

        new{T, N}(obj, Base.unsafe_wrap(Array{T, 1}, Ptr{T}(obj.cpp_object), N), true)
    end

    @inline function Vec{T, N}(data_raw::AbstractArray{T, 1}) where {T, N}
        if size(data_raw, 1) != N
            throw("Array is improper Size for Vec declared")
        end
        new{T, N}(nothing, data_raw, false)
    end
end

function Base.deepcopy_internal(x::Vec{T,N}, y::IdDict) where {T, N}
    if haskey(y, x)
        return y[x]
    end
    ret = Base.copy(x)
    y[x] = ret
    return ret
end

Base.size(A::Vec) = Base.size(A.data)
Base.axes(A::Vec) = Base.axes(A.data)
Base.IndexStyle(::Type{Vec{T,N}}) where {T, N} = IndexLinear()

Base.strides(A::Vec{T,N}) where {T, N} = (1)
function Base.copy(A::Vec{T,N}) where {T, N}
    return Vec{T, N}(copy(A.data))
end
Base.pointer(A::Vec) = Base.pointer(A.data)

Base.unsafe_convert(::Type{Ptr{T}}, A::Vec{S, N}) where {T, S, N} = Base.unsafe_convert(Ptr{T}, A.data)

@inline function Base.getindex(A::Vec{T,N}, I::Int) where {T, N}
    @boundscheck checkbounds(A.data, I)
    return A.data[I]
end

@inline function Base.setindex!(A::Vec, x, I::Int)
    @boundscheck checkbounds(A.data, I)
    A.data[I] = x
    return A
end