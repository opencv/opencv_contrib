#Adapted from IndirectArray

struct Mat{T <: dtypes} <: AbstractArray{T,3}
    mat
    data_raw
    data

    @inline function Mat{T}(mat, data_raw::AbstractArray{T,3}) where {T <: dtypes}
        data = reinterpret(T, data_raw)
        new{T}(mat, data_raw, data)
    end

    @inline function Mat(data_raw::AbstractArray{T, 3}) where {T <: dtypes}
        data = reinterpret(T, data_raw)
        mat = nothing
        new{T}(mat, data_raw, data)
    end
end

function Base.deepcopy_internal(x::Mat{T}, y::IdDict) where {T}
    if haskey(y, x)
        return y[x]
    end
    ret = Base.copy(x)
    y[x] = ret
    return ret
end

Base.size(A::Mat) = size(A.data)
Base.axes(A::Mat) = axes(A.data)
Base.IndexStyle(::Type{Mat{T}}) where {T} = IndexCartesian()

Base.strides(A::Mat{T}) where {T} = strides(A.data)
Base.copy(A::Mat{T}) where {T} = Mat(copy(A.data_raw))
Base.pointer(A::Mat) = Base.pointer(A.data)

Base.unsafe_convert(::Type{Ptr{T}}, A::Mat{S}) where {T, S} = Base.unsafe_convert(Ptr{T}, A.data)

@inline function Base.getindex(A::Mat{T}, I::Vararg{Int,3}) where {T}
    @boundscheck checkbounds(A.data, I...)
    @inbounds ret = A.data[I...]
    ret
end

@inline function Base.setindex!(A::Mat, x, I::Vararg{Int,3})
    @boundscheck checkbounds(A.data, I...)
    A.data[I...] = x
    return A
end
