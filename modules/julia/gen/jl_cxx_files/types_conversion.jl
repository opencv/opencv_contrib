function cpp_to_julia(var::CxxScalar{T}) where {T}
    var = Vec{T, 4}(var)
    return (var[1], var[2], var[3], var[4])
end

function cpp_to_julia(var::CxxVec{T, N}) where {T, N}
    return Vec{T, N}(var)
end

function julia_to_cpp(sc::Scalar)
    if size(sc,1)==0
        return CxxScalar{Float64}(0,0,0,0)
    elseif size(sc, 1) == 1
        return CxxScalar{Float64}(Float64(sc[1]), 0, 0, 0)
    elseif size(sc,1) == 2
        return CxxScalar{Float64}(Float64(sc[1]), Float64(sc[2]), 0, 0)
    elseif size(sc,1) == 3
        return CxxScalar{Float64}(Float64(sc[1]), Float64(sc[2]), Float64(sc[3]), 0)
    end
    return CxxScalar{Float64}(Float64(sc[1]), Float64(sc[2]), Float64(sc[3]), Float64(sc[4]))
end

function julia_to_cpp(vec::Vec{T, N}) where {T, N}
    return CxxVec{T, N}(Base.pointer(vec))
end

function julia_to_cpp(var::Array{T, 1}) where {T <: Scalar}
    ret = CxxWrap.StdVector{CxxScalar}()
    for x in var
        push!(ret, julia_to_cpp(x))
    end
    return ret
end

function julia_to_cpp(var::Array{Vec{T, N}, 1}) where {T, N}
    ret = CxxWrap.StdVector{CxxVec{T, N}}()
    for x in var
        push!(ret, julia_to_cpp(x))
    end
    return ret
end

function julia_to_cpp(var::Array{T, 1}) where {T}
    if size(var, 1) == 0
        return CxxWrap.StdVector{T}()
    end
    ret = CxxWrap.StdVector{typeof(julia_to_cpp(var[1]))}()
    for x in var
        push!(ret, julia_to_cpp(x))
    end
    return ret
end

function cpp_to_julia(var::CxxWrap.StdVector{T}) where {T <: CxxScalar}
    ret = Array{Scalar, 1}()
    for x in var
        push!(ret, cpp_to_julia(x))
    end
    return ret
end

function cpp_to_julia(var::CxxWrap.StdVector{CxxVec{T, N}}) where {T, N}
    ret = Array{Vec{T, N}, 1}()
    for x in var
        push!(ret, cpp_to_julia(x))
    end
    return ret
end

function cpp_to_julia(var::CxxWrap.StdVector{T}) where {T}
    if size(var, 1) == 0
        return Array{T, 1}()
    end
    ret = Array{typeof(cpp_to_julia(var[1])), 1}()
    for x in var
        push!(ret, cpp_to_julia(x))
    end
    return ret
end