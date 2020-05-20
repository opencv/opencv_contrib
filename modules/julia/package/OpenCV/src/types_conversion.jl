function cpp_to_julia(var::CxxScalar{T}) where {T}
    var = Vec{T, 4}(var)
    return (var[1], var[2], var[3], var[4])   
end

function cpp_to_julia(var::CxxVec{T, N}) where {T, N}
    return Vec{T, N}(var)
end

function julia_to_cpp(sc::Scalar)
    if size(sc,1)==1
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