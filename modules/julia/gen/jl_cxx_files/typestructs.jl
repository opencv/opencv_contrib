
struct Point{T}
  x::T
  y::T
end

struct Point3{T}
  x::T
  y::T
  z::T
end


struct Size{T}
  width::T
  height::T
end


struct Rect{T}
  x::T
  y::T
  width::T
  height::T
end

struct RotatedRect
    center::Point{Float32}
    size::Size{Float32}
    angle::Float32
end

struct Range
    start::Int32
    end_::Int32
end

struct TermCriteria
    type::Int32
    maxCount::Int32
    epsilon::Float64
end

struct cvComplex{T}
  re::T
  im::T
end
