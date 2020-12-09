import ImageSegmentation
import ColorTypes
import Base
using ImageSegmentation

struct TwoColor{T} <: ColorTypes.Color{T,2}
    red::T
    green::T
end

# implement method for `convert` so that Atom can display the color
Base.convert(::Type{ColorTypes.RGB24}, x::TwoColor{T}) where T = ColorTypes.RGB24(x.red, x.green, 0.0)

# disable showing a matrix of colors for now
Base.Multimedia.showable(::MIME"image/png", img::AbstractMatrix{TwoColor{T}}) where T = false

# methods requiered to able to run felzenszwalb algorithm
Base.zero(::Type{TwoColor{T}}) where T = TwoColor(zero(T), zero(T))
Base.:/(c::TwoColor{T}, x::Number) where T = TwoColor(T(c.red/x), T(c.green/x))
Base.:-(c1::TwoColor{T}, c2::TwoColor{T}) where T = TwoColor{T}(c1.red - c2.red, c1.green - c2.green)
Base.:+(c1::TwoColor{T}, c2::TwoColor{T}) where T = TwoColor{T}(c1.red + c2.red, c1.green + c2.green)
TwoColor{T}(c::TwoColor{T}) where T = TwoColor{T}(c.red, c.green)
Base.abs2(c::TwoColor{Float64}) = c.red*c.red + c.green*c.green

function Base.iterate(c::TwoColor, i=1)
    fns = fieldnames(TwoColor)
    if (i % UInt) - 1 < length(fns)
        (getfield(c, fns[i]), i + 1)
    else
        nothing
    end
end
