import ColorTypes
import ImageSegmentation
export EmbColor

# a ColorType where the components are stored in a vector
# e.g. EmbColor{Float32,3}(rand(Float32,3))
struct EmbColor{T<:AbstractFloat,N} <: ColorTypes.Color{T,N} #where T<:AbstractFloat
    x::AbstractVector{T}
end

# add our own show method so that ColorTypes show isn't used
function Base.Multimedia.show(io::IO, vec::EmbColor)
    print(io, vec.x)
end

# Juno (atom) uses ColorTypes.RGB24 to try and display the vector
function Base.convert(::Type{ColorTypes.RGB24}, vec::T) where T<:EmbColor
    N = length(vec.x)
    @assert N > 3
    ColorTypes.RGB24(vec.x[1:3]...)
end

# disable showing a matrix of colors for now, TODO
Base.Multimedia.showable(::MIME"image/png", img::AbstractMatrix{EmbColor{T,N}}) where {T<:AbstractFloat,N} = false

# math routines needed for Felzenszwalb algorithm
Base.convert(::Type{AbstractArray{T,1}}, vec::EmbColor{T,N}) where {T,N} = vec.x
Base.:-(c1::EmbColor{T,N}, c2::EmbColor{T,N}) where {T,N} = EmbColor((c1.x - c2.x))
Base.:+(c1::EmbColor{T,N}, c2::EmbColor{T,N}) where {T,N} = EmbColor((c1.x + c2.x))
Base.:/(c1::EmbColor{T,N}, i::Number) where {T,N} = EmbColor(c1.x ./ i)
Base.abs2(c::EmbColor{T,N}) where {T,N} = abs2(c.x)
Base.iterate(c::EmbColor{T,N}, i=1) where {T,N} = iterate(c.x, i)
Base.zero(::Type{EmbColor{T,N}}) where {T,N} = EmbColor{T,N}(zeros(T, N))

# Felzenszwalb doesn't use `eltype` for some reason, but instead has this
# `meantype` function so we implement our own version here
ImageSegmentation.meantype(::Type{EmbColor{T,N}}) where {T<:AbstractFloat,N} = EmbColor{T,N}

# enable EmbColor(rand(Float32, 3)) construction
EmbColor(x::AbstractVector{T}) where T<:AbstractFloat = EmbColor{T,length(x)}(x)

function EmbColor(x::AbstractArray{T,3}) where T<:AbstractFloat
    nx, ny, nemb = size(x)
    # eventually julia will have a better way of doing this (https://github.com/JuliaLang/julia/pull/32310)
    #  map(f, eachslice(v, dims=(2,3)))
    [EmbColor(x[i,j,:]) for i in 1:nx, j in 1:ny]
end

function segment_array(field::Array{T,3}; k=1.0) where T<:AbstractFloat
    emb_field = SpatialClustering.EmbColor(field)
    segments = ImageSegmentation.felzenszwalb(emb_field, k, 0)
    segments.image_indexmap
end
