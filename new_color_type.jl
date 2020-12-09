import ImageSegmentation
import ColorTypes
import Base
using ImageSegmentation

img_size = (100, 100)
field = ColorTypes.RGB.(rand(Float32, img_size), rand(Float32, img_size), rand(Float32, img_size))

ImageSegmentation.felzenszwalb(field, 0.8)


struct TwoColor{T} <: ColorTypes.Color{T,2}
    red::T
    green::T
end

dtype = Float64

# implement method for `convert` so that Atom can display the color
Base.convert(::Type{ColorTypes.RGB24}, x::TwoColor{T}) where T = ColorTypes.RGB24(x.red, x.green, 0.0)
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

field2 = TwoColor.(rand(dtype, img_size), rand(dtype, img_size))

ImageSegmentation.felzenszwalb(field2, 0.8)

iterate(ColorTypes.RGB(1.0, 1.0, 1.0))

@edit sum(abs2, field[1,1])

sum(abs2, field2[1,1])

@edit ColorTypes.RGB(1., 1., 1.)

Color = ColorTypes.Color

_colon(I::CartesianIndex{N}, J::CartesianIndex{N}) where N =
    map((i,j) -> i:j, Tuple(I), Tuple(J))

function felzenszwalbnew(img::AbstractArray{T, 2}, k::Real, min_size::Int = 0) where T<:Union{Real,Color}

    rows, cols = size(img)
    num_vertices = rows*cols
    num_edges = 4*rows*cols - 3*rows - 3*cols + 2
    edges = Array{ImageEdge}(undef, num_edges)

    R = CartesianIndices(size(img))
    I1, Iend = first(R), last(R)
    num = 1
    for I in R
        for J in CartesianIndices(_colon(max(I1, I-I1), min(Iend, I+I1)))
            if I >= J
                continue
            end
            edges[num] = ImageEdge((I[2]-1)*rows+I[1], (J[2]-1)*rows+J[1], sqrt(sum(abs2,(img[I])-(img[J]))))
            #edges[num] = ImageEdge((I[2]-1)*rows+I[1], (J[2]-1)*rows+J[1], rand(Float32))
            num += 1
        end
    end

    index_map, num_segments = felzenszwalb(edges, num_vertices, k, min_size)

    result              = similar(img, Int)
    labels              = Array(1:num_segments)
    region_means        = Dict{Int, T}()
    region_pix_count    = Dict{Int, Int}()

    for j in axes(img, 2)
        for i in axes(img, 1)
            result[i, j] = index_map[(j-1)*rows+i]
            region_pix_count[result[i,j]] = get(region_pix_count, result[i, j], 0) + 1
            region_means[result[i,j]] = get(region_means, result[i,j], zero(T)) + (img[i, j] - get(region_means, result[i,j], zero(T)))/region_pix_count[result[i,j]]
        end
    end

    return SegmentedImage(result, labels, region_means, region_pix_count)
end

felzenszwalbnew(field, 0.2)

felzenszwalb(field, 0.2)

felzenszwalbnew(field2, 0.2)

c = TwoColor(1.0, 1.0)
sum(abs2, c)

Base._sum(abs2, field, :)

mapreduce(abs2, Base.add_sum, field2)


iterate(ColorTypes.RGB(1., 1., 1.))

@edit iterate(rand(3))

typeof(fieldnames(typeof(field2[1,1])))

fieldnames()
