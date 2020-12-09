import NCDatasets
import Plots
import ShiftedArrays
import Statistics
import ImageSegmentation
import Images
import ColorTypes
import Random

ds = NCDatasets.Dataset("embeddings_scene_pca.nc")
@show ds

field = ds["emb"][:]
field = replace!(field, missing => 0.0)

x = ds["x"]
y = ds["y"]


function mean_variance(v)
# calculate the variance acros x and y for each embedding dimension separately
# (dim 1), and then calculate the mean
    Statistics.mean(Statistics.var(field; dims=(2,3))[:])
end

#mean_variance(field)

Plots.heatmap(x, y, field[:,:,1]; aspect=1)

ones(size(field))

mask = rand(Float32, size(field)[2:3]) .> 0.5

field[..,mask]

size(field)[1:2]

Statistics.mean()

struct EmbColor <:

img = field[:,:,:]
img = float.(img)

struct Emb5Color{T<:AbstractFloat} <: ColorTypes.Color{T,5}
    a::T
    b::T
    c::T
    d::T
    e::T
end

Base.zero(x::Emb5Color) = Emb5Color(zero(size(x)))

Emb5Color.(img[1,:,:], img[2,:,:], img[3,:,:], img[4,:,:], img[5,:,:])

scale(x) = (x .- minimum(x)) ./(maximum(x) - minimum(x))
scale(img[:,:,1])

img2 = Images.RGB.(scale(img[:,:,1]), scale(img[:,:,2]), scale(img[:,:,3]), )

segments = ImageSegmentation.felzenszwalb(img2, 0.8)

@show img

function get_random_color(seed)
    Random.seed!(seed)
    rand(Images.RGB{Images.N0f8})
end

segments_map = map(i->get_random_color(i), ImageSegmentation.labels_map(segments))
segments_map = ImageSegmentation.labels_map(segments)

Plots.heatmap(field[:,:,1]; subplot=1)
Plots.heatmap!(segments_map; alpha=0.3)

Images.get_ran

ds_out = NCDatasets.Dataset("embeddings_scene_pca_segmented.nc", "c")
NCDatasets.defDim(ds_out, "x", length(x))
NCDatasets.defDim(ds_out, "y", length(y))
v = NCDatasets.defVar(ds_out, "class", Float32, ("y", "x"))
v[:] = segments_map
close(ds_out)

Int.(segments_map)
Images.rawview(segments_map)

ImageSegmentation.labels_map(segments)

img_rgb = Images.RGB.(field[:,:,1], field[:,:,2], field[:,:,3])

Images.accum


struct TwoPartColor{T<:AbstractFloat} <: ColorTypes.Color{T,2}
    a::T
    b::T
    #TwoPartColor{T}(a::T, b::T) where T = new{T}(a, b)
end

Base.zero(::Type{C}) where {C<:TwoPartColor} = TwoPartColor(0, 0)

TwoPartColor(0.0, 0.0)

@edit ColorTypes.HSV(0.0, 0.0, 0.0)
TP(0.0, 0.0, 0.0)

delete!(TwoPartColor)


img_twopart = TwoPartColor{Float32}.(field[:,:,1], field[:,:,2])

Convml.TwoPartColor
ImageSegmentation.felzenszwalb(img_twopart, 0.8)

TwoPartColor(x::AbstractArray{T,1}) where T = TwoPartColor{T}.(x[1,:,:], x[2,:,:])

float.(field)
TwoPartColor(float.(field))

v = TwoPartColor{Float32}.(field[:,:,1], field[:,:,2])
size(v)
size(TwoPartColor{Float32}.([field[:,:,n] for n in 1:2]...))

unpack

@edit zero(Float32)

# Base.zero(::Type{T}) where {T<:Number} = convert(T,0)

@edit ImageSegmentation.meantype(typeof(img_rgb))
meantype(::Type{T}) where T = typeof(zero(Images.accum(T))/2)

@edit zero(Images.accum(typeof(img_rgb[1,1])))
zero(Images.accum(typeof(img_twopart[1,1])))

### Using 10 emb dimensions

struct Emb10Color{T<:AbstractFloat} <: ColorTypes.Color{T,10}
    # need an eval in here to add  struct elements
    x1::T
    x2::T
    x3::T
    x4::T
    x5::T
    x6::T
    x7::T
    x8::T
    x9::T
    x10::T
    #for n in 1:100
    #    @eval $(Symbol(@sprintf("x_%d", n)))::T
    #end
end
Base.zero(::Type{C}) where {C<:Emb10Color} = Emb10Color(zeros(10)...)
@edit Base.show(IOBuffer(), ColorTypes.Color)

Base.delete_method(@which Emb10Color{Float64}(rand(Float32, 10)...) / 2.0)
Base.delete_method(@which Emb10Color{Float32}(rand(Float32, 10)...) / 2.0)
function (/)(c::Emb10Color{T}, x::Number) where {T<:AbstractFloat}
    vals = [getfield(c, name)*inv(T(x)) for name in fieldnames(typeof(c))]
    Emb10Color{T}(vals...)
end

function (/)(c::Emb10Color{Float64}, x::Int64)
    vals = [getfield(c, name)*inv(Float64(x)) for name in fieldnames(typeof(c))]
    Emb10Color{Float64}(vals...)
end

/(c::Emb10Color{Float64}, x::Int64) = missing

size(Emb10Color{Float64}.([field[:,:,n] for n in 1:10]...)./Int64(1.0))

ImageSegmentation.felzenszwalb(Emb10Color{Float32}.([field[:,:,n] for n in 1:10]...), 0.8)

using Printf


zero(Images.accum(Emb10Color{Float64}))
