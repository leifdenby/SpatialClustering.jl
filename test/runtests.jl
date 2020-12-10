using Test
using SpatialClustering

Nemb = 10
dtype = Float32
nx, ny = 20, 15

@test EmbColor{Float32,3}(rand(Float32,3)) isa EmbColor
@test EmbColor(rand(dtype,3)) isa EmbColor
@test eltype(EmbColor(rand(dtype,(nx, ny, Nemb)))) == EmbColor{dtype,Nemb}

field = EmbColor(rand(Float32, (nx, ny, Nemb)))
@test field isa AbstractMatrix{EmbColor{Float32,Nemb}}

import ImageSegmentation
seg = ImageSegmentation.felzenszwalb(field, 0.2)

@test size(seg.image_indexmap) == (nx, ny)
