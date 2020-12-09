import NCDatasets
import Plots
import ImageSegmentation
import ColorTypes

include("embedding_2d.jl")

ds = NCDatasets.Dataset("embeddings_scene_pca.nc")
@show ds

field = ds["emb"][:]
field = Float32.(replace!(field, missing => 0.0))

x = ds["x"]
y = ds["y"]

Plots.heatmap(x, y, field[:,:,1]; aspect=1)

color_field = TwoColor.(field[:,:,1], field[:,:,2])
color_field_1 = ColorTypes.RGB.(field[:,:,1], field[:,:,2], zeros(size(field)[1:2]))

ImageSegmentation.felzenszwalb(color_field, 0.2)
ImageSegmentation.felzenszwalb(color_field_1, 0.2)
