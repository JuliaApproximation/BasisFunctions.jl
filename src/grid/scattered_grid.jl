# scattered_grid.jl

"A grid corresponding to an unstructured collection of points."
struct ScatteredGrid{T} <: AbstractGrid{T}
    points     ::  Vector{T}
end

length(g::ScatteredGrid) = length(g.points)

size(g::ScatteredGrid) = (length(g),)

unsafe_getindex(g::ScatteredGrid, idx) = g.points[idx]
