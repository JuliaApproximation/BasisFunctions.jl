
"A grid corresponding to an unstructured collection of points."
struct ScatteredGrid{T} <: AbstractGrid{T,1}
    points     ::  Vector{T}
end

name(grid::ScatteredGrid) = "Scattered grid"

size(g::ScatteredGrid) = (length(g.points),)

unsafe_getindex(g::ScatteredGrid, idx) = g.points[idx]

setindex!(g::ScatteredGrid, val, i::Int) = g.points[i] = val
