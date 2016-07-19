# scattered_grid.jl

"A grid corresponding to an unstructured collection of points."
immutable ScatteredGrid{N,T} <: AbstractGrid{N,T}
    points     ::  Array{Vec{N,T},1}
end

length(g::ScatteredGrid) = length(g.points)

size(g::ScatteredGrid) = (length(g),)

getindex(g::ScatteredGrid, idx::Int) = g.points[idx]
