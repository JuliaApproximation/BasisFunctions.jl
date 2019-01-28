
"Grids are arrays of points."
abstract type AbstractGrid{T,N} <: AbstractArray{T,N}
end

const AbstractGrid1d{T <: Number,N} = AbstractGrid{T,N}
# Todo: remove this one again, it is not sufficiently generic
const AbstractGrid2d{T <: Number,N} = AbstractGrid{SVector{2,T},N}

# The dimension of a grid is the dimension of its elements
dimension(grid::AbstractGrid) = dimension(eltype(grid))

IndexStyle(grid::AbstractGrid{1,T}) where {T} = IndexLinear()
IndexStyle(grid::AbstractGrid{N,T}) where {N,T} = IndexCartesian()

@propagate_inbounds function getindex(grid::AbstractGrid{T,1}, i::Int) where {T}
    checkbounds(grid, i)
    unsafe_getindex(grid, i)
end

resize(grid::AbstractGrid{T}, dims...) where {T} = similargrid(grid, T, dims...)

hasextension(grid::AbstractGrid) = false

iscomposite(grid::AbstractGrid) = false
