# discretegridspace.jl

"""
A DiscreteGridSpace is a discrete basis that can represent a sampled function on a grid.
"""
immutable DiscreteGridSpace{G,N,T} <: AbstractBasis{N,T}
	grid		::	G

	DiscreteGridSpace(grid::AbstractGrid{N}) = new(grid)
end

typealias DiscreteGridSpace1d{G,ELT} DiscreteGridSpace{G,ELT,1}


DiscreteGridSpace{N,T}(grid::AbstractGrid{N,T}, ELT = T) = DiscreteGridSpace{typeof(grid),N,ELT}(grid)

DiscreteGridSpace(tpg::TensorProductGrid, ELT = numtype(tpg)) =
    TensorProductSet( [ DiscreteGridSpace(grid(tpg, j), ELT) for j in 1:tp_length(tpg)]...)

promote_eltype{G,N,T,S}(s::DiscreteGridSpace{G,N,T}, ::Type{S}) = DiscreteGridSpace(s.grid, promote_type(T,S))

resize{G,N,T}(s::DiscreteGridSpace{G,N,T}, n) = DiscreteGridSpace(resize(grid(s), n), T)

grid(s::DiscreteGridSpace) = s.grid

for op in (:length, :size, :left, :right)
    @eval $op(b::DiscreteGridSpace) = $op(grid(b))
end


rescale(s::DiscreteGridSpace, a, b) = DiscreteGridSpace(rescale(grid(s)))

