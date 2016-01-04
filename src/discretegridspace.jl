# discretegridspace.jl

"""
A DiscreteGridSpace is a discrete basis that can represent a sampled function on a grid.
"""
immutable DiscreteGridSpace{G,ELT,N,T} <: AbstractBasis{N,T}
	grid		::	G

	DiscreteGridSpace(grid::AbstractGrid{N,T}) = new(grid)
end

typealias DiscreteGridSpace1d{G,ELT,T} DiscreteGridSpace{G,ELT,1,T}


DiscreteGridSpace{N,T}(grid::AbstractGrid{N,T}, ELT = T) = DiscreteGridSpace{typeof(grid),ELT,N,T}(grid)

DiscreteGridSpace(tpg::TensorProductGrid, ELT) = TensorProductSet( [ DiscreteGridSpace(grid(tpg, j), ELT) for j in 1:tp_length(tpg)]...)

eltype{G,ELT,N,T}(::Type{DiscreteGridSpace{G,ELT,N,T}}) = ELT

grid(b::DiscreteGridSpace) = b.grid

for op in (:length, :size, :left, :right)
    @eval $op(b::DiscreteGridSpace) = $op(grid(b))
end


rescale(s::DiscreteGridSpace, a, b) = DiscreteGridSpace(rescale(grid(s)))

