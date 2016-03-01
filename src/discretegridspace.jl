# discretegridspace.jl

"""
A DiscreteGridSpace is a discrete basis that can represent a sampled function on a grid.
"""
immutable DiscreteGridSpace{G,ELT,N} <: AbstractBasis{N,ELT}
	grid		::	G

	DiscreteGridSpace(grid::AbstractGrid{N}) = new(grid)
end

typealias DiscreteGridSpace1d{G,ELT} DiscreteGridSpace{G,ELT,1}


DiscreteGridSpace{N,T}(grid::AbstractGrid{N,T}, ELT = T) = DiscreteGridSpace{typeof(grid),ELT,N}(grid)

DiscreteGridSpace(tpg::TensorProductGrid, ELT = eltype(tpg)) =
    TensorProductSet( [ DiscreteGridSpace(grid(tpg, j), ELT) for j in 1:tp_length(tpg)]...)

eltype{G,ELT,N}(::Type{DiscreteGridSpace{G,ELT,N}}) = ELT

grid(b::DiscreteGridSpace) = b.grid

for op in (:length, :size, :left, :right)
    @eval $op(b::DiscreteGridSpace) = $op(grid(b))
end


rescale(s::DiscreteGridSpace, a, b) = DiscreteGridSpace(rescale(grid(s)))

similar{G,ELT,N}(s::DiscreteGridSpace{G,ELT,N}, ELTnew, n) = DiscreteGridSpace{G,ELTnew,N}(grid(s))
