# discretegridspace.jl

"""
A DiscreteGridSpace is a discrete basis that can represent a sampled function on a grid.
"""
struct DiscreteGridSpace{G,ELT,T} <: FunctionSet{T}
	# TODO: remove the G parameter and leave grid untyped.
	# Currently, this clashes with dispatch on Extension in FrameFun's subgrid.jl
	grid		::	G

	DiscreteGridSpace{G,ELT,T}(grid::AbstractGrid) where {G,ELT,T} = new(grid)
end

const DiscreteGridSpace1d{G,T <: Number} = DiscreteGridSpace{G,T}

name(s::DiscreteGridSpace) = "A discrete grid space"

DiscreteGridSpace(grid::AbstractGrid{SVector{N,T}}, ELT = T) where {N,T} = DiscreteGridSpace{typeof(grid),ELT,SVector{N,T}}(grid)

DiscreteGridSpace(grid::AbstractGrid{T}, ELT = T) where {T <: Number} = DiscreteGridSpace{typeof(grid),ELT,T}(grid)


# Both the rangetype and coefficient type of a discrete grid space are ELT
rangetype(::Type{DiscreteGridSpace{G,ELT,T}}) where {G,ELT,T} = ELT
coefficient_type(::Type{DiscreteGridSpace{G,ELT,T}}) where {G,ELT,T} = ELT

set_promote_domaintype(s::DiscreteGridSpace{G,ELT,T}, ::Type{S}) where {G,ELT,T,S} = DiscreteGridSpace(s.grid, S)

# resize{G,N,T}(s::DiscreteGridSpace{G,N,T}, n) = DiscreteGridSpace(resize(grid(s), n), T)

similar(dgs::DiscreteGridSpace, grid::AbstractGrid) = DiscreteGridSpace(grid, eltype(dgs))

grid(s::DiscreteGridSpace) = s.grid

for op in (:length, :size, :left, :right)
    @eval $op(b::DiscreteGridSpace) = $op(grid(b))
end

# Convenience function: add grid as extra parameter to has_transform
has_transform(s::FunctionSet, dgs::DiscreteGridSpace) =
	has_grid_transform(s, dgs, grid(dgs))
# and provide a default
has_grid_transform(s::FunctionSet, dgs, grid) = false


is_discrete(s::DiscreteGridSpace) = true

# Delegate a map to the underlying grid, but retain the element type of the space
apply_map(s::DiscreteGridSpace, map) = DiscreteGridSpace(apply_map(grid(s), map), eltype(s))

function tensorproduct(s::DiscreteGridSpace...)
	ELT = promote_type(map(coefficient_type, s)...)
	DiscreteGridSpace(cartesianproduct(map(grid, s)...), ELT)
end

# Implement composite interface for spaces with a productgrid
element(s::DiscreteGridSpace, i) = _element(s, i, grid(s))
_element(s::DiscreteGridSpace, i, grid) = DiscreteGridSpace(element(grid, i), coefficient_type(s))

nb_elements(s::DiscreteGridSpace) = nb_elements(grid(s))

elements(s::DiscreteGridSpace) = map(t->DiscreteGridSpace(t, coefficient_type(s)), elements(grid(s)))

# Make a DiscreteGridSpace with the same eltype as the given function set
gridspace(s::FunctionSet, g = grid(s)) = DiscreteGridSpace(g, coefficient_type(s))

sample(s::DiscreteGridSpace, f) = sample(grid(s), f, eltype(s))
