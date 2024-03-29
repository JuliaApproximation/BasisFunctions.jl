
"Discrete weights associated with a grid."
abstract type GridWeight{T} <: DiscreteWeight{T} end

grid(μ::GridWeight) = points(μ)

DomainIntegrals._isnormalized(μ, points, weights::NormalizedArray) = true
DomainIntegrals.allequal(A::FillArrays.AbstractFill) = true

# A supermeasure might exist if the grid has a supergrid
supermeasure(μ::GridWeight) = _supermeasure(points(μ), weights(μ))
_supermeasure(points::AbstractGrid, weights::SubArray) = discretemeasure(supergrid(points), parent(weights))
_supermeasure(points::AbstractGrid, weights::Ones) = discretemeasure(supergrid(points))
function _supermeasure(points::AbstractGrid, weights::Fill)
    super = supergrid(points)
    discretemeasure(super, Fill(weights[1], size(super)))
end
_supermeasure(points::AbstractGrid, weights) = discretemeasure(supergrid(points), weights)
_supermeasure(points::ProductGrid, weights::AbstractOuterProductArray) =
    productmeasure(map(_supermeasure, components(points), components(weights))...)

support(μ::GridWeight) = _support(μ, points(μ))
_support(μ::GridWeight, points::MappedGrid) = forward_map(points) * support(supermeasure(μ))
_support(μ::GridWeight, points) = support(points)



"A discrete measure with all weights equal to one."
struct UniformGridWeight{T,G} <: GridWeight{T}
    points  ::  G
end

UniformGridWeight(points) = UniformGridWeight{eltype(points)}(points)
UniformGridWeight{T}(points::G) where {T,G} = UniformGridWeight{T,G}(points)

weights(μ::UniformGridWeight) = uniformweights(points(μ))

"Generate uniform weights for the given grid."
uniformweights(grid::AbstractGrid) = uniformweights(grid, numtype(grid))
uniformweights(grid::AbstractGrid, ::Type{T}) where {T} = Ones{T}(size(grid))
uniformweights(grid::ProductGrid{T,N}, ::Type{V}) where {T,N,V} =
    tensorproduct(ntuple(k->Ones{V}(size(grid,k)) ,Val(N)))
uniformweights(grid::SubGrid, ::Type{T}) where {T} =
    view(uniformweights(supergrid(grid), T), subindices(grid))



"A typed grid weight knows the types of the grid and the points, which may be helpful in dispatch."
abstract type TypedGridWeight{T,G,W} <: GridWeight{T} end

"A generic weighted discrete measure, associated with a set of points and weights."
struct GenericGridWeight{T,G,W} <: TypedGridWeight{T,G,W}
    points  ::  G
    weights ::  W

    function GenericGridWeight{T,G,W}(grid, weights) where {T,G,W}
        # @assert size(grid) == size(weights)
        grid isa Union{SubGrid,ProductSubGrid} || @assert size(grid) == size(weights)
        new(grid, weights)
    end
end

GenericGridWeight(grid::AbstractGrid, weights) =
    GenericGridWeight{eltype(grid)}(grid, weights)
GenericGridWeight{T}(grid::G, weights::W) where {T,G,W} =
    GenericGridWeight{T,G,W}(grid, weights)



const DiscreteProductWeight{T,G,W} = TypedGridWeight{T,G,W} where G<:ProductGrid where W<:AbstractOuterProductArray

DomainIntegrals.productmeasure(measures::DiscreteWeight...) =
    discretemeasure(ProductGrid(map(points, measures)...), tensorproduct(map(weights, measures)...))
components(m::DiscreteProductWeight) = map(discretemeasure, components(points(m)), components(weights(m)))
component(m::DiscreteProductWeight, i) = discretemeasure(component(points(m), i), component(weights(m), i))

function stencilarray(m::DiscreteProductWeight)
    A = Any[]
    push!(A, component(m,1))
    for i = 2:length(components(m))
        push!(A," ⊗ ")
        push!(A, component(m,i))
    end
    A
end



"Construct a discrete measure with the given points (and optionally weights)."
discretemeasure(grid::AbstractGrid) = discretemeasure(grid, uniformweights(grid))

function discretemeasure(grid::AbstractGrid, weights::Ones)
    @assert size(grid) == size(weights)
    uniformdiscretemeasure(grid)
end
uniformdiscretemeasure(grid::AbstractGrid) = UniformGridWeight(grid)

discretemeasure(grid::AbstractGrid, weights) = GenericGridWeight(grid, weights)

points(m::DiracWeight) = ScatteredGrid([m.point])
weights(::DiracWeight{T}) where T = Ones{T}(1)


## Some special cases

DiracComb(grid::AbstractEquispacedGrid) = UniformGridWeight(grid)
NormalizedDiracComb(grid::AbstractEquispacedGrid) =
    discretemeasure(grid, NormalizedArray{prectype(grid)}(size(grid)))

# mappedmeasure(map, measure::DiscreteWeight) =
#     discretemeasure(MappedGrid(weights(measure), points(measure), map))
forward_map(μ::GridWeight) = forward_map(points(μ))
apply_map(measure::DiscreteWeight, map) =
    discretemeasure(map_grid(map, points(measure)), weights(measure))

ismappedmeasure(μ::GridWeight) = points(μ) isa MappedGrid

"A discrete measure that represents a quadrature rule."
struct QuadratureMeasure{T,P,W} <: DiscreteWeight{T}
    points  ::  P
    weights ::  W
end

QuadratureMeasure(points::AbstractArray{T}, weights) where {T} =
    QuadratureMeasure{T}(points, weights)
QuadratureMeasure{T}(points::P, weights::W) where {T,P,W} =
    QuadratureMeasure{T,P,W}(points, weights)
