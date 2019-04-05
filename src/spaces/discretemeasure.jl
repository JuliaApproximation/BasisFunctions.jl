

struct GenericDiscreteMeasure{T,GRID<:AbstractGrid,W} <: DiscreteMeasure{T}
    grid   ::  GRID
    weights   ::  W
    function GenericDiscreteMeasure(grid::AbstractGrid, weights)
        grid isa Union{AbstractSubGrid,TensorSubGrid} || @assert size(grid) == size(weights)
        new{eltype(grid),typeof(grid),typeof(weights)}(grid, weights)
    end
end
genericweights(grid::AbstractGrid) = Ones{subeltype(eltype(grid))}(size(grid)...)
genericweights(grid::ProductGrid) = tensorproduct([Ones{subeltype(eltype(grid))}(l) for l in size(grid)]...)

DiscreteMeasure(grid::AbstractGrid, weights=genericweights(grid)) =
    GenericDiscreteMeasure(grid, weights)

name(m::GenericDiscreteMeasure) = "Generic discrete measure on grid $(typeof(grid(m)))"
strings(m::GenericDiscreteMeasure) = (name(m), (string(grid(m)),), (string(string(weights(m))),))
isprobabilitymeasure(m::GenericDiscreteMeasure) = weights(m) isa ProbabilityArray


"A Dirac function at a point `x`."
struct DiracMeasure{T} <: DiscreteMeasure{T}
    x   ::  T
end

support(m::DiracMeasure) = Point(m.x)
name(m::DiracMeasure) = "Dirac measure at x = $(m.x)"
point(m::DiracMeasure) = m.x
grid(m::DiracMeasure) = ScatteredGrid(m.x)
checkbounds(::DiracMeasure, i) = (convert(Int,i)==1) || throw(BoundsError())
isprobabilitymeasure(::DiracMeasure) = true
weights(::DiracMeasure{T}) where T = Ones{T}(1)


const DiracCombMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W <: Ones
DiracCombMeasure(eg::AbstractEquispacedGrid) = DiscreteMeasure(eg)
name(m::DiracCombMeasure) = "Dirac comb measure with sampling weights (GenericWeightMeasure)"


const WeightedDiracCombMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid
WeightedDiracCombMeasure(eg::AbstractEquispacedGrid, weights) = DiscreteMeasure(eg, weights)
name(m::WeightedDiracCombMeasure) = "Dirac comb measure with generic weight (GenericWeightMeasure)"


const DiracCombProbabilityMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W<:ProbabilityArray
DiracCombProbabilityMeasure(eg::AbstractEquispacedGrid) = DiscreteMeasure(eg, ProbabilityArray{eltype(eg)}(size(eg)))
@inline isprobabilitymeasure(m::DiracCombProbabilityMeasure) = true
name(m::DiracCombProbabilityMeasure) = "Dirac comb measure with probability weights (GenericWeightMeasure)"

const UniformDiracCombMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W<:FillArrays.AbstractFill


######################################################
# Generating new measures from existing measures
######################################################
# The weighs saved are the weights of the supermeasure. (there may be a better solution cf SubMeasure)
const DiscreteSubMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractSubGrid

DiscreteMeasure(grid::Union{AbstractSubGrid,TensorSubGrid}) = DiscreteSubMeasure(grid, genericweights(supergrid(grid)))
DiscreteSubMeasure(grid::Union{AbstractSubGrid,TensorSubGrid}) = DiscreteSubMeasure(grid, genericweights(supergrid(grid)))

DiscreteSubMeasure(grid, weights) = (@assert size(weights)==size(supergrid(grid)); DiscreteMeasure(grid, weights))

weights(measure::DiscreteSubMeasure) = subweights(measure, subindices(grid(measure)), measure.weights)
subweights(_, subindices, w) = w[subindices]
unsafe_discrete_weight(m::DiscreteSubMeasure, i) where {T} = Base.unsafe_getindex(m.weights, subindices(m.grid)[i])
isprobabilitymeasure(::DiscreteSubMeasure) = false

name(m::DiscreteSubMeasure) = "Restriction of a "*name(supermeasure(m))

supermeasure(measure::DiscreteSubMeasure) =
    supermeasure(measure, supergrid(grid(measure)), measure.weights)

function supermeasure(::DiscreteSubMeasure, grid::AbstractGrid, weights)
    if length(weights) != length(grid)
        error("weights of supermeasure could not be recovered")
    else
        DiscreteMeasure(grid, weights)
    end
end

# # TODO move subgrid to BasisFunctions or SubMeasure to FrameFun
# restrict(measure::DiscreteMeasure, domain::Domain) = DiscreteSubMeasure(subgrid(grid(measure), domain), weights(measure))


const DiscreteMappedMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:MappedGrid
MappedMeasure(map, measure::DiscreteMeasure) =
DiscreteMeasure(MappedGrid(grid(measure), map), weights(measure))

name(m::DiscreteMappedMeasure) = "Mapping of a "*name(supermeasure(m))
mapping(measure::DiscreteMappedMeasure) = mapping(grid(measure))
supermeasure(m::DiscreteMappedMeasure) = DiscreteMeasure(supergrid(grid(m)), m.weights)
apply_map(measure::DiscreteMeasure, map) = DiscreteMeasure(apply_map(grid(measure)), weights(measure))
apply_map(measure::DiscreteMappedMeasure, map) = MappedMeasure(map*mapping(measure), supermeasure(measure))
support(m::DiscreteMappedMeasure) = mapping(m) * support(supermeasure(m))


const DiscreteProductMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:ProductGrid where  W<:AbstractOuterProductArray

const DiscreteTensorSubMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:TensorSubGrid where W<:AbstractOuterProductArray
supermeasure(m::DiscreteTensorSubMeasure) = ProductMeasure(map(supermeasure, elements(m))...)

name(m::DiscreteProductMeasure) = "Discrete Product Measure"
ProductMeasure(measures::DiscreteMeasure...) =
    DiscreteMeasure(ProductGrid(map(grid, measures)...), tensorproduct(map(weights, measures)...))


iscomposite(m::DiscreteProductMeasure) = true
elements(m::DiscreteProductMeasure) = map(DiscreteMeasure, elements(grid(m)), elements(weights(m)))
element(m::DiscreteProductMeasure, i) = DiscreteMeasure(element(grid(m), i), element(weights(m), i))
function stencilarray(m::DiscreteProductMeasure)
    A = Any[]
    push!(A, element(m,1))
    for i = 2:length(elements(m))
        push!(A," ⊗ ")
        push!(A, element(m,i))
    end
    A
end
