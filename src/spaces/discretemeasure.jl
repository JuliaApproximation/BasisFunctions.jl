

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

discretemeasure(grid::AbstractGrid, weights=genericweights(grid)) =
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
DiracCombMeasure(eg::AbstractEquispacedGrid) = discretemeasure(eg)
name(m::DiracCombMeasure) = "Dirac comb measure with sampling weights (GenericWeightMeasure)"


const WeightedDiracCombMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid
WeightedDiracCombMeasure(eg::AbstractEquispacedGrid, weights) = discretemeasure(eg, weights)
name(m::WeightedDiracCombMeasure) = "Dirac comb measure with generic weight (GenericWeightMeasure)"


const DiracCombProbabilityMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W<:ProbabilityArray
DiracCombProbabilityMeasure(eg::AbstractEquispacedGrid) = discretemeasure(eg, ProbabilityArray{eltype(eg)}(size(eg)))
@inline isprobabilitymeasure(m::DiracCombProbabilityMeasure) = true
name(m::DiracCombProbabilityMeasure) = "Dirac comb measure with probability weights (GenericWeightMeasure)"

const UniformDiracCombMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W<:FillArrays.AbstractFill


######################################################
# Generating new measures from existing measures
######################################################

struct DiscreteSubMeasure{T,M<:DiscreteMeasure{T},I} <: DiscreteMeasure{T}
    supermeasure   :: M
    indices        :: I
    DiscreteSubMeasure(measure::DiscreteMeasure{T}, indices) where T = new{T,typeof(measure),typeof(indices)}(measure, indices)
end
subindices(measure::DiscreteSubMeasure) = measure.indices
supermeasure(measure::DiscreteSubMeasure) = measure.supermeasure
supermeasure(measure::GenericDiscreteMeasure{T,<:AbstractSubGrid,W}) where {T,W} =
    GenericDiscreteMeasure(supergrid(grid(measure)), weights(measure))
discretemeasure(grid::Union{AbstractSubGrid,TensorSubGrid}) = DiscreteSubMeasure(GenericDiscreteMeasure(supergrid(grid), genericweights(supergrid(grid))),subindices(grid))
DiscreteSubMeasure(grid::Union{AbstractSubGrid,TensorSubGrid}) = DiscreteSubMeasure(GenericDiscreteMeasure(grid, genericweights(supergrid(grid))),subindices(grid))

weights(measure::DiscreteSubMeasure) = subweights(measure, subindices(measure), weights(supermeasure(measure)))
subweights(_, subindices, w) = w[subindices]
grid(measure::DiscreteSubMeasure) = subgrid(measure, subindices(measure), grid(supermeasure(measure)))
Grids.subgrid(_, subindices, g) = g[subindices]
unsafe_discrete_weight(m::DiscreteSubMeasure, i) where {T} = Base.unsafe_getindex(weights(supermeasure(measure)), subindices(measure)[i])
isprobabilitymeasure(::DiscreteSubMeasure) = false

name(m::DiscreteSubMeasure) = "Restriction of a "*name(supermeasure(m))


# # TODO move subgrid to BasisFunctions or SubMeasure to FrameFun
# restrict(measure::DiscreteMeasure, domain::Domain) = DiscreteSubMeasure(subgrid(grid(measure), domain), weights(measure))


const DiscreteMappedMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:MappedGrid
MappedMeasure(map, measure::DiscreteMeasure) =
discretemeasure(MappedGrid(grid(measure), map), weights(measure))

name(m::DiscreteMappedMeasure) = "Mapping of a "*name(supermeasure(m))
mapping(measure::DiscreteMappedMeasure) = mapping(grid(measure))
supermeasure(m::DiscreteMappedMeasure) = discretemeasure(supergrid(grid(m)), m.weights)
apply_map(measure::DiscreteMeasure, map) = discretemeasure(apply_map(grid(measure)), weights(measure))
apply_map(measure::DiscreteMappedMeasure, map) = MappedMeasure(map*mapping(measure), supermeasure(measure))
support(m::DiscreteMappedMeasure) = mapping(m) * support(supermeasure(m))


const DiscreteProductMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:ProductGrid where  W<:AbstractOuterProductArray

const DiscreteTensorSubMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:TensorSubGrid where W<:AbstractOuterProductArray
supermeasure(m::DiscreteTensorSubMeasure) = ProductMeasure(map(supermeasure, elements(m))...)

name(m::DiscreteProductMeasure) = "Discrete Product Measure"
ProductMeasure(measures::DiscreteMeasure...) =
    discretemeasure(ProductGrid(map(grid, measures)...), tensorproduct(map(weights, measures)...))


iscomposite(m::DiscreteProductMeasure) = true
elements(m::DiscreteProductMeasure) = map(discretemeasure, elements(grid(m)), elements(weights(m)))
element(m::DiscreteProductMeasure, i) = discretemeasure(element(grid(m), i), element(weights(m), i))
function stencilarray(m::DiscreteProductMeasure)
    A = Any[]
    push!(A, element(m,1))
    for i = 2:length(elements(m))
        push!(A," ⊗ ")
        push!(A, element(m,i))
    end
    A
end



abstract type VectorAlias{T} <: AbstractVector{T} end
getindex(vector::VectorAlias, i::Int) = getindex(vector.vector, i)
unsafe_getindex(vector::VectorAlias, i::Int) = unsafe_getindex(vector.vector, i)
size(vector::VectorAlias) = size(vector.vector)

abstract type FunctionVector{T} <: AbstractVector{T} end
length(vector::FunctionVector) = vector.n
size(vector::FunctionVector) = (length(vector),)

function getindex(vector::FunctionVector, i::Int)
    @boundscheck (1 <= i <= length(vector)) || throw(BoundsError())
    @inbounds unsafe_getindex(vector, i)
end


struct ChebyshevUWeights{T} <: FunctionVector{T}
    n   :: Int
end

unsafe_getindex(weights::ChebyshevUWeights{T}, i::Int) where {T} =
    convert(T,π)/(weights.n + 1) * sin(convert(T,weights.n + 1 -i) / (weights.n + 1) * convert(T,π))^2



struct ChebyshevTGaussMeasure{T} <:DiscreteMeasure{T}
    grid    :: ChebyshevTNodes{T}
    weights :: ChebyshevTWeights{T}
    ChebyshevTGaussMeasure{T}(n::Int) where T =
        new{T}(ChebyshevTNodes{T}(n), ChebyshevTWeights{T}(n))
    ChebyshevTGaussMeasure(n::Int) = ChebyshevTGaussMeasure{Float64}(n)
end

struct ChebyshevUGaussMeasure{T} <:DiscreteMeasure{T}
    grid   :: ChebyshevUNodes{T}
    weights:: ChebyshevUWeights{T}
    ChebyshevUGaussMeasure{T}(n::Int) where T =
        new{T}(ChebyshevUNodes{T}(n), ChebyshevUWeights{T}(n))
    ChebyshevUGaussMeasure(n::Int) = ChebyshevUGaussMeasure{Float64}(n)
end

struct LegendreWeights{T} <: VectorAlias{T}
    vector  ::  Vector{T}
end
struct LegendreGaussMeasure{T} <: DiscreteMeasure{T}
    grid    :: LegendreNodes{T}
    weights :: LegendreWeights{T}

    function LegendreGaussMeasure{Float64}(n::Int)
        x,w = gausslegendre(n)
        new{Float64}(LegendreNodes(x), LegendreWeights(w))
    end
    function LegendreGaussMeasure{T}(n::Int) where T
        x,w = legendre(T, n)
        new{T}(LegendreNodes(x), LegendreWeights(w))
    end
    LegendreGaussMeasure(n::Int) = LegendreGaussMeasure{Float64}(n)
end

struct LaguerreWeights{T} <: VectorAlias{T}
    α       ::  T
    vector  ::  Vector{T}
end
struct LaguerreGaussMeasure{T} <: DiscreteMeasure{T}
    α       :: T
    grid    :: LaguerreNodes{T}
    weights :: LaguerreWeights{T}

    function LaguerreGaussMeasure{Float64}(n::Int, α::Float64)
        x,w = gausslaguerre(n, α)
        new{Float64}(α, LaguerreNodes(α, x), LaguerreWeights(α, w))
    end
    function LaguerreGaussMeasure{T}(n::Int, α::T) where T
        x,w = laguerre(n, α)
        new{T}(α, LaguerreNodes(α, x), LaguerreWeights(α, w))
    end
    LaguerreGaussMeasure(n::Int, α::T) where T = LaguerreGaussMeasure{T}(n, α)
end

struct HermiteWeights{T} <: VectorAlias{T}
    vector  ::  Vector{T}
end
struct HermiteGaussMeasure{T} <: DiscreteMeasure{T}
    grid    :: HermiteNodes{T}
    weights :: HermiteWeights{T}

    function HermiteGaussMeasure{Float64}(n::Int)
        x,w = gausshermite(n)
        new{Float64}(HermiteNodes(x), HermiteWeights(w))
    end
    function HermiteGaussMeasure{T}(n::Int) where T
        x,w = hermite(T, n)
        new{T}(HermiteNodes(x), HermiteWeights(w))
    end
    HermiteGaussMeasure(n::Int) = HermiteGaussMeasure{Float64}(n)
end

struct JacobiWeights{T} <: VectorAlias{T}
    α       ::  T
    β       ::  T
    vector  ::  Vector{T}
end
struct JacobiGaussMeasure{T} <: DiscreteMeasure{T}
    α       :: T
    β       :: T
    grid    :: JacobiNodes{T}
    weights :: JacobiWeights{T}

    function JacobiGaussMeasure{Float64}(n::Int, α::Float64, β::Float64)
        x,w = gaussjacobi(n, α, β)
        new{Float64}(α, β, JacobiNodes(α, β, x), JacobiWeights(α, β, w))
    end
    function JacobiGaussMeasure{T}(n::Int, α::T, β::T) where T
        x,w = jacobi(n, α, β)
        new{T}(α, β, JacobiNodes(α, β, x), JacobiWeights(α, β, w))
    end
    JacobiGaussMeasure(n::Int, α::T, β::T) where T = JacobiGaussMeasure{T}(n, α, β)
end
