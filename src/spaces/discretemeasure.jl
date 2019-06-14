

struct GenericDiscreteMeasure{T,GRID<:AbstractGrid,W} <: DiscreteMeasure{T}
    grid   ::  GRID
    weights   ::  W
    function GenericDiscreteMeasure(grid::AbstractGrid, weights)
        grid isa Union{AbstractSubGrid,TensorSubGrid} || @assert size(grid) == size(weights)
        new{eltype(grid),typeof(grid),typeof(weights)}(grid, weights)
    end
end
≈(m1::DiscreteMeasure, m2::DiscreteMeasure) = ≈(map(grid,(m1,m2))...) &&  ≈(map(weights,(m1,m2))...)
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
grid(m::DiracMeasure) = ScatteredGrid([m.x])
isprobabilitymeasure(::DiracMeasure) = true
weights(::DiracMeasure{T}) where T = Ones{T}(1)


const DiracCombMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W <: Ones
DiracCombMeasure(eg::AbstractEquispacedGrid) = discretemeasure(eg)
name(m::DiracCombMeasure) = "Dirac comb measure with sampling weights"


const WeightedDiracCombMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid
WeightedDiracCombMeasure(eg::AbstractEquispacedGrid, weights) = discretemeasure(eg, weights)
name(m::WeightedDiracCombMeasure) = "Dirac comb measure with generic weight"


const DiracCombProbabilityMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W<:ProbabilityArray
DiracCombProbabilityMeasure(eg::AbstractEquispacedGrid) = discretemeasure(eg, ProbabilityArray{eltype(eg)}(size(eg)))
@inline isprobabilitymeasure(m::DiracCombProbabilityMeasure) = true
name(m::DiracCombProbabilityMeasure) = "Dirac comb measure with probability weights"

const UniformDiracCombMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W<:FillArrays.AbstractFill


######################################################
# Generating new measures from existing measures
######################################################

struct DiscreteSubMeasure{T,M<:DiscreteMeasure{T},G<:AbstractGrid} <: DiscreteMeasure{T}
    supermeasure   :: M
    subgrid        :: G
    DiscreteSubMeasure(measure::DiscreteMeasure{T}, grid::AbstractGrid) where T = new{T,typeof(measure),typeof(grid)}(measure, grid)
end
subindices(measure::DiscreteSubMeasure) = subindices(measure.subgrid)
supermeasure(measure::DiscreteSubMeasure) = measure.supermeasure
discretemeasure(grid::Union{AbstractSubGrid,TensorSubGrid}) = DiscreteSubMeasure(discretemeasure(supergrid(grid)), grid)
_discretesubmeasure(grid::Union{AbstractSubGrid,TensorSubGrid},weights) = DiscreteSubMeasure(discretemeasure(supergrid(grid),weights), grid)

weights(measure::DiscreteSubMeasure) = subweights(measure, subindices(measure), weights(supermeasure(measure)))
subweights(_, subindices, w) = w[subindices]

grid(measure::DiscreteSubMeasure) = measure.subgrid

isprobabilitymeasure(::DiscreteSubMeasure) = false

name(m::DiscreteSubMeasure) = "Restriction of a "*name(supermeasure(m))


# # TODO move subgrid to BasisFunctions or SubMeasure to FrameFun
restrict(measure::DiscreteMeasure, domain::Domain) = DiscreteSubMeasure(measure, subgrid(grid(measure), domain))


const DiscreteMappedMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:MappedGrid
mappedmeasure(map, measure::DiscreteMeasure) =
discretemeasure(MappedGrid(grid(measure), map), weights(measure))

name(m::DiscreteMappedMeasure) = "Mapping of a "*name(supermeasure(m))
mapping(measure::DiscreteMappedMeasure) = mapping(grid(measure))
supermeasure(m::DiscreteMappedMeasure) = discretemeasure(supergrid(grid(m)), m.weights)
apply_map(measure::DiscreteMeasure, map) = discretemeasure(apply_map(grid(measure),map), weights(measure))
support(m::DiscreteMappedMeasure) = mapping(m) * support(supermeasure(m))



const DiscreteProductMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:ProductGrid where  W<:AbstractOuterProductArray

const DiscreteTensorSubMeasure{T,G,W} = DiscreteSubMeasure{T,M,G} where {T,M<:BasisFunctions.DiscreteProductMeasure,G<:TensorSubGrid}

name(m::DiscreteTensorSubMeasure) = "Tensor of submeasures (supermeasure:"*name(supermeasure(m))
elements(m::DiscreteTensorSubMeasure) = map(BasisFunctions._discretesubmeasure,elements(grid(m)),elements(weights(supermeasure(m))))
element(m::DiscreteTensorSubMeasure, i) = BasisFunctions._discretesubmeasure(element(grid(m),i),element(weights(supermeasure(m)),i))
iscomposite(m::DiscreteTensorSubMeasure) = true


name(m::DiscreteProductMeasure) = "Discrete Product Measure"
productmeasure(measures::DiscreteMeasure...) =
    discretemeasure(ProductGrid(map(grid, measures)...), tensorproduct(map(weights, measures)...))


iscomposite(m::DiscreteProductMeasure) = true
elements(m::DiscreteProductMeasure) = map(discretemeasure, elements(grid(m)), elements(weights(m)))
element(m::DiscreteProductMeasure, i) = discretemeasure(element(grid(m), i), element(weights(m), i))
function stencilarray(m::Union{DiscreteProductMeasure,DiscreteTensorSubMeasure})
    A = Any[]
    push!(A, element(m,1))
    for i = 2:length(elements(m))
        push!(A," ⊗ ")
        push!(A, element(m,i))
    end
    A
end


struct ChebyshevTGaussMeasure{T} <:DiscreteMeasure{T}
    grid    :: GridArrays.ChebyshevTNodes{T}
    weights :: GridArrays.ChebyshevTWeights{T}
    function ChebyshevTGaussMeasure{T}(n::Int) where T
        x, w = GridArrays.gausschebyshev(T,n)
        new{T}(x, w)
    end
    ChebyshevTGaussMeasure(n::Int) = ChebyshevTGaussMeasure{Float64}(n)
end

struct ChebyshevUGaussMeasure{T} <:DiscreteMeasure{T}
    grid   :: GridArrays.ChebyshevUNodes{T}
    weights:: GridArrays.ChebyshevUWeights{T}
    function ChebyshevUGaussMeasure{T}(n::Int) where T
        x, w = GridArrays.gausschebyshevu(T,n)
        new{T}(x, w)
    end
    ChebyshevUGaussMeasure(n::Int) = ChebyshevUGaussMeasure{Float64}(n)
end

struct LegendreGaussMeasure{T} <: DiscreteMeasure{T}
    grid    :: GridArrays.LegendreNodes{T}
    weights :: GridArrays.LegendreWeights{T}

    function LegendreGaussMeasure{T}(n::Int) where T
        x, w = GridArrays.gausslegendre(T, n)
        new{T}(x, w)
    end
    LegendreGaussMeasure(n::Int) = LegendreGaussMeasure{Float64}(n)
end

struct LaguerreGaussMeasure{T} <: DiscreteMeasure{T}
    α       :: T
    grid    :: GridArrays.LaguerreNodes{T}
    weights :: GridArrays.LaguerreWeights{T}

    function LaguerreGaussMeasure{T}(n::Int, α::T) where T
        x, w = GridArrays.gausslaguerre(T, n, α)
        new{T}(α, x, w)
    end
    LaguerreGaussMeasure(n::Int, α::T) where T = LaguerreGaussMeasure{T}(n, α)
end

struct HermiteGaussMeasure{T} <: DiscreteMeasure{T}
    grid    :: GridArrays.HermiteNodes{T}
    weights :: GridArrays.HermiteWeights{T}

    function HermiteGaussMeasure{T}(n::Int) where T
        x, w = GridArrays.gausshermite(T,n)
        new{T}(x, w)
    end
    HermiteGaussMeasure(n::Int) = HermiteGaussMeasure{Float64}(n)
end

struct JacobiGaussMeasure{T} <: DiscreteMeasure{T}
    α       :: T
    β       :: T
    grid    :: GridArrays.JacobiNodes{T}
    weights :: GridArrays.JacobiWeights{T}

    function JacobiGaussMeasure{T}(n::Int, α::T, β::T) where T
        x, w = GridArrays.gaussjacobi(T, n, α, β)
        new{T}(α, β, x, w)
    end
    JacobiGaussMeasure(n::Int, α::T, β::T) where T = JacobiGaussMeasure{T}(n, α, β)
end
