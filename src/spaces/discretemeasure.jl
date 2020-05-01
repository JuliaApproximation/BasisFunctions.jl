
@deprecate grid(m::DiscreteMeasure) points(m)

function default_applymeasure(measure::DiscreteMeasure, f::Function; options...)
    integral(f, measure; options...)
end


struct GenericDiscreteMeasure{T,GRID<:AbstractGrid,W} <: DiscreteMeasure{T}
    points    ::  GRID
    weights   ::  W
    function GenericDiscreteMeasure(grid::AbstractGrid, weights)
        grid isa Union{AbstractSubGrid,TensorSubGrid} || @assert size(grid) == size(weights)
        new{eltype(grid),typeof(grid),typeof(weights)}(grid, weights)
    end
end
≈(m1::DiscreteMeasure, m2::DiscreteMeasure) = ≈(map(points,(m1,m2))...) &&  ≈(map(weights,(m1,m2))...)
genericweights(grid::AbstractGrid) = Ones{subeltype(eltype(grid))}(size(grid)...)
genericweights(grid::ProductGrid{T,S,N}) where {T,S,N} =
    tensorproduct(ntuple(k->Ones{subeltype(eltype(grid))}(size(grid,k)) ,Val(N)))

discretemeasure(grid::AbstractGrid, weights=genericweights(grid)) =
    GenericDiscreteMeasure(grid, weights)

name(m::GenericDiscreteMeasure) = "Generic discrete measure on grid $(typeof(points(m)))"
strings(m::GenericDiscreteMeasure) = (name(m), (string(points(m)),), (string(string(weights(m))),))
isnormalized(m::GenericDiscreteMeasure) = weights(m) isa NormalizedArray


name(m::DiracMeasure) = "Dirac measure at x = $(m.point)"
points(m::DiracMeasure) = ScatteredGrid([m.point])
weights(::DiracMeasure{T}) where T = Ones{T}(1)


const DiracComb{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W <: Ones
DiracComb(eg::AbstractEquispacedGrid) = discretemeasure(eg)
name(m::DiracComb) = "Dirac comb measure with sampling weights"


const WeightedDiracComb{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid
WeightedDiracComb(eg::AbstractEquispacedGrid, weights) = discretemeasure(eg, weights)
name(m::WeightedDiracComb) = "Dirac comb measure with generic weight"


const NormalizedDiracComb{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W<:NormalizedArray
NormalizedDiracComb(eg::AbstractEquispacedGrid) = discretemeasure(eg, NormalizedArray{eltype(eg)}(size(eg)))
isnormalized(m::NormalizedDiracComb) = true
name(m::NormalizedDiracComb) = "Dirac comb measure with normalized weights"

const UniformDiracComb{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:AbstractEquispacedGrid where W<:FillArrays.AbstractFill


const DiscreteMappedMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:MappedGrid
mappedmeasure(map, measure::DiscreteMeasure) =
discretemeasure(MappedGrid(points(measure), map), weights(measure))

name(m::DiscreteMappedMeasure) = "Mapping of a "*name(supermeasure(m))
mapping(measure::DiscreteMappedMeasure) = mapping(points(measure))
supermeasure(m::DiscreteMappedMeasure) = discretemeasure(supergrid(points(m)), m.weights)
apply_map(measure::DiscreteMeasure, map) = discretemeasure(apply_map(points(measure),map), weights(measure))
support(m::DiscreteMappedMeasure) = mapping(m) * support(supermeasure(m))



const DiscreteProductMeasure{T,G,W} = GenericDiscreteMeasure{T,G,W} where G<:ProductGrid where  W<:AbstractOuterProductArray

name(m::DiscreteProductMeasure) = "Discrete Product Measure"
productmeasure(measures::DiscreteMeasure...) =
    discretemeasure(ProductGrid(map(points, measures)...), tensorproduct(map(weights, measures)...))

iscomposite(m::DiscreteProductMeasure) = true
elements(m::DiscreteProductMeasure) = map(discretemeasure, elements(points(m)), elements(weights(m)))
element(m::DiscreteProductMeasure, i) = discretemeasure(element(points(m), i), element(weights(m), i))

function stencilarray(m::DiscreteProductMeasure)
    A = Any[]
    push!(A, element(m,1))
    for i = 2:length(elements(m))
        push!(A," ⊗ ")
        push!(A, element(m,i))
    end
    A
end

struct ChebyshevTGaussMeasure{T} <:DiscreteMeasure{T}
    points  :: GridArrays.ChebyshevTNodes{T}
    weights :: GridArrays.ChebyshevTWeights{T}
    function ChebyshevTGaussMeasure{T}(n::Int) where T
        x, w = GridArrays.gausschebyshev(T,n)
        new{T}(x, w)
    end
    ChebyshevTGaussMeasure(n::Int) = ChebyshevTGaussMeasure{Float64}(n)
end

struct ChebyshevUGaussMeasure{T} <:DiscreteMeasure{T}
    points :: GridArrays.ChebyshevUNodes{T}
    weights:: GridArrays.ChebyshevUWeights{T}
    function ChebyshevUGaussMeasure{T}(n::Int) where T
        x, w = GridArrays.gausschebyshevu(T,n)
        new{T}(x, w)
    end
end
ChebyshevUGaussMeasure(n::Int) = ChebyshevUGaussMeasure{Float64}(n)

struct LegendreGaussMeasure{T} <: DiscreteMeasure{T}
    points  :: GridArrays.LegendreNodes{T}
    weights :: GridArrays.LegendreWeights{T}

    function LegendreGaussMeasure{T}(n::Int) where T
        x, w = GridArrays.gausslegendre(T, n)
        new{T}(x, w)
    end
end
LegendreGaussMeasure(n::Int) = LegendreGaussMeasure{Float64}(n)

struct LaguerreGaussMeasure{T} <: DiscreteMeasure{T}
    α       :: T
    points  :: GridArrays.LaguerreNodes{T}
    weights :: GridArrays.LaguerreWeights{T}

    function LaguerreGaussMeasure{T}(n::Int, α::T) where T
        x, w = GridArrays.gausslaguerre(T, n, α)
        new{T}(α, x, w)
    end
end
LaguerreGaussMeasure(n::Int, α::T) where T = LaguerreGaussMeasure{T}(n, α)

struct HermiteGaussMeasure{T} <: DiscreteMeasure{T}
    points  :: GridArrays.HermiteNodes{T}
    weights :: GridArrays.HermiteWeights{T}

    function HermiteGaussMeasure{T}(n::Int) where T
        x, w = GridArrays.gausshermite(T,n)
        new{T}(x, w)
    end
end
HermiteGaussMeasure(n::Int) = HermiteGaussMeasure{Float64}(n)

struct JacobiGaussMeasure{T} <: DiscreteMeasure{T}
    α       :: T
    β       :: T
    points  :: GridArrays.JacobiNodes{T}
    weights :: GridArrays.JacobiWeights{T}

    function JacobiGaussMeasure{T}(n::Int, α::T, β::T) where T
        x, w = GridArrays.gaussjacobi(T, n, α, β)
        new{T}(α, β, x, w)
    end
end
JacobiGaussMeasure(n::Int, α::T, β::T) where T = JacobiGaussMeasure{T}(n, α, β)
