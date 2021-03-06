
"""
A `FunctionSpace{S,T}` is the supertype of all function spaces. The space
contains functions that map type `S` to type `T`.
Thus, `S` is the domaintype and `T` is the codomain type.
"""
abstract type FunctionSpace{S,T}
end

domaintype(::Type{FunctionSpace{S,T}}) where {S,T} = S
domaintype(::Type{S}) where {S <: FunctionSpace} = domaintype(supertype(S))
domaintype(::FunctionSpace{S,T}) where {S,T} = S

codomaintype(::Type{FunctionSpace{S,T}}) where {S,T} = T
codomaintype(::Type{S}) where {S <: FunctionSpace} = codomaintype(supertype(S))
codomaintype(::FunctionSpace{S,T}) where {S,T} = T

length(s::FunctionSpace) = Inf

"""
`GenericFunctionSpace` is a generic function space for functions about which we
have no additional information apart from their domain and codomain types.
"""
struct GenericFunctionSpace{S,T} <: FunctionSpace{S,T}
end

# The zero element of a generic function space
zero(space::GenericFunctionSpace{S,T}) where {S,T} = x::S -> zero(T)

# The multiplicative identity element of a generic function space
one(space::GenericFunctionSpace{S,T}) where {S,T} = x::S -> one(T)


"""
The Hilbert space of all functions for which the inner product induced by a
measure is bounded.
"""
struct MeasureSpace{M <: Measure,S,T} <: FunctionSpace{S,T}
    measure     ::  M

    # Ensure that the domain types of the space and the measure are the same
    MeasureSpace{M,S,T}(measure::Measure{S}) where {M,S,T} = new(measure)
end

MeasureSpace(measure::Measure{T}) where {T} =
    MeasureSpace{typeof(measure),T,codomaintype(measure)}(measure)

measure(s::MeasureSpace) = s.measure

domain(s::MeasureSpace) = support(measure(s))

space(m::Measure) = MeasureSpace(m)


const FourierSpace{T} = MeasureSpace{FourierMeasure{T},T,T}

FourierSpace() = FourierSpace{Float64}()
FourierSpace{T}() where {T} = MeasureSpace(FourierMeasure{T}())


const ChebyshevTSpace{T} = MeasureSpace{ChebyshevMeasure{T},T,T}

ChebyshevTSpace() = ChebyshevTSpace{Float64}()
ChebyshevTSpace{T}() where {T} = MeasureSpace(ChebyshevTMeasure{T}())



struct L2{S,T} <: FunctionSpace{S,T}
    domain  ::  Domain{S}
end

L2() = L2{Float64}()
L2{T}() where {T} = L2{T}(DomainSets.FullSpace{T}())

L2(domain::Domain{S}) where {S} = L2{subeltype(S)}(domain)
L2{T}(domain::Domain{S}) where {S,T} = L2{S,T}(domain)

domain(space::L2) = space.domain


abstract type WeightedL2{S,T} <: FunctionSpace{S,T} end

struct ChebyshevSpace{S,T} <: WeightedL2{S,T}
end

ChebyshevSpace() = ChebyshevSpace{Float64}()
ChebyshevSpace{T}() where {T} = ChebyshevSpace{T,T}()

domain(space::ChebyshevSpace{S,T}) where {S,T} = ChebyshevInterval{S}()


struct ProductSpace{S,T} <: FunctionSpace{S,T}
    spaces
end

function tensorproduct(spaces::FunctionSpace...)
    S = Tuple{map(domaintype, spaces)...}
    T = mapreduce(codomaintype, promote_type, spaces)
    ProductSpace{S,T}(spaces)
end

elements(space::ProductSpace) = space.spaces
