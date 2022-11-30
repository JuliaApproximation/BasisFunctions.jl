
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

show(io::IO, s::FunctionSpace) = print(io, "Space : $(domaintype(s)) → $(codomaintype(s))")
Display.object_parentheses(s::FunctionSpace) = true


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
A function space equipped with a measure.

The measure `μ` induces a bilinear form `(f,g) = int(f, g, dμ)` and possibly a
(semi)norm or inner product. This depends on the properties of the measure.
"""
struct MeasureSpace{M <: Measure,S,T} <: FunctionSpace{S,T}
    measure     ::  M

    # Ensure that the domain types of the space and the measure are the same
    MeasureSpace{M,S,T}(measure::Weight{S}) where {M,S,T} = new(measure)
end

MeasureSpace(measure::Weight{T}) where {T} =
    MeasureSpace{typeof(measure),T,codomaintype(measure)}(measure)

measure(s::MeasureSpace) = s.measure

domain(s::MeasureSpace) = support(measure(s))

space(m::Weight) = MeasureSpace(m)

show(io::IO, s::MeasureSpace) = print(io, "Function space with measure: $(measure(s))")
Display.object_parentheses(s::MeasureSpace) = true


const FourierSpace{T} = MeasureSpace{FourierWeight{T},T,T}

FourierSpace() = FourierSpace{Float64}()
FourierSpace{T}() where {T} = MeasureSpace(FourierWeight{T}())
show(io::IO, s::FourierSpace) = print(io, "Fourier space : $(domaintype(s)) → $(codomaintype(s))")


const ChebyshevTSpace{T} = MeasureSpace{ChebyshevTWeight{T},T,T}

ChebyshevTSpace() = ChebyshevTSpace{Float64}()
ChebyshevTSpace{T}() where {T} = MeasureSpace(ChebyshevTWeight{T}())


"The space of Lebesgue integrable functions."
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

"A product function space."
struct ProductSpace{S,T} <: FunctionSpace{S,T}
    spaces
end

function tensorproduct(spaces::FunctionSpace...)
    S = Tuple{map(domaintype, spaces)...}
    T = mapreduce(codomaintype, promote_type, spaces)
    ProductSpace{S,T}(spaces)
end

components(space::ProductSpace) = space.spaces
