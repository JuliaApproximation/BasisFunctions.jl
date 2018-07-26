# spaces.jl

"""
An `AbstractFunctionSpace{S,T}` is the supertype of all function spaces. The space
contains functions that map type `S` to type `T`.
Thus, `S` is the domaintype and `T` is the codomain type.
"""
abstract type AbstractFunctionSpace{S,T}
end

domaintype(::Type{AbstractFunctionSpace{S,T}}) where {S,T} = S
domaintype(::Type{S}) where {S <: AbstractFunctionSpace} = domaintype(supertype(S))
domaintype(::AbstractFunctionSpace{S,T}) where {S,T} = S

codomaintype(::Type{AbstractFunctionSpace{S,T}}) where {S,T} = T
codomaintype(::Type{S}) where {S <: AbstractFunctionSpace} = codomaintype(supertype(S))
codomaintype(::AbstractFunctionSpace{S,T}) where {S,T} = T

"""
`FunctionSpace` is a generic function space for functions about which we have
no additional information apart from their domain and codomain types.
"""
struct FunctionSpace{S,T} <: AbstractFunctionSpace{S,T}
end

# The zero element of a generic function space
zero(space::FunctionSpace{S,T}) where {S,T} = x::S -> zero(T)

# The multiplicative identity element of a generic function space
one(space::FunctionSpace{S,T}) where {S,T} = x::S -> one(T)
