# platform.jl

"""
A platform represents a sequence of dictionaries.

A platform typically has a primal and a dual sequence. The platform maps an
index to a set of parameter values, which is then used to generate a primal
or dual dictionary.
"""
abstract type Platform
end

"""
A `GenericPlatform` stores a primal and dual dictionary generator, along with
a sequence of parameter values.
"""
struct GenericPlatform <: Platform
    super_platform
    primal_generator
    dual_generator
    sampler_generator
    parameter_sequence
    name
end

GenericPlatform(; super_platform=nothing, primal = None, dual = None, sampler = None, params = None,
    name = "Generic Platform") = GenericPlatform(super_platform, primal, dual, sampler, params, name)

function primal(platform::GenericPlatform, i)
    param = platform.parameter_sequence[i]
    platform.primal_generator(param)
end

function dual(platform::GenericPlatform, i)
    param = platform.parameter_sequence[i]
    platform.dual_generator(param)
end

function sampler(platform::GenericPlatform, i)
    param = platform.parameter_sequence[i]
    platform.sampler_generator(param)
end

name(platform::GenericPlatform) = platform.name

A(platform::Platform, i; options...) = apply(sampler(platform, i), primal(platform, i); options...)

function Z(platform::Platform, i; options...)
    dict = dual(platform, i)
    (coeftype(dict)(1)/length(dict))*apply(sampler(platform, i),dict; options...)
end

Zt(platform::Platform, i; options...) = Z(platform, i; options...)'

"""
Initalized with a series of generators, it generates tensorproduct dictionaries
given a series of lengths.
"""
struct TensorGenerator{T}
    fun
end
(TG::TensorGenerator)(n::Int...) = TG(collect(n))
(TG::TensorGenerator)(n::AbstractVector{Int}) = tensorproduct(TG.fun(n))

tensor_generator(::Type{T}, generators...) where {T} = TensorGenerator{T}( n ->([gi(ni)  for (ni, gi) in  zip(collect(n), collect(generators))]))


#######################
# Parameter sequences
#######################

"""
A `DimensionSequence` is a sequence of dimensions.
It can be indexed with an integer.
"""
abstract type DimensionSequence
end

"A doubling sequence with a given initial value."
struct DoublingSequence <: DimensionSequence
    initial ::  Int
end

# We arbitrarily choose a default initial value of 2
DoublingSequence() = DoublingSequence(2)

initial(s::DoublingSequence) = s.initial

getindex(s::DoublingSequence, idx::Int) = initial(s) * 2<<(idx-2)


"A doubling sequence with a given initial value."
struct MultiplySequence <: DimensionSequence
    initial ::  Int
    t       ::  Real
end

# We arbitrarily choose a default initial value of 2
MultiplySequence() = DoublingSequence()

initial(s::MultiplySequence) = s.initial

getindex(s::MultiplySequence, idx::Int) =  round(Int, initial(s) *s.t^(idx-1))


"A tensor product sequences with given initial values."
struct TensorSequence
    sequences
end

getindex(s::TensorSequence, idx::Int) = [si[idx] for si in s.sequences]
