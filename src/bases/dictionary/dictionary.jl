
export domaintype,
    codomaintype,
    coefficienttype,
    prectype


"""
A `Dictionary{S,T}` is an ordered family of functions, in which each function
maps a variable of type `S` to a variable of type `T`. The dictionary can be
thought of as an array, where the elements are functions that map `S` to `T`.

A `Dictionary{S,T}` has domain type `S` and codomain type `T`. The domain type
corresponds to the type of a domain in the `DomainSets.jl` package, and it is the
type of the expected argument to the elements of the dictionary. The
codomain type is the type of the output.

Each dictionary is ordered via its index set: the ordering is determined by the
iterator of the index set. A dictionary `d` can be indexed in several ways:
- the linear index is a positive natural number between `1` and `length(d)`
- the native index is an index that more closely corresponds to the conventional
  mathematical notation of the dictionary, if that differs from linear indexing.
  For example, polynomials may have degree ranging from `0` to `length(d)-1`.
  Fourier series may have negative frequencies.

Some dictionaries define other types of indices, such as multilinear and product
indices. See `CompositeDictionary` and `ProductDictionary`. There is always a
bijection between the different types of index sets supported by a dictionary.

A dictionary is the basic building block of the package. They can support several
operations, such as interpolation, evaluation, differentiation and others. This
functionality is made available using a specific interface. See the documentation
of the features for a description of that interface and the syntax.
"""
abstract type Dictionary{S,T}
end

# Useful abstraction for special cases
const Dictionary1d{S <: Number,T} = Dictionary{S,T}
# Warning: these are shaky definitions of multivariate functions
const Dictionary2d{S <: Number,T} = Union{Dictionary{NTuple{2,S},T},Dictionary{SVector{2,S},T}}
const Dictionary3d{S <: Number,T} = Union{Dictionary{NTuple{3,S},T},Dictionary{SVector{3,S},T}}
const Dictionary4d{S <: Number,T} = Union{Dictionary{NTuple{4,S},T},Dictionary{SVector{4,S},T}}


"The type of the elements of the domain of the dictionary."
domaintype(::Type{<:Dictionary{S,T}}) where {S,T} = S
domaintype(dict::Dictionary) = domaintype(typeof(dict))

"The type of the elements of the codomain of the dictionary."
codomaintype(::Type{<:Dictionary{S,T}}) where {S,T} = T
codomaintype(dict::Dictionary) = codomaintype(typeof(dict))

"The type of the expansion coefficients in a dictionary."
# By default we set it equal to the codomaintype
coefficienttype(D::Type{<:Dictionary{S,T}}) where {S,T} = codomaintype(D)
coefficienttype(dict::Dictionary) = coefficienttype(typeof(dict))

prectype(D::Type{<:Dictionary}) = prectype(domaintype(D), codomaintype(D))
numtype(D::Type{<:Dictionary}) = numtype(domaintype(D), codomaintype(D))

# The dimension of a function set is the dimension of its domain type
dimension(dict::Dictionary) = dimension(domaintype(dict))

dimension(dict::Dictionary, i) = dimension(element(dict, i))

"Are the functions in the dictionary real-valued?"
isreal(d::Dictionary) = isreal(codomaintype(d))


"Is the dictionary a (truncation of a) basis?"
isbasis(d::Dictionary) = false

"Is the dictionary orthogonal (with respect to the given measure)?"
isorthogonal(d::Dictionary) = hasmeasure(d) && isorthogonal(d, measure(d))
isorthogonal(d::Dictionary, μ::AbstractMeasure) = isorthonormal(d, μ)

"Is the dictionary orthonormal (with respect to the given measure)?"
isorthonormal(d::Dictionary) = hasmeasure(d) && isorthonormal(d, measure(d))
isorthonormal(d::Dictionary, μ::AbstractMeasure) = false

"Is the dictionary biorthogonal (with respect to the given measure)?"
isbiorthogonal(d::Dictionary) = hasmeasure(d) && isbiorthogonal(d, measure(d))
isbiorthogonal(d::Dictionary, μ::AbstractMeasure) = isorthogonal(d, μ)

size(d::Dictionary, j) = size(d)[j]

"""
The length and size of a dictionary may not be enough to uniquely determine the
size of the dictionary, if it has additional internal structure. The output of
dimensions is such that `resize(dict, dimensions(dict)) == dict`.
"""
dimensions(d::Dictionary1d) = length(d)
dimensions(d::Dictionary) = size(d)

length(d::Dictionary) = prod(size(d))

firstindex(d::Dictionary) = first(eachindex(d))
lastindex(d::Dictionary) = last(eachindex(d))


#############################
# Domain and codomain type
#############################

similar(s::Dictionary, ::Type{T}) where {T} = similar(s, T, size(s))

similar(s::Dictionary, size::Int...) = similar(s, domaintype(s), size...)
similar(s::Dictionary, dims::Base.Dims) = similar(s, domaintype(s), dims...)

similar(s::Dictionary, ::Type{T}, dims::Base.Dims) where {T} = similar(s, T, dims...)

# This is a default routine that ony appies when nothing changes.
function similar(dict::Dictionary{S,T}, ::Type{S}, n::Int) where {S,T}
    @assert n == length(dict)
    dict
end

resize(s::Dictionary, dims...) = similar(s, domaintype(s), dims...)


widen(d::Dictionary) = similar(d, widen(domaintype(d)))


"Return a set of zero coefficients in the native format of the set."
zeros(s::Dictionary) = zeros(coefficienttype(s), s)
ones(s::Dictionary) = ones(coefficienttype(s), s)

# By default we assume that the native format corresponds to an array of the
# same size as the set. This is not true, e.g., for multidicts.
zeros(::Type{T}, s::Dictionary) where {T} = zeros(T, size(s))
ones(::Type{T}, s::Dictionary) where {T} = ones(T, size(s))

"What is the type of the coefficient vector of the dictionary?"
containertype(d::Dictionary) = typeof(zeros(d))

"Transform `a` to a coefficient vector for the given dictionary."
tocoefficientformat(a, d::Dictionary) = reshape(a, size(d))

function rand(dict::Dictionary)
    c = zeros(dict)
    T = coefficienttype(dict)
    for i in eachindex(c)
        c[i] = rand(T)
    end
    c
end


###########
# Indexing
###########

IndexStyle(d::Dictionary) = IndexLinear()

"""
Dictionaries are ordered lists. Their ordering is defined by the way their
index sets are ordered.

The `ordering` of a dictionary returns a list-like object that can be indexed
with integers between `1` and `length(dict)`. This operation returns the
corresponding native index. This defines the ordering of the native index set
of the dictionary.
"""
ordering(dict::Dictionary) = Base.OneTo(length(dict))

eachindex(d::Dictionary) = eachindex(IndexStyle(d), d)
eachindex(::IndexLinear, d::Dictionary) = axes1(d)
eachindex(::IndexCartesian, d::Dictionary) = CartesianIndices(axes(d))

axes1(d::Dictionary) = Base.OneTo(length(d))
axes(d::Dictionary) = map(Base.OneTo, size(d))

"Compute the native index corresponding to the given index."
native_index(dict::Dictionary, idx) = _native_index(dict, idx)
# We redirect to a fallback _native_index in case the concrete dictionary
# did not implement native_index.  We explicitly convert a linear index using the ordering.
# Anything else we throw an error because the index looks invalid
_native_index(dict::Dictionary, idx::Int) = ordering(dict)[idx]
_native_index(dict::Dictionary, idx::NativeIndex) = idx
_native_index(dict::Dictionary, idx::AbstractShiftedIndex) = idx
_native_index(dict::Dictionary, idx) = throw(ArgumentError("invalid index: $idx"))

"Compute the linear index corresponding to the given index."
linear_index(dict::Dictionary, idx) = _linear_index(dict, idx)
# We can accept an integer unchanged, anything else we pass to the ordering
_linear_index(dict::Dictionary, idx::Int) = idx
_linear_index(dict::Dictionary, idxn) = ordering(dict)[idxn]


##################################################
## Conversion between coefficient representations
##################################################

"""
Convert the set of coefficients in the native format of the dictionary
to a linear list in a vector.
"""
function linearize_coefficients(dict::Dictionary, coef_native)
    coef_linear = zeros(eltype(coef_native), length(dict))
    linearize_coefficients!(dict, coef_linear, coef_native)
end

linearize_coefficients!(dict::Dictionary, coef_linear::Vector, coef_native) =
    copyto!(coef_linear, coef_native)
# Note that copyto! is defined in util/common.jl

"""
Convert a linear set of coefficients back to the native representation of the dictionary.
"""
function delinearize_coefficients(dict::Dictionary, coef_linear::Vector)
    coef_native = zeros(eltype(coef_linear), dict)
    delinearize_coefficients!(dict, coef_native, coef_linear)
end

delinearize_coefficients!(dict::Dictionary, coef_native, coef_linear::Vector) =
    copyto!(coef_native, coef_linear)

"Promote the given coefficients to the native representation of the dictionary."
native_coefficients(dict::Dictionary, coef) = _native_coefficients(dict, coef)
# TODO: we create an unnecessary copy here if the native type is a vector
_native_coefficients(dict::Dictionary, coef::Vector) = delinearize_coefficients(dict, coef)
_native_coefficients(dict::Dictionary, coef) = coef

# Sets have a native size and a linear size. However, there is not necessarily a
# bijection between the two. You can always convert a native size to a linear size,
# but the other direction can be done in general only approximately.
# For example, a 2D tensor product set can only support sizes of the form n1 * n2. Its native size may be
# (n1,n2) and its linear size n1*n2, but not any integer n maps to a native size tuple.
# By convention, we denote a native size variable by size_n.
"Compute the native size best corresponding to the given linear size."
approximate_native_size(d::Dictionary, size_l) = size_l

"Compute the linear size corresponding to the given native size."
linear_size(d::Dictionary, size_n) = size_n

"Suggest a suitable size, close to `n`, to resize the given dictionary."
approx_length(d::Dictionary, n::Int) = n
approx_length(d::Dictionary, n::Real) = approx_length(d, round(Int,n))




###############################
## Properties of function sets
###############################

"""
Does the dictionary implement a differentiation operator?
An optional second argument may specify an exact order of the derivative.
"""
hasderivative(Φ::Dictionary) = false

function hasderivative(Φ::Dictionary, order)
    # We have to be correct for zero order derivatives, because even if the
    # dictionary itself does not support derivatives, it could be part of a
    # composition where other elements do
    if orderiszero(order)
        true
    else
        order == 1 ? hasderivative(Φ) : false
    end
end

"Does the dictionary implement an antiderivative?"
hasantiderivative(Φ::Dictionary) = false

function hasantiderivative(Φ::Dictionary, order)
    if orderiszero(order)
        true
    else
        order == 1 ? hasantiderivative(Φ) : false
    end
end


"Does the dictionary have an associated interpolation grid?"
hasinterpolationgrid(d::Dictionary) = false

function grid(d::Dictionary)
    error("replace grid(dict) by interpolation_grid(dict)")
    interpolation_grid(d)
end

"Does the dictionary have a transform associated with some space?"
hastransform(d1::Dictionary, d2) = false

# Convenience functions: default grid, and conversion from grid to space
hastransform(d::Dictionary) = hasinterpolationgrid(d) && hastransform(d, interpolation_grid(d))
hastransform(d::Dictionary, grid::AbstractGrid) =
    hastransform(d, GridBasis{coefficienttype(d)}(grid))

"Does the dictionary support extension and restriction operators?"
hasextension(d::Dictionary) = false


# A concrete Dictionary may also override extension_set and restriction_set
# The default is simply to resize.
extension_set(d::Dictionary, n) = resize(d, n)
restriction_set(d::Dictionary, n) = resize(d, n)


###############################
## Iterating over dictionaries
###############################

# Default iterator over sets of functions: based on underlying index iterator.
function iterate(d::Dictionary)
    iter = eachindex(d)
    first_item, first_state = iterate(iter)
    (d[first_item], (iter, (first_item, first_state)))
end

function iterate(d::Dictionary, state)
    iter, iter_tuple = state
    iter_item, iter_state = iter_tuple
    next_tuple = iterate(iter, iter_state)
    if next_tuple != nothing
        next_item, next_state = next_tuple
        (d[next_item], (iter,next_tuple))
    end
end



include("dict_evaluation.jl")
include("dict_moments.jl")
