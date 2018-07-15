# dictionary.jl


######################
# Type hierarchy
######################

"""
A `Dictionary{S,T}` is an ordered family of functions, in which each function
maps a variable of type `S` to a variable of type `T`.

A `Dictionary{S,T}` has domain type `S` and codomain type `T`. The domain type
corresponds to the type of a domain in the `Domains.jl` package, and it is the
type of the expected argument to the elements of the dictionary. The
codomain type is the type of the output.

Each dictionary is ordered via its index set: the ordering is determined by the
iterator of the index set. A dictionary `d` can be indexed in several ways:
- the linear index is a positive natural number between `1` and `length(d)`
- the natural index is an index that more closely corresponds to the conventional
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

FunDict = Dictionary

# Useful abstraction for special cases
const Dictionary1d{S <: Number,T} = Dictionary{S,T}
# Warning: not all 2d function sets have NTuple{2,S} type, they could have (S1,S2) type
const Dictionary2d{S <: Number,T} = Dictionary{NTuple{2,S},T}
const Dictionary3d{S <: Number,T} = Dictionary{NTuple{3,S},T}
const Dictionary4d{S <: Number,T} = Dictionary{NTuple{4,S},T}


"The type of the elements of the domain of the dictionary."
domaintype(::Type{Dictionary{S,T}}) where {S,T} = S
domaintype(::Type{D}) where {D <: Dictionary} = domaintype(supertype(D))
domaintype(dict::Dictionary{S,T}) where {S,T} = S

"The type of the elements of the codomain of the dictionary."
codomaintype(D::Type{Dictionary{S,T}}) where {S,T} = T
codomaintype(::Type{D}) where {D <: Dictionary} = codomaintype(supertype(D))
codomaintype(dict::Dictionary{S,T}) where {S,T} = T

"The type of the expansion coefficients in a dictionary."
# By default we set it equal to the codomaintype
coefficient_type(dict::Dictionary) = codomaintype(dict)

# The dimension of a function set is the dimension of its domain type
dimension(dict::Dictionary) = dimension(domaintype(dict))

dimension(dict::Dictionary, i) = dimension(element(dict, i))

"Are the functions in the dictionary real-valued?"
isreal(d::Dictionary) = isreal(codomaintype(d))


# TODO: we need to properly define the semantics of the functions that follow

# Is a given set a basis? In general, it is not. But it could be.
# Hence, we need a property for it:
is_basis(d::Dictionary) = false

# Any basis is a frame
is_frame(d::Dictionary) = is_basis(d)


"Property to indicate whether a dictionary is orthogonal."
is_orthogonal(d::Dictionary) = false

"Property to indicate whether a dictionary is orthonormal"
is_orthonormal(d::Dictionary) = false

"Property to indicate whether a dictionary is biorthogonal (or a Riesz basis)."
is_biorthogonal(d::Dictionary) = is_orthogonal(d)

"Return the size of the dictionary."
size(d::Dictionary) = (length(d),)

"Return the size of the j-th dimension of the dictionary (if applicable)."
size(d::Dictionary, j) = j==1 ? length(d) : throw(BoundsError())

endof(d::Dictionary) = length(d)

"Is the dictionary composite, i.e. does it consist of several components?"
is_composite(d::Dictionary) = false

"""
The instantiate function takes a dict type, size and numeric type as argument, and
returns an instance of the type with the given size and numeric type and using
default values for other parameters. This means the given type is usually abstract,
since it is given without parameters.

This function is mainly used to create instances for testing purposes.
"""
instantiate{S <: Dictionary}(::Type{S}, n) = instantiate(S, n, Float64)


##############################
# Domain and codomain type
##############################

"Promote the domain type of the dictionary."
promote_domaintype(dict::Dictionary{S,T}, ::Type{S}) where {S,T} = dict
promote_domaintype(dict::Dictionary{S,T}, ::Type{U}) where {S,T,U} = dict_promote_domaintype(dict, U)

promote_domaintype(dict1::Dictionary{S,T1}, dict2::Dictionary{S,T2}) where {S,T1,T2} = (dict1,dict2)

function promote_domaintype(dict1::Dictionary{S1,T1}, dict2::Dictionary{S2,T2}) where {S1,S2,T1,T2}
    S = promote_type(S1,S2)
    promote_domaintype(dict1, S), promote_domaintype(dict2, S)
end

promote_domaintype(dict1::Dictionary, dict2::Dictionary, dicts::Dictionary...) =
    promote_domaintype(promote_domaintype(dict1,dict2), dicts...)

"Promote the domain sub type of the dictionary."
promote_domainsubtype(d::Dictionary{S,T}, ::Type{U}) where {S<:Number,T,U<:Number} = promote_domaintype(d, U)

promote_domainsubtype(dict::Dictionary{SVector{N,S},T}, ::Type{S}) where {N,S<:Number,T} = dict
promote_domainsubtype(dict::Dictionary{SVector{N,S},T}, ::Type{U}) where {N,S<:Number,T,U} =
    promote_domaintype(dict, SVector{N,U})

promote_domainsubtype(dict::Dictionary{NTuple{N,S},T}, ::Type{S}) where {N,S<:Number,T} = dict
promote_domainsubtype(dict::Dictionary{NTuple{N,S},T}, ::Type{U}) where {N,S<:Number,T,U} =
    promote_domaintype(dict, NTuple{N,U})

"Promote the coefficient type of the dictionary."
promote_coefficient_type(dict::Dictionary{S,T}, ::Type{T}) where {S,T} = dict
promote_coefficient_type(dict::Dictionary{S,T}, ::Type{U}) where {S,T,U} = dict_promote_coeftype(dict, U)

promote_coefficient_type(dict1::Dictionary{S1,T}, dict2::Dictionary{S2,T}) where {S1,S2,T} = (dict1,dict2)

function promote_coefficient_type(dict1::Dictionary{S1,T1}, dict2::Dictionary{S2,T2}) where {S1,S2,T1,T2}
    T = promote_type(T1,T2)
    promote_coefficient_type(dict1, T), promote_coefficient_type(dict2, T)
end

promote_coefficient_type(dict1::Dictionary, dict2::Dictionary, dicts::Dictionary...) =
    promote_coefficient_type(promote_coefficient_type(dict1,dict2), dicts...)


promote_coeftype = promote_coefficient_type

widen(d::Dictionary) = promote_domaintype(d, widen(domaintype(d)))

# similar returns a similar basis of a given size and numeric type
# It can be implemented in terms of resize and promote_domaintype.
similar(d::Dictionary, ::Type{T}, n) where {T} = resize(promote_domaintype(d, T), n)

# Support resize of a 1D set with a tuple of a single element, so that one can
# write statements of the form resize(s, size(some_set)) in all dimensions.
resize(s::Dictionary1d, n::NTuple{1,Int}) = resize(s, n[1])



"Return a set of zero coefficients in the native format of the set."
zeros(s::Dictionary) = zeros(coefficient_type(s), s)
ones(s::Dictionary) = ones(coefficient_type(s), s)

# By default we assume that the native format corresponds to an array of the
# same size as the set. This is not true, e.g., for multidicts.
zeros(::Type{T}, s::Dictionary) where {T} = zeros(T, size(s))
ones(::Type{T}, s::Dictionary) where {T} = ones(T, size(s))


function rand(dict::Dictionary)
    c = zeros(dict)
    T = coeftype(dict)
    for i in eachindex(c)
        c[i] = random_value(T)
    end
    c
end


###########
# Indexing
###########

"""
Dictionaries are ordered lists. Their ordering is defined by the way their
index sets are ordered.

The `ordering` of a dictionary returns a list-like object that can be indexed
with integers between `1` and `length(dict)`. This operation returns the
corresponding native index. This defines the ordering of the native index set
of the dictionary.
"""
ordering(dict::Dictionary) = Base.OneTo(length(dict))

# By convention, `eachindex` returns the most efficient way to iterate over the
# indices of a dictionary. This is not necessarily the linear index.
# We call eachindex on `ordering(d)`.
eachindex(d::Dictionary) = eachindex(ordering(d))

"Compute the native index corresponding to the given index."
native_index(dict::Dictionary, idx) = _native_index(dict, idx)
# We redirect to a fallback _native_index in case the concrete dictionary
# did not implement native_index.  We explicitly convert a linear index using the ordering.
# Anything else we throw an error because the index looks invalid
_native_index(dict::Dictionary, idx::LinearIndex) = ordering(dict)[idx]
_native_index(dict::Dictionary, idx::NativeIndex) = idx
_native_index(dict::Dictionary, idx) = throw(ArgumentError("invalid index: $idx"))

"Compute the linear index corresponding to the given index."
linear_index(dict::Dictionary, idx) = _linear_index(dict, idx)
# We can accept an integer unchanged, anything else we pass to the ordering
_linear_index(dict::Dictionary, idx::LinearIndex) = idx
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
    copy!(coef_linear, coef_native)
# Note that copy! is defined in util/common.jl

"""
Convert a linear set of coefficients back to the native representation of the dictionary.
"""
function delinearize_coefficients(dict::Dictionary, coef_linear::Vector)
    coef_native = zeros(eltype(coef_linear), dict)
    delinearize_coefficients!(dict, coef_native, coef_linear)
end

delinearize_coefficients!(dict::Dictionary, coef_native, coef_linear::Vector) =
    copy!(coef_native, coef_linear)

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

# The following properties are not implemented as traits with types, because they are
# not intended to be used in a time-critical path of the code.

"Does the dictionary implement a derivative?"
has_derivative(d::Dictionary) = false

"Does the dictionary implement an antiderivative?"
has_antiderivative(d::Dictionary) = false

"Does the dictionary have an associated interpolation grid?"
has_grid(d::Dictionary) = false

"Does the dictionary have a transform associated with some space?"
has_transform(d1::Dictionary, d2) = false

"Does the dictionary have a transform associated with some space that is unitary"
has_unitary_transform(d::Dictionary) = has_transform(d)
# If a dict has a transform, we assume it is unitary. If it is not,
# this function has to be over written.

# Convenience functions: default grid, and conversion from grid to space
has_transform(d::Dictionary) = has_grid(d) && has_transform(d, grid(d))
has_transform(d::Dictionary, grid::AbstractGrid) =
    has_transform(d, gridbasis(grid, codomaintype(d)))

"Does the grid span the same interval as the dictionary"
has_grid_equal_span(set::Dictionary1d, grid::AbstractGrid1d) =
    (1+(infimum(support(set)) - leftendpoint(grid))≈1) && (1+(supremum(support(set)) - rightendpoint(grid))≈1)

"Does the dictionary support extension and restriction operators?"
has_extension(d::Dictionary) = false


# A concrete Dictionary may also override extension_set and restriction_set
# The default is simply to resize.
extension_set(d::Dictionary, n) = resize(d, n)
restriction_set(d::Dictionary, n) = resize(d, n)


###############################
## Iterating over dictionaries
###############################

# Default iterator over sets of functions: based on underlying index iterator.
function start(d::Dictionary)
    iter = eachindex(d)
    (iter, start(iter))
end

function next(d::Dictionary, state)
    iter, iter_state = state
    idx, iter_newstate = next(iter,iter_state)
    (d[idx], (iter,iter_newstate))
end

done(d::Dictionary, state) = done(state[1], state[2])




####################
## Bounds checking
####################

# We hook into Julia's bounds checking system. See the Julia documentation.
# This is based on the set of indices, as returned by `indices(dict)`.
# One thing to take into account in our setting is that the map from linear indices
# to native indices and vice-versa requires knowledge of the dictionary. Hence,
# we do some conversions before `indices(dict)` is called and passed on.
checkbounds(dict::Dictionary, I...) = checkbounds(Bool, dict, I...) || throw(BoundsError())

# We make a special case for a linear index
checkbounds(::Type{Bool}, dict::Dictionary, i::LinearIndex) = checkindex(Bool, linearindices(dict), i)

# We also convert native indices to linear indices, before moving on.
# (This is more difficult to do later on, e.g. in checkindex, because that routine
#  does not have access to the dict anymore)
checkbounds(::Type{Bool}, dict::Dictionary, i::NativeIndex) =
    checkbounds(Bool, dict, linear_index(dict, i))
checkbounds(::Type{Bool}, dict::Dictionary, i::MultilinearIndex) =
    checkbounds(Bool, dict, linear_index(dict, i))

# And here we call checkbounds_indices with indices(dict)
checkbounds(::Type{Bool}, dict::Dictionary, I...) = checkbounds_indices(Bool, indices(dict), I)

"Return the support of the idx-th basis function. Default is support of the dictionary."
support(dict::Dictionary, idx) = support(dict)
# Warning: the functions above and below may be wrong for certain concrete
# dictionaries, for example for univariate functions with non-connected support.
# Make sure to override, and make sure that the overridden version is called.

tolerance(dict::Dictionary) = tolerance(domaintype(dict))

"Does the given point lie inside the support of the given set function or dictionary?"
in_support(dict::Dictionary, idx, x) = default_in_support(dict, idx, x)
in_support(dict::Dictionary, x) = default_in_support(dict, x)

# Add a fallback for linear indices, convert to native index so that it is sufficient
# for dictionaries to implement native indices
in_support(dict::Dictionary, idx::LinearIndex, x) =
    in_support(dict, native_index(dict, idx), x)

default_in_support(dict::Dictionary, idx, x) = approx_in(x, support(dict, idx), tolerance(dict))
default_in_support(dict::Dictionary, x) = approx_in(x, support(dict), tolerance(dict))


##############################################
## Evaluating set elements and expansions
##############################################

"""
A member function of a dictionary is evaluated using the `eval_element` routine.
It takes as arguments the dictionary, the index of the member function and
the point in which to evaluate.

This function performs bounds checking on the index and also checks whether the
point x lies inside the support of the function. A `BoundsError` is thrown for
an index out of bounds. The value `0` is returned when x is outside the support.

After the check on the index, the function calls `unsafe_eval_element1.` This
function checks whether `x` lies in the support, and then calls
`unsafe_eval_element`. The latter function should be implemented by a concrete
dictionary. Any user who wants to avoid the bounds check or the support check
can intercept `eval_element` or `unsafe_eval_element1` respectively.
"""
function eval_element(dict::Dictionary, idx, x)
    # We convert to a native index before bounds checking
    idxn = native_index(dict, idx)
    @boundscheck checkbounds(dict, idxn)
    unsafe_eval_element1(dict, idxn, x)
end

# For linear indices, bounds checking is very efficient, so we intercept this case
# and only convert to a native index after the bounds check.
function eval_element(dict::Dictionary, idx::LinearIndex, x)
    @boundscheck checkbounds(dict, idx)
    unsafe_eval_element1(dict, native_index(dict, idx), x)
end

unsafe_eval_element1(dict::Dictionary, idx, x) =
    in_support(dict, idx, x) ? unsafe_eval_element(dict, idx, x) : zero(codomaintype(dict))

# Catch any index and convert to native index, in case it got through to here
unsafe_eval_element(dict::Dictionary, idx, x) =
    unsafe_eval_element(dict, native_index(dict, idx), x)

"""
Evaluate a member function with a boundscheck on the index, but without checking
the support of the function.
"""
function eval_element_extension(dict::Dictionary, idx, x)
    @boundscheck checkbounds(dict, idx)
    # We skip unsafe_evaluate_element1 and jump to unsafe_eval_element
    unsafe_eval_element(dict, native_index(dict, idx), x)
end

# Convenience function: evaluate a function on a grid.
# We implement unsafe_eval_element1, so the bounds check on idx has already happened
# TODO: implement using broadcast instead, because evaluation in a grid is like vectorization
@inline unsafe_eval_element1(dict::Dictionary, idx, grid::AbstractGrid) =
    _default_unsafe_eval_element_in_grid(dict, idx, grid)

function _default_unsafe_eval_element_in_grid(dict::Dictionary, idx, grid::AbstractGrid)
    result = zeros(gridbasis(grid, codomaintype(dict)))
    for k in eachindex(grid)
        @inbounds result[k] = eval_element(dict, idx, grid[k])
    end
    result
end


"""
This function is exactly like `eval_element`, but it evaluates the derivative
of the element instead.
"""
function eval_element_derivative(dict::Dictionary, idx, x)
    idxn = native_index(dict, idx)
    @boundscheck checkbounds(dict, idxn)
    unsafe_eval_element_derivative1(dict, idxn, x)
end

function eval_element_derivative(dict::Dictionary, idx::LinearIndex, x)
    @boundscheck checkbounds(dict, idx)
    unsafe_eval_element_derivative1(dict, native_index(dict, idx), x)
end

function eval_element_extension_derivative(dict::Dictionary, idx, x)
    @boundscheck checkbounds(dict, idx)
    unsafe_eval_element_derivative(dict, native_index(dict, idx), x)
end

function unsafe_eval_element_derivative1(dict::Dictionary{S,T}, idx, x) where {S,T}
    in_support(dict, idx, x) ? unsafe_eval_element_derivative(dict, idx, x) : zero(T)
end

unsafe_eval_element_derivative(dict::Dictionary, idx, x) =
    unsafe_eval_element_derivative(dict, native_index(dict, idx), x)



"""
Evaluate an expansion given by the set of coefficients in the point x.
"""
function eval_expansion(dict::Dictionary, coefficients, x)
    @assert size(coefficients) == size(dict)

    T = span_codomaintype(dict)
    z = zero(T)
    # It is safer below to use eval_element than unsafe_eval_element, because of
    # the check on the support.
    @inbounds for idx in eachindex(coefficients)
        z = z + coefficients[idx] * eval_element(dict, idx, x)
    end
    z
end

function eval_expansion(dict::Dictionary, coefficients, grid::AbstractGrid)
    @assert dimension(dict) == dimension(grid)
    @assert size(coefficients) == size(dict)
    # TODO: reenable test once product grids and product sets have compatible types again
    # @assert eltype(grid) == domaintype(dict)

    T = coeftype(dict)
    E = evaluation_operator(dict, gridbasis(grid, T))
    E * coefficients
end


#######################
## Application support
#######################

"""
Compute the moment of the given basisfunction, i.e. the integral on its
support.
"""
function moment(dict::Dictionary1d, idx)
    @boundscheck checkbounds(dict, idx)
    unsafe_moment(dict, native_index(dict, idx))
end

# This routine is called after the boundscheck. Call another function,
# default moment, so that unsafe_moment of the concrete dictionary can still
# fall back to `default_moment` as well for some values of the index.
unsafe_moment(dict::Dictionary1d, idx) = default_moment(dict, idx)

# Default to numerical integration
default_moment(dict::Dictionary1d, idx) = quadgk(dict[idx], left(d), right(d))[1]
