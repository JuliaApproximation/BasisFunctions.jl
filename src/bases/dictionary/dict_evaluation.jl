# This file contains routine for bounds checking and the interface
# for the evaluation of dictionary elements.

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
checkbounds(::Type{Bool}, dict::Dictionary, i::Int) = checkindex(Bool, Base.OneTo(length(dict)), i)

# We also convert some native indices to linear indices, before moving on.
# (This is more difficult to do later on, e.g. in checkindex, because that routine
#  does not have access to the dict anymore)
checkbounds(::Type{Bool}, dict::Dictionary, i::NativeIndex) =
    checkbounds(Bool, dict, linear_index(dict, i))
checkbounds(::Type{Bool}, dict::Dictionary, i::AbstractShiftedIndex) =
    checkbounds(Bool, dict, linear_index(dict, i))
checkbounds(::Type{Bool}, dict::Dictionary, i::MultilinearIndex) =
    checkbounds(Bool, dict, linear_index(dict, i))
checkbounds(::Type{Bool}, dict::Dictionary, idx::NTuple{N,Int}) where {N} =
    checkbounds(Bool, dict, CartesianIndex(idx))


# And here we call checkbounds_indices with indices(dict)
@inline checkbounds(::Type{Bool}, dict::Dictionary, I...) = checkbounds_indices(Bool, axes(dict), I)

"Return the support of the idx-th basis function. Default is support of the dictionary."
function dict_support(dict, idx)
    checkdictionary(dict)
    checkbounds(dict, idx)
    support(dict)
end
"Does the given point lie inside the support of the given function or dictionary?"
in_support(dict::Dictionary, x) = dict_in_support(dict, x)
in_support(dict::Dictionary, idx, x) = dict_in_support(dict, native_index(dict, idx), x)

# Concrete types can override this if there is a more efficient implementation
dict_in_support(dict, x) = default_in_support(dict, x)
dict_in_support(dict, idx, x) = default_in_support(dict, idx, x)

# by default we invoke the support of the dictionary
default_in_support(dict, x) = approx_in(x, support(dict), tolerance(dict))
default_in_support(dict, idx, x) = approx_in(x, dict_support(dict, idx), tolerance(dict))




#################################################
## Evaluating dictionary elements and expansions
#################################################

"Is the grid compatible with the dictionary for efficient use?"
iscompatiblegrid(d::Dictionary, g::AbstractArray) = false

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
function eval_element(dict::Dictionary, idx::Int, x)
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


"""
This function is exactly like `eval_element`, but it evaluates the derivative
of the element instead.
"""
eval_element_derivative(dict::Dictionary, idx, x) =
    eval_element_derivative(dict, idx, x, difforder(dict))

function eval_element_derivative(dict::Dictionary, idx, x, order)
    if orderiszero(order)
        eval_element(dict, idx, x)
    else
        idxn = native_index(dict, idx)
        @boundscheck checkbounds(dict, idxn)
        unsafe_eval_element_derivative1(dict, idxn, x, order)
    end
end

function eval_element_derivative(dict::Dictionary, idx::Int, x, order)
    if orderiszero(order)
        eval_element(dict, idx, x)
    else
        @boundscheck checkbounds(dict, idx)
        unsafe_eval_element_derivative1(dict, native_index(dict, idx), x, order)
    end
end

function eval_element_extension_derivative(dict::Dictionary, idx, x, order)
    if orderiszero(order)
        eval_element_extension(dict, idx, x)
    else
        @boundscheck checkbounds(dict, idx)
        unsafe_eval_element_derivative(dict, native_index(dict, idx), x, order)
    end
end

function unsafe_eval_element_derivative1(dict::Dictionary, idx, x, order)
    in_support(dict, idx, x) ? unsafe_eval_element_derivative(dict, idx, x, order) : zero(codomaintype(dict))
end

# Fallback, may throw StackOverFlowError if the method is not defined for native indices
unsafe_eval_element_derivative(dict::Dictionary, idx, x, order) =
    unsafe_eval_element_derivative(dict, native_index(dict, idx), x, order)

eval_gradient(dict::Dictionary, idx, x) = _eval_gradient(dict, idx, x, domaintype(dict))

_eval_gradient(dict, idx, x, ::Type{<:Number}) = eval_element_derivative(dict, idx, x, 1)

function _eval_gradient(dict, idx, x, ::Type{SVector{N,T}}) where {N,T}
    @boundscheck checkbounds(dict, idx)
    if in_support(dict, x)
        SVector{N,codomaintype(dict)}((unsafe_eval_element_derivative(dict, idx, x, dimension_tuple(Val{N}(), dim)) for dim in 1:N)...)
    else
        zero(SVector{N,codomaintype(dict)})
    end
end

"""
Evaluate an expansion given by the set of coefficients in the point x.
"""
function eval_expansion(dict::Dictionary, coefficients, x; options...)
    @assert size(coefficients) == size(dict)
    in_support(dict, x) ? unsafe_eval_expansion(dict, coefficients, x) : zero(span_codomaintype(dict, coefficients))
end

unsafe_eval_expansion(dict::Dictionary, coefficients, x) =
    default_unsafe_eval_expansion(dict, coefficients, x)

default_eval_expansion(dict::Dictionary, coefficients, x) =
    sum(coefficients[idx]*val for (idx,val) in pointvalues(dict, x))

default_unsafe_eval_expansion(dict::Dictionary, coefficients, x) =
    sum(coefficients[idx]*val for (idx,val) in unsafe_pointvalues(dict, x))

function eval_expansion_derivative(dict::Dictionary, coefficients, x, order; options...)
    @assert size(coefficients) == size(dict)
    in_support(dict, x) ? unsafe_eval_expansion_derivative(dict, coefficients, x, order) : zero(span_codomaintype(dict, coefficients))
end

unsafe_eval_expansion_derivative(dict::Dictionary, coefficients, x, order) =
    default_unsafe_eval_expansion_derivative(dict, coefficients, x, order)

default_eval_expansion_derivative(dict::Dictionary, coefficients, x, order) =
    sum(coefficients[idx]*eval_element_derivative(dict, idx, x, order) for idx in eachindex(dict))

default_unsafe_eval_expansion_derivative(dict::Dictionary, coefficients, x, order) =
    sum(coefficients[idx]*unsafe_eval_element_derivative(dict, idx, x, order) for idx in eachindex(dict))


# Evaluation of a dictionary means evaluation of all elements.
(dict::Dictionary)(x) = dict_eval(dict, x)

function dict_eval(dict::Dictionary, x)
    result = zeros(dict)
    dict_eval!(result, dict, x)
end

function dict_eval!(result, dict, x)
    for (idx,idxn) in enumerate(ordering(dict))
        result[idx] = unsafe_eval_element1(dict, idxn, x)
    end
    result
end

# unsafe because we don't check for x to be in the support of the dictionary
function unsafe_dict_eval(dict::Dictionary, x)
    result = zeros(dict)
    unsafe_dict_eval!(result, dict, x)
end

function unsafe_dict_eval!(result, dict, x)
    for (idx, z) in unsafe_pointvalues(dict, x)
        result[idx] = z
    end
    result
end



"""
Supertype of iterators over the pointwise evaluation of a dictionary.

The value iterator depends on a point `x` in the support of the dictionary, and
iterates over all the function values of the elements of the dictionary in that
point. Value iterators can be useful if the computation of all elements is required
and computing them in one go is more efficient than computing them one by one
independently.

For example, orthogonal polynomial iterators may implement the three-term recurrence
relation. The command
'[val for val in pointvalues(ops, x)]'
may be more efficient than
'[eval_element(dict, i, x) for i in eachindex(dict)]'
"""
abstract type DictionaryValueIterator{T} end

length(iter::DictionaryValueIterator) = length(dictionary(iter))
size(iter::DictionaryValueIterator) = size(dictionary(iter))
eltype(iter::DictionaryValueIterator{T}) where {T} = T

dictionary(iter::DictionaryValueIterator) = iter.dict
point(iter::DictionaryValueIterator) = iter.x

"Iterate over the values of a dictionary at a point."
struct GenericDictValueIterator{S,D<:Dictionary,I,T} <: DictionaryValueIterator{T}
    dict    ::  D
    x       ::  S
    idxiter ::  I
end

GenericDictValueIterator(Φ::Dictionary, x) = GenericDictValueIterator(Φ, x, eachindex(Φ))
GenericDictValueIterator(Φ::Dictionary, x, I) =
    GenericDictValueIterator{typeof(x),typeof(Φ),typeof(I),codomaintype(Φ)}(Φ, x, I)

function pointvalues(Φ::Dictionary, x)
    # We discourage the use of the iterator for points outside the support,
    # so that the iterator itself can invoke unsafe_eval_element.
    @assert in_support(Φ, x)
    GenericDictValueIterator(Φ, x)
end

# omit the check on the point being inside the domain
unsafe_pointvalues(Φ::Dictionary, x) = GenericDictValueIterator(Φ, x)

indexiterator(iter::GenericDictValueIterator) = iter.idxiter

function iterate(iter::GenericDictValueIterator)
    state = iterate(indexiterator(iter))
    if state != nothing
        idx, idx_state = state
        (idx, BasisFunctions.unsafe_eval_element(dictionary(iter), idx, point(iter))), idx_state
    else
        nothing
    end
end

function iterate(iter::GenericDictValueIterator, state)
    newstate = iterate(indexiterator(iter), state)
    if newstate != nothing
        idx, idx_state = newstate
        (idx, BasisFunctions.unsafe_eval_element(dictionary(iter), idx, point(iter))), idx_state
    end
end
