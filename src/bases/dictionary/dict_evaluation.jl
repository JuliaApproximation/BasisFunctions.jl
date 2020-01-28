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
checkbounds(::Type{Bool}, dict::Dictionary, i::LinearIndex) = checkindex(Bool, Base.OneTo(length(dict)), i)

# We also convert some native indices to linear indices, before moving on.
# (This is more difficult to do later on, e.g. in checkindex, because that routine
#  does not have access to the dict anymore)
checkbounds(::Type{Bool}, dict::Dictionary, i::NativeIndex) =
    checkbounds(Bool, dict, linear_index(dict, i))
checkbounds(::Type{Bool}, dict::Dictionary, i::AbstractShiftedIndex) =
    checkbounds(Bool, dict, linear_index(dict, i))
checkbounds(::Type{Bool}, dict::Dictionary, i::MultilinearIndex) =
    checkbounds(Bool, dict, linear_index(dict, i))

# And here we call checkbounds_indices with indices(dict)
@inline checkbounds(::Type{Bool}, dict::Dictionary, I...) = checkbounds_indices(Bool, axes(dict), I)

"Return the support of the idx-th basis function. Default is support of the dictionary."
support(dict::Dictionary, idx) = support(dict)
# Warning: the functions above and below may be wrong for certain concrete
# dictionaries, for example for univariate functions with non-connected support.
# Make sure to override, and make sure that the overridden version is called.

tolerance(dict::Dictionary) = tolerance(codomaintype(dict))

"Does the given point lie inside the support of the given function or dictionary?"
in_support(dict::Dictionary, x) = dict_in_support(dict, x)
in_support(dict::Dictionary, idx, x) = dict_in_support(dict, idx, x)
# The mechanism is as follows:
# - in_support(dict::Dictionary, ...) calls dict_in_support
# - any linear index is converted to a native index
# - concrete dictionary should implement dict_in_support
# The reasoning is that concrete dictionaries need not all worry about handling
# linear indices. Yet, they are free to implement other types of indices.
# If a more efficient algorithm is available for linear indices, then the concrete
# dictionary can still intercept the call to in_support.
# The delegation to a method with a different name (dict_in_support) makes it
# substantially easier to deal with ambiguity errors.

# This is the standard conversion to a native_index for any index of type
# LinearIndex. This calls a different function, hence it is fine if the native
# index happens to be a linear index.
in_support(dict::Dictionary, idx::LinearIndex, x) =
    dict_in_support(dict, native_index(dict, idx), x)

# The default fallback is implemented below in terms of the support of the dictionary:
dict_in_support(dict::Dictionary, idx, x) = default_in_support(dict, idx, x)
dict_in_support(dict::Dictionary, x) = default_in_support(dict, x)

default_in_support(dict::Dictionary, idx, x) = approx_in(x, support(dict, idx), tolerance(dict))
default_in_support(dict::Dictionary, x) = approx_in(x, support(dict), tolerance(dict))




#################################################
## Evaluating dictionary elements and expansions
#################################################

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
unsafe_eval_element1(dict::Dictionary, idx, grid::AbstractGrid) =
    BasisFunctions._default_unsafe_eval_element_in_grid(dict, idx, grid)

function _default_unsafe_eval_element_in_grid(dict::Dictionary, idx, grid::AbstractGrid)
    result = zeros(GridBasis(dict, grid))
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

derivative_dict(dict::Dictionary; options...) = derivative_dict(dict, 1; options...)

"""
Evaluate an expansion given by the set of coefficients in the point x.
"""
function eval_expansion(dict::Dictionary, coefficients, x; options...)
    @assert size(coefficients) == size(dict)
    in_support(dict, x) ? unsafe_eval_expansion(dict, coefficients, x) : zero(span_codomaintype(dict, coefficients))
end

unsafe_eval_expansion(dict::Dictionary, coefficients, x) =
    default_eval_expansion(dict, coefficients, x)

function default_eval_expansion(dict::Dictionary, coefficients, x)
    T = span_codomaintype(dict, coefficients)
    z = zero(T)
    # It is safer below to use eval_element than unsafe_eval_element, because of
    # the check on the support.
    @inbounds for idx in eachindex(coefficients)
        z = z + coefficients[idx] * eval_element(dict, idx, x)
    end
    z
end

function eval_expansion(dict::Dictionary, coefficients, grid::AbstractGrid; options...)
    @assert dimension(dict) == GridArrays.dimension(grid)
    @assert size(coefficients) == size(dict)
    # TODO: reenable test once product grids and product sets have compatible types again
    # @assert eltype(grid) == domaintype(dict)

    T = coefficienttype(dict)
    E = evaluation_operator(dict, GridBasis{T}(grid); options...)
    E * coefficients
end

# Evaluation of a dictionary means evaluation of all elements.
(dict::Dictionary)(x) =
    in_support(dict, x) ? unsafe_dict_eval(dict, x) : zeros(codomaintype(dict),size(dict))

# evaluation of the dictionary is "unsafe" because the routine can assume that the
# support of x has already been checked
unsafe_dict_eval(dict::Dictionary, x) =
    unsafe_dict_eval!(zeros(dict), dict, x)

function unsafe_dict_eval!(result, dict::Dictionary, x)
    for i in eachindex(dict)
        result[i] = unsafe_eval_element(dict, i, x)
    end
    result
end
