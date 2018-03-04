# dictionary.jl


######################
# Type hierarchy
######################

"""
A `Dictionary{S,T}` is an ordered list of functions, in which each function
maps a variable of type `S` to a variable of type `T`.

A `Dictionary{S,T}` has domain type `S` and codomain type `T`. The domain type
corresponds to the type of a domain in the `Domains.jl` package, and it is the
type of the expected argument to the elements of the function set. The
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

"The default type of the expansion coefficients in a dictionary."
# By default we set it equal to the codomaintype
coefficient_type(dict::Dictionary) = codomaintype(dict)

# The dimension of a function set is the dimension of its domain type
dimension(dict::Dictionary) = dimension(domaintype(dict))

dimension(dict::Dictionary, i) = dimension(element(dict, i))

"Are the functions in the dictionary are real-valued?"
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

promote_domaintype(dict1::Dictionary{S,T}, dict2::Dictionary{S,T}) where {S,T} = (dict1,dict2)

function promote_domaintype(dict1::Dictionary{S1,T1}, dict2::Dictionary{S2,T2}) where {S1,S2,T1,T2}
    S = promote_type(S1,S2)
    promote_domaintype(dict1, S), promote_domaintype(dict2, S)
end

function promote_eltype(dict::Dictionary, args...)
    error("Calling promote_eltype on a dictionary is deprecated.")
end

"Promote the domain sub type of the dictionary."
promote_domainsubtype(d::Dictionary{S,T}, ::Type{U}) where {S<:Number,T,U<:Number} = promote_domaintype(d, U)

promote_domainsubtype(dict::Dictionary{SVector{N,S},T}, ::Type{S}) where {N,S<:Number,T} = dict
promote_domainsubtype(dict::Dictionary{SVector{N,S},T}, ::Type{U}) where {N,S<:Number,T,U} =
    promote_domaintype(dict, SVector{N,U})

promote_domainsubtype(dict::Dictionary{NTuple{N,S},T}, ::Type{S}) where {N,S<:Number,T} = dict
promote_domainsubtype(dict::Dictionary{NTuple{N,S},T}, ::Type{U}) where {N,S<:Number,T,U} =
    promote_domaintype(dict, NTuple{N,U})

widen(d::Dictionary) = promote_domaintype(d, widen(domaintype(d)))

# similar returns a similar basis of a given size and numeric type
# It can be implemented in terms of resize and promote_domaintype.
similar(d::Dictionary, ::Type{T}, n) where {T} = resize(promote_domaintype(d, T), n)

# Support resize of a 1D set with a tuple of a single element, so that one can
# write statements of the form resize(s, size(some_set)) in all dimensions.
resize(s::Dictionary1d, n::NTuple{1,Int}) = resize(s, n[1])



"Return a set of zero coefficients in the native format of the set."
zeros(s::Dictionary) = zeros(coefficient_type(s), s)

# By default we assume that the native format corresponds to an array of the
# same size as the set. This is not true, e.g., for multidicts.
zeros(::Type{T}, s::Dictionary) where {T} = zeros(T, size(s))



###########
# Indexing
###########


# A native index has to be distinguishable from linear indices by type. A linear
# index is an int. If a native index also has an integer type, then its value
# should be wrapped in a different type. That is the purpose of NativeIndex.
# Concrete types with a meaningful name can inherit from this abstract type.
# If the native index is not an integer, then no wrapping is necessary.
abstract type NativeIndex end

# We assume that the index is stored in the 'index' field
index(idxn::NativeIndex) = idxn.index

length(idxn::NativeIndex) = 1

getindex(idxn::NativeIndex, i) = (assert(i==1); index(idxn))

"Compute the native index corresponding to the given linear index."
native_index(d::Dictionary, idx::Int) = idx
# By default, the native index of a set is its linear index.
# The given idx argument should always be an Int for conversion from linear indices.
# Subtypes may add definitions for other types, such as multilinear indices.
# Typing idx to be Int here narrows the scope of false definitions for subsets.
# The downside is we have to write idx::Int everywhere else too in order to avoid ambiguities.

"Compute the linear index corresponding to the given native index."
linear_index(D::Dictionary, idxn) = idxn::Int
# We don't specify an argument type to idxn here, even though we would only want
# this default to apply to Int's. Instead, we add an assertion to the result.
# Adding Int to idxn causes lots of ambiguities. Not adding the assertion either leads to
# potentially wrong definitions, since in that case linear_index doesn't do anything.
# This can lead to StackOverflowError's in other parts of the code that expect
# the type of idxn to change to Int after calling linear_index.

"""
Convert the set of coefficients in the native format of the dictionary
to a linear list. The order of the coefficients in this list is determined by
the order of the elements in the dictionary.
"""
# Allocate memory for the linear set and call linearize_coefficients! to do the work
function linearize_coefficients(dict::Dictionary, coef_native)
    coef_linear = zeros(eltype(coef_native), length(dict))
    linearize_coefficients!(dict, coef_linear, coef_native)
end

linearize_coefficients!(dict::Dictionary, coef_linear::Vector, coef_native) =
    copy!(coef_linear, coef_native)

"""
Convert a linear set of coefficients back to the native representation of the dictionary.
"""
function delinearize_coefficients(dict::Dictionary, coef_linear::AbstractVector{T}) where {T}
    coef_native = zeros(eltype(coef_linear), dict)
    delinearize_coefficients!(dict, coef_native, coef_linear)
end

delinearize_coefficients!(dict::Dictionary, coef_native, coef_linear::Vector) =
    copy!(coef_native, coef_linear)

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

# Default set of linear indices: from 1 to length(s)
# Default algorithms assume this indexing for the basis functions, and the same
# linear indexing for the set of coefficients.
# The indices may also have tensor-product structure, for tensor product sets.
eachindex(d::Dictionary) = 1:length(d)



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

"Does the set have a transform associated with some space?"
has_transform(d1::Dictionary, d2) = false

"Does the set have a transform associated with some space that is unitary"
has_unitary_transform(d::Dictionary) = has_transform(d)
# If a dict has a transform, we assume it is unitary. If it is not,
# this function has to be over written.

# Convenience functions: default grid, and conversion from grid to space
has_transform(d::Dictionary) = has_grid(d) && has_transform(d, grid(d))
has_transform(d::Dictionary, grid::AbstractGrid) = has_transform(d, gridbasis(grid))

"Does the set support extension and restriction operators?"
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


tolerance(dict::Dictionary) = tolerance(domaintype(dict))

# Provide this implementation which Base does not include anymore
# TODO: hook into the Julia checkbounds system, once such a thing is developed.
checkbounds(i::Int, j::Int) = (1 <= j <= i) ? nothing : throw(BoundsError())

# General bounds check: the linear index has to be in the range 1:length(s)
checkbounds(d::Dictionary, i::Int) = checkbounds(length(d), i)

# Fail-safe backup when i is not an integer: convert to linear index (which is an integer)
checkbounds(d::Dictionary, i) = checkbounds(d, linear_index(d, i))

"Return the support of the idx-th basis function."
support(d::Dictionary1d, idx) = (left(d,idx), right(d,idx))

"Does the given point lie inside the support of the given set function?"
in_support(dict::Dictionary1d, idx, x) = left(dict, idx)-tolerance(dict) <= x <= right(dict, idx)+tolerance(dict)

# isless doesn't work when comparing complex numbers. It may happen that a real
# function set uses a complex element type, or that the user evaluates at a
# complex point.  Our default is to check for a zero imaginary part, and then
# convert x to a real number. Genuinely complex function sets should override.
in_support{T <: Complex}(dict::Dictionary1d, idx, x::T) =
    abs(imag(x)) <= tolerance(dict) && in_support(dict, idx, real(x))


##############################################
## Evaluating set elements and expansions
##############################################

"""
You can evaluate a member function of a set using the eval_set_element routine.
It takes as arguments the function set, the index of the member function and
the point in which to evaluate.

This function performs bounds checking on the index and also checks whether the
point x lies inside the support of the function. A BoundsError() is thrown for
an index out of bounds. By default, the value 0 is returned when x is outside
the support. This value can be changed with an optional extra argument.

After the checks, this routine calls eval_element on the concrete set.
"""
function eval_set_element(dict::Dictionary, idx, x, outside_value = zero(codomaintype(dict)); extend=false)
    checkbounds(dict, idx)
    extend || in_support(dict, idx, x) ? eval_element(dict, idx, x) : outside_value
end

# We use a special routine for evaluation on a grid, since we can hoist the boundscheck.
# We pass on any extra arguments to eval_set_element!, hence the outside_val... argument here
function eval_set_element(dict::Dictionary, idx, grid::AbstractGrid, outside_value...)
    result = zeros(gridspace(grid, codomaintype(dict)))
    eval_set_element!(result, dict, idx, grid, outside_value...)
end

function eval_set_element!(result, dict::Dictionary, idx, grid::AbstractGrid, outside_value = zero(codomaintype(dict)))
    @assert size(result) == size(grid)
    checkbounds(dict, idx)

    @inbounds for k in eachindex(grid)
        result[k] = eval_set_element(dict, idx, grid[k], outside_value)
    end
    result
end


"""
This function is exactly like `eval_set_element`, but it evaluates the derivative
of the element instead.
"""
function eval_set_element_derivative(dict::Dictionary, idx, x, outside_value = zero(codomaintype(dict)); extend=false)
    checkbounds(dict, idx)
    extend || in_support(dict, idx, x) ? eval_element_derivative(dict, idx, x) : outside_value
end

function eval_set_element_derivative(dict::Dictionary, idx, grid::AbstractGrid, outside_value...)
    result = zeros(gridspace(grid, codomaintype(dict)))
    eval_set_element_derivative!(result, dict, idx, grid, outside_value...)
end

function eval_set_element_derivative!(result, dict::Dictionary, idx, grid::AbstractGrid, outside_value = zero(codomaintype(dict)))
    @assert size(result) == size(grid)
    checkbounds(dict, idx)

    @inbounds for k in eachindex(grid)
        result[k] = eval_set_element_derivative(dict, idx, grid[k], outside_value)
    end
    result
end


"""
Evaluate an expansion given by the set of coefficients `coefficients` in the point x.
"""
function eval_expansion(dict::Dictionary, coefficients, x; options...)
    T = span_codomaintype(dict, coefficients)
    z = zero(T)

    # It is safer below to use eval_set_element than eval_element, because of
    # the check on the support. We elide the boundscheck with @inbounds (perhaps).
    @inbounds for idx in eachindex(dict)
        z = z + coefficients[idx] * eval_set_element(dict, idx, x; options...)
    end
    z
end

function eval_expansion(dict::Dictionary, coefficients, grid::AbstractGrid)
    @assert dimension(dict) == dimension(grid)
    @assert size(coefficients) == size(dict)
    # TODO: reenable test once product grids and product sets have compatible types again
    # @assert eltype(grid) == domaintype(set)

    span = Span(dict, eltype(coefficients))
    T = codomaintype(span)
    E = evaluation_operator(span, gridspace(grid, T))
    E * coefficients
end

# There is no need for an eval_expansion! method, since one can use evaluation_operator for that purpose


#######################
## Application support
#######################

"""
Compute the moment of the given basisfunction, i.e. the integral on its
support.
"""
# Default to numerical integration
moment(d::Dictionary1d, idx) = quadgk(d[idx], left(d), right(d))[1]
