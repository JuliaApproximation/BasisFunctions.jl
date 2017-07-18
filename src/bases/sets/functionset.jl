# functionset.jl


######################
# Type hierarchy
######################

"""
A `FunctionSet` is any set of functions with a finite size. It is typically the
truncation of an infinite set, but that need not be the case.

A `FunctionSet{T}` has domain type `T`. This type corresponds to the type of a
domain in the `Domains.jl` package, and it is the type of the expected argument
to the elements of the function set.

Each function set is ordered. There is a one-to-one map between the integers
1:length(s) and the elements of the set. This map defines the order of
coefficients in a vector that represents a function expansion in the set.

A `FunctionSet` has two types of indexing: native indexing and linear indexing.
Linear indexing is used to order elements of the set into a vector, as explained
above. Native indices are closer to the mathematical definitions of the basis
functions. For example, a tensor product set consisting of M functions in the
first dimension times N functions in the second dimension may have a native index
(i,j), with 1 <= i <= M and 1 <= j <= N. The native representation of an expansion
in this set is a matrix of size M x N. In contrast, the linear representation is
a large vector of length MN.
Another example is given by orthogonal polynomials: they are typically indexed
by their degree. Hence, their native index ranges from 0 to N-1, but their linear
index ranges from 1 to N.

Computations in this package are typically performed using native indexing where
possible. Linear indexing is used to convert representations into a form suitable
for linear algebra: expansions turn into vectors, and linear operators turn into
matrices.
"""
abstract type FunctionSet{T}
end


# Useful abstraction for special cases
const FunctionSet1d{T <: Number} = FunctionSet{T}
# Warning: not all 2d function sets have SVector{2,T} type, they could have (S,T) type
const FunctionSet2d{T} = FunctionSet{SVector{2,T}}
const FunctionSet3d{T} = FunctionSet{SVector{3,T}}
const FunctionSet4d{T} = FunctionSet{SVector{4,T}}


"The type of the elements of the domain of the set."
domaintype(::Type{FunctionSet{T}}) where {T} = T
domaintype(::Type{S}) where {S <: FunctionSet} = domaintype(supertype(S))
domaintype(set::FunctionSet) = domaintype(typeof(set))

"The type of the elements of the codomain of the set."
rangetype(::Type{S}) where {S <: FunctionSet} = domaintype(S)
rangetype(set::FunctionSet) = rangetype(typeof(set))

"The default type of the expansion coefficients in a function set."
coefficient_type(::Type{S}) where {S <: FunctionSet} = rangetype(S)
coefficient_type(set::FunctionSet) = coefficient_type(typeof(set))

function eltype(set::FunctionSet)
    error("Error: calling eltype on a function set is deprecated. Use a span instead.")
end

function eltype(set1::FunctionSet, set2::FunctionSet)
    promote_type(eltype(set1), eltype(set2))
end

# One can talk about dimensions for a set in Euclidean space
ndims(::FunctionSet1d) = 1
ndims(set::FunctionSet{SVector{N,T}}) where {N,T} = N

"Property to indicate whether the functions in the set are real-valued (for real arguments)."
isreal(s::FunctionSet) = isreal(rangetype(s))



# Is a given set a basis? In general, it is not. But it could be.
# Hence, we need a property for it:
is_basis(s::FunctionSet) = false

# Any basis is a frame
is_frame(s::FunctionSet) = is_basis(s)


"Property to indicate whether a basis is orthogonal."
is_orthogonal(s::FunctionSet) = false

"Property to indicate whether a basis is orthonormal"
is_orthonormal(s::FunctionSet) = false

"Property to indicate whether a basis is biorthogonal (or a Riesz basis)."
is_biorthogonal(s::FunctionSet) = is_orthogonal(s)

"Return the size of the set."
size(s::FunctionSet) = (length(s),)

"Return the size of the j-th dimension of the set (if applicable)."
size(s::FunctionSet, j) = j==1 ? length(s) : throw(BoundsError())

endof(s::FunctionSet) = length(s)

"Is the function set composite, i.e. does it consist of several components?"
is_composite(s::FunctionSet) = false

"""
The instantiate function takes a set type, size and numeric type as argument, and
returns an instance of the type with the given size and numeric type and using
default values for other parameters. This means the given type is usually abstract,
since it is given without parameters.

This function is mainly used to create instances for testing purposes.
"""
instantiate{S <: FunctionSet}(::Type{S}, n) = instantiate(S, n, Float64)

"Promote the domain type of the function set."
promote_domaintype(set::FunctionSet{T}, ::Type{T}) where {T} = set
promote_domaintype(set::FunctionSet{T}, ::Type{S}) where {T,S} = set_promote_domaintype(set, S)

promote_domaintype(set1::FunctionSet{T}, set2::FunctionSet{T}) where {T} = (set1,set2)

function promote_domaintype(set1::FunctionSet{T}, set2::FunctionSet{S}) where {T,S}
    U = promote_type(T,S)
    promote_domaintype(set1, U), promote_domaintype(set2, U)
end

function promote_eltype(set::FunctionSet, args...)
    error("Calling promote_eltype on a function set is deprecated.")
end

widen(s::FunctionSet) = promote_domaintype(s, widen(domaintype(s)))

# similar returns a similar basis of a given size and numeric type
# It can be implemented in terms of resize and promote_domaintype.
similar(s::FunctionSet, ::Type{T}, n) where {T} = resize(promote_domaintype(s, T), n)

# Support resize of a 1D set with a tuple of a single element, so that one can
# write statements of the form resize(s, size(some_set)) in all dimensions.
resize(s::FunctionSet1d, n::NTuple{1,Int}) = resize(s, n[1])


"Return a set of zero coefficients in the native format of the set."
zeros(s::FunctionSet) = zeros(coefficient_type(s), s)

# By default we assume that the native format corresponds to an array of the
# same size as the set. This is not true, e.g., for multisets.
zeros(::Type{T}, s::FunctionSet) where {T} = zeros(T, size(s))



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
native_index(s::FunctionSet, idx::Int) = idx
# By default, the native index of a set is its linear index.
# The given idx argument should always be an Int for conversion from linear indices.
# Subtypes may add definitions for other types, such as multilinear indices.
# Typing idx to be Int here narrows the scope of false definitions for subsets.
# The downside is we have to write idx::Int everywhere else too in order to avoid ambiguities.

"Compute the linear index corresponding to the given native index."
linear_index(s::FunctionSet, idxn) = idxn::Int
# We don't specify an argument type to idxn here, even though we would only want
# this default to apply to Int's. Instead, we add an assertion to the result.
# Adding Int to idxn causes lots of ambiguities. Not adding the assertion either leads to
# potentially wrong definitions, since in that case linear_index doesn't do anything.
# This can lead to StackOverflowError's in other parts of the code that expect
# the type of idxn to change to Int after calling linear_index.

"""
Convert the set of coefficients in the native format of the set to a linear list.
The order of the coefficients in this list is determined by the order of the
elements in the set.
"""
# Allocate memory for the linear set and call linearize_coefficients! to do the work
function linearize_coefficients(set::FunctionSet, coef_native)
    coef_linear = zeros(eltype(coef_native), length(set))
    linearize_coefficients!(set, coef_linear, coef_native)
end

linearize_coefficients!(set::FunctionSet, coef_linear::Vector, coef_native) =
    copy!(coef_linear, coef_native)

"""
Convert a linear set of coefficients back to the native representation of the set.
"""
function delinearize_coefficients(set::FunctionSet, coef_linear::AbstractVector{T}) where {T}
    coef_native = zeros(eltype(coef_linear), set)
    delinearize_coefficients!(set, coef_native, coef_linear)
end

delinearize_coefficients!(set::FunctionSet, coef_native, coef_linear::Vector) =
    copy!(coef_native, coef_linear)

# Sets have a native size and a linear size. However, there is not necessarily a
# bijection between the two. You can always convert a native size to a linear size,
# but the other direction can be done in general only approximately.
# For example, a 2D tensor product set can only support sizes of the form n1 * n2. Its native size may be
# (n1,n2) and its linear size n1*n2, but not any integer n maps to a native size tuple.
# By convention, we denote a native size variable by size_n.
"Compute the native size best corresponding to the given linear size."
approximate_native_size(s::FunctionSet, size_l) = size_l

"Compute the linear size corresponding to the given native size."
linear_size(s::FunctionSet, size_n) = size_n

"Suggest a suitable size, close to n, to resize the given function set."
approx_length(s::FunctionSet, n::Int) = n
approx_length(s::FunctionSet, n::Real) = approx_length(s, round(Int,n))

# Default set of linear indices: from 1 to length(s)
# Default algorithms assume this indexing for the basis functions, and the same
# linear indexing for the set of coefficients.
# The indices may also have tensor-product structure, for tensor product sets.
eachindex(s::FunctionSet) = 1:length(s)



###############################
## Properties of function sets
###############################

# The following properties are not implemented as traits with types, because they are
# not intended to be used in a time-critical path of the code.

"Does the set implement a derivative?"
has_derivative(s::FunctionSet) = false

"Does the set implement an antiderivative?"
has_antiderivative(s::FunctionSet) = false

"Does the set have an associated interpolation grid?"
has_grid(s::FunctionSet) = false

"Does the set have a transform associated with some space?"
has_transform(s1::FunctionSet, s2) = false

"Does the set have a transform associated with some space that is unitary"
has_unitary_transform(s::FunctionSet) = has_transform(s)
# If a set has a transform, we assume it is unitary. If it is not, this function has to be over written.

# Convenience functions: default grid, and conversion from grid to space
has_transform(s::FunctionSet) = has_grid(s) && has_transform(s, grid(s))
has_transform(s::FunctionSet, grid::AbstractGrid) = has_transform(s, DiscreteGridSpace(grid, rangetype(s)))

"Does the set support extension and restriction operators?"
has_extension(s::FunctionSet) = false

# A concrete FunctionSet has spaces associated with derivatives or antiderivatives of a certain order,
# and it should implement the following introspective functions:
# derivative_set(s::MyFunctionSet, order) = ...
# antiderivative_set(s::MyFunctionSet, order) = ...
# where order is either an Int (in 1D) or a tuple of Int's (in higher dimensions).

# The default order is 1 for 1d sets:
derivative_set(s::FunctionSet1d) = derivative_set(s, 1)
antiderivative_set(s::FunctionSet1d) = antiderivative_set(s, 1)

# Catch tuples with just one element and convert to Int
derivative_set(s::FunctionSet1d, order::Tuple{Int}) = derivative_set(s, order[1])
antiderivative_set(s::FunctionSet1d, order::Tuple{Int}) = antiderivative_set(s, order[1])

# This is a candidate for a better implementation. How does one generate a
# unit vector in a tuple?
# ASK is this indeed a better implementation?
dimension_tuple(n, dim) = ntuple(k -> (k==dim? 1: 0), n)

# Convenience function to differentiate in a given dimension
derivative_set(s::FunctionSet; dim=1) = derivative_set(s, dimension_tuple(ndims(s), dim))
antiderivative_set(s::FunctionSet; dim=1) = antiderivative_set(s, dimension_tuple(ndims(s), dim))

# A concrete FunctionSet may also override extension_set and restriction_set
# The default is simply to resize.
extension_set(s::FunctionSet, n) = resize(s, n)
restriction_set(s::FunctionSet, n) = resize(s, n)

#######################
## Iterating over sets
#######################

# Default iterator over sets of functions: based on underlying index iterator.
function start(s::FunctionSet)
    iter = eachindex(s)
    (iter, start(iter))
end

function next(s::FunctionSet, state)
    iter, iter_state = state
    idx, iter_newstate = next(iter,iter_state)
    (s[idx], (iter,iter_newstate))
end

done(s::FunctionSet, state) = done(state[1], state[2])


tolerance(set::FunctionSet) = tolerance(domaintype(set))

# Provide this implementation which Base does not include anymore
# TODO: hook into the Julia checkbounds system, once such a thing is developed.
checkbounds(i::Int, j::Int) = (1 <= j <= i) ? nothing : throw(BoundsError())

# General bounds check: the linear index has to be in the range 1:length(s)
checkbounds(s::FunctionSet, i::Int) = checkbounds(length(s), i)

# Fail-safe backup when i is not an integer: convert to linear index (which is an integer)
checkbounds(s::FunctionSet, i) = checkbounds(s, linear_index(s, i))

"Return the support of the idx-th basis function."
support(s::FunctionSet1d, idx) = (left(s,idx), right(s,idx))

"Does the given point lie inside the support of the given set function?"
in_support(set::FunctionSet1d, idx, x) = left(set, idx)-tolerance(set) <= x <= right(set, idx)+tolerance(set)

# isless doesn't work when comparing complex numbers. It may happen that a real
# function set uses a complex element type, or that the user evaluates at a
# complex point.  Our default is to check for a zero imaginary part, and then
# convert x to a real number. Genuinely complex function sets should override.
in_support{T <: Complex}(set::FunctionSet1d, idx, x::T) =
    abs(imag(x)) <= tolerance(set) && in_support(set, idx, real(x))


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
function eval_set_element(set::FunctionSet, idx, x, outside_value = zero(rangetype(set)))
    checkbounds(set, idx)
    in_support(set, idx, x) ? eval_element(set, idx, x) : outside_value
end

# We use a special routine for evaluation on a grid, since we can hoist the boundscheck.
# We pass on any extra arguments to eval_set_element!, hence the outside_val... argument here
function eval_set_element(set::FunctionSet, idx, grid::AbstractGrid, outside_value...)
    result = zeros(DiscreteGridSpace(grid, rangetype(set)))
    eval_set_element!(result, set, idx, grid, outside_value...)
end

function eval_set_element!(result, set::FunctionSet, idx, grid::AbstractGrid, outside_value = zero(rangetype(set)))
    @assert size(result) == size(grid)
    checkbounds(set, idx)

    @inbounds for k in eachindex(grid)
        result[k] = eval_set_element(set, idx, grid[k], outside_value)
    end
    result
end

"""
Evaluate an expansion given by the set of coefficients `coefficients` in the point x.
"""
function eval_expansion(set::FunctionSet, coefficients, x)
    T = rangetype(span(set, eltype(coefficients)))
    z = zero(T)

    # It is safer below to use eval_set_element than eval_element, because of
    # the check on the support. We elide the boundscheck with @inbounds (perhaps).
    @inbounds for idx in eachindex(set)
        z = z + coefficients[idx] * eval_set_element(set, idx, x)
    end
    z
end

function eval_expansion(set::FunctionSet, coefficients, grid::AbstractGrid)
    @assert ndims(set) == ndims(grid)
    @assert size(coefficients) == size(set)
    @assert eltype(grid) == domaintype(set)

    T = rangetype(span(set, eltype(coefficients)))
    E = evaluation_operator(set, DiscreteGridSpace(grid, T))
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
moment(s::FunctionSet1d, idx) = quadgk(s[idx], left(s), right(s))[1]
