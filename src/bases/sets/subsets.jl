# subsets.jl

checkbounds(set::FunctionSet, idx::Range) =
    (checkbounds(set, first(idx)); checkbounds(set, last(idx)))

checkbounds(set::FunctionSet, idx::CartesianRange) =
    (checkbounds(set, first(idx)); checkbounds(set, last(idx)))

function checkbounds(set::FunctionSet, idx::Array)
    for i in idx
        checkbounds(set, i)
    end
end

#######################
# An abstract subset
#######################

# We start with a generic description of subsets and imlement as much
# functionality as possible at the abstract level of Subset.
# This is followed by specialized concrete types of subsets.

"""
A Subset is an abstract subset of a function set. It is characterized by the
underlying larger set, and a collection of indices into that set.
"""
abstract type Subset{N,T} <: FunctionSet{N,T}
end

# We assume that the underlying set is stored in a field called superset
superset(s::Subset) = s.superset

# We assume that the underlying indices are stored in a field called indices
indices(s::Subset) = s.indices
indices(s::Subset, i) = s.indices[i]

# The concrete subset should implement `similar_subset`, a routine that
# returns a subset of a similar type as itself, but with a different underlying set.
set_promote_eltype{N,T}(s::Subset{N,T}, ::Type{T}) = s
set_promote_eltype{N,T,S}(s::Subset{N,T}, ::Type{S}) =
    similar_subset(s, promote_eltype(superset(s), S), indices(s))

apply_map(s::Subset, map) = similar_subset(s, apply_map(superset(s), map), indices(s))


name(s::Subset) = "Subset of " * name(superset(s)) * " with indices " * string(indices(s))

length(s::Subset) = length(indices(s))

size(s::Subset) = size(indices(s))

eachindex(s::Subset) = eachindex(indices(s))

##
# For various properties it is not a priori clear whether a subset has them or not,
# even if the underlying set does. For each function foo, we invoke a new function
# subset_foo, that has the underlying set and the range of indices as additional
# arguments. This function call can be intercepted and implemented for specific kinds
# of subsets, if it is known that foo applies.

resize(s::Subset, n) = subset_resize(s, n, superset(s), indices(s))

has_grid(s::Subset) = subset_has_grid(s, superset(s), indices(s))
has_derivative(s::Subset) = subset_has_derivative(s, superset(s), indices(s))
has_antiderivative(s::Subset) = subset_has_antiderivative(s, superset(s), indices(s))
has_transform(s::Subset) = subset_has_transform(s, superset(s), indices(s))

derivative_set(s::Subset, order; options...) = subset_derivative_set(s, order, superset(s), indices(s); options...)
antiderivative_set(s::Subset, order; options...) = subset_antiderivative_set(s, order, superset(s), indices(s); options...)

grid(s::Subset) = subset_grid(s, superset(s), indices(s))


# By default, we have none of the properties
subset_has_derivative(s::Subset, superset, indices) = false
subset_has_antiderivative(s::Subset, superset, indices) = false
subset_has_transform(s::Subset, superset, indices) = false
subset_has_grid(s::Subset, superset, indices) = false




for op in (:isreal, :is_orthogonal, :is_basis)
    @eval $op(set::Subset) = $op(superset(set))
end


native_index(s::Subset, idx::Int) = idx
linear_index(s::Subset, idxn::Int) = idxn

for op in [:left, :right]
    @eval $op(s::Subset) = $op(superset(s))
end

for op in [:left, :right, :moment, :norm]
    @eval $op(s::Subset, i) = $op(superset(s), indices(s, i))
end

in_support(s::Subset, i, x) = in_support(superset(s), indices(s, i), x)

eval_element(s::Subset, i, x) = eval_element(superset(s), indices(s, i), x)




#####################
# Concrete subsets
#####################


"""
A LargeSubset is a large subset of a function set. Operators associated with the
subset are implemented in terms of corresponding operators on the underlying set.
This often leads to an explicit extension to the full set, but it can take
advantage of possible fast implementations for the underlying set. For large subsets
this is more efficient than iterating over the individual elements.
"""
struct LargeSubset{SET, IDX, N, T} <: Subset{N,T}
    superset    ::  SET
    indices     ::  IDX

    function LargeSubset{SET,IDX,N,T}(set::FunctionSet{N,T}, idx::IDX) where {SET,IDX,N,T}
        checkbounds(set, idx)
        new(set, idx)
    end
end

LargeSubset{N,T}(set::FunctionSet{N,T}, indices) =
    LargeSubset{typeof(set),typeof(indices),N,T}(set, indices)

similar_subset(s::LargeSubset, set, indices) = LargeSubset(set, indices)


grid(s::LargeSubset) = grid(superset(s))

function extension_operator(s1::LargeSubset, s2::FunctionSet; options...)
    @assert s2 == superset(s1)
    IndexExtensionOperator(s1, s2, indices(s1))
end

function restriction_operator(s1::FunctionSet, s2::LargeSubset; options...)
    @assert s1 == superset(s2)
    IndexRestrictionOperator(s1, s2, indices(s2))
end

# In general, the derivative set of a subset can be the whole derivative set
# of the underlying set. We can not know generically whether the derivative set
# can be indexed as well. Same for antiderivative.
# Yet, we can generically define a differentiation_operator by extending the subset
# to the whole set and then invoking the differentiation operator of the latter,
# and we choose that to be the default.
subset_has_derivative(s::LargeSubset, superset, indices) = has_derivative(superset)
subset_has_antiderivative(s::LargeSubset, superset, indices) = has_antiderivative(superset)

subset_derivative_set(s::LargeSubset, order, superset, indices; options...) =
    derivative_set(superset, order; options...)
subset_antiderivative_set(s::LargeSubset, order, superset, indices; options...) =
    antiderivative_set(superset, order; options...)

function differentiation_operator(s1::LargeSubset, s2::FunctionSet, order::Int; options...)
    @assert s2 == derivative_set(s1, order)
    D = differentiation_operator(superset(s1), s2, order; options...)
    E = extension_operator(s1, superset(s1); options...)
    D*E
end

function antidifferentiation_operator(s1::LargeSubset, s2::FunctionSet, order::Int; options...)
    @assert s2 == antiderivative_set(s1, order)
    D = antidifferentiation_operator(superset(s1), s2, order; options...)
    E = extension_operator(s1, superset(s1); options...)
    D*E
end




"""
A SmallSubset is a subset of a function set with a small number of indices.
The difference with a regular function subset is that operators on a small set
are implemented by iterating explicitly over the indices, and not in terms of
an operator on the full underlying set.
"""
struct SmallSubset{SET,IDX,N,T} <: Subset{N,T}
    superset    ::  SET
    indices     ::  IDX

    function SmallSubset{SET,IDX,N,T}(set::FunctionSet{N,T}, indices::IDX) where {SET,IDX,N,T}
        checkbounds(set, indices)
        new(set, indices)
    end
end

SmallSubset{N,T}(set::FunctionSet{N,T}, indices) =
    SmallSubset{typeof(set),typeof(indices),N,T}(set, indices)


similar_subset(s::SmallSubset, set, indices) = SmallSubset(set, indices)


"""
A SingletonSubset represent a single element from an underlying set.
"""
struct SingletonSubset{SET,IDX,N,T} <: Subset{N,T}
    superset    ::  SET
    index       ::  IDX

    function SingletonSubset{SET,IDX,N,T}(set::FunctionSet{N,T}, index::IDX) where {SET,IDX,N,T}
        checkbounds(set, index)
        new(set, index)
    end
end

SingletonSubset{N,T}(set::FunctionSet{N,T}, index) =
    SingletonSubset{typeof(set),typeof(index),N,T}(set, index)

similar_subset(s::SingletonSubset, set, index) = SingletonSubset(set, index)

# Override the default `indices` because the field has a different name
indices(s::SingletonSubset) = s.index
indices(s::SingletonSubset, i) = s.index[i]

index(s::SingletonSubset) = s.index

eachindex(s::SingletonSubset) = 1:1

# Internally, we use StaticArrays (SVector) to represent points, except in
# 1d where we use scalars. Here, for convenience, you can call a function with
# x, y, z arguments and so on. These are wrapped into an SVector.
(s::SingletonSubset)(x) = eval_set_element(superset(s), index(s), x)
(s::SingletonSubset)(x, y...) = s(SVector(x, y...))


#########################################
# The default logic of creating subsets
#########################################

# We define the behaviour of getindex for function sets: `subset` create a
# suitable subset based on the type of the indices
getindex(s::FunctionSet, idx) = subset(s, idx)
getindex(s::FunctionSet, idx, indices...) = subset(s, (idx, indices...))

# - By default we generate a large subset
subset(s::FunctionSet, indices) = LargeSubset(s, indices)

# - If the type indicates there is only one element we create a singleton subset
for singleton_type in (:Int, :Tuple, :CartesianIndex, :NativeIndex)
    @eval subset(s::FunctionSet, idx::$singleton_type) = SingletonSubset(s, idx)
    # Specialize for Subset's in order to avoid ambiguities...
    @eval subset(s::Subset, idx::$singleton_type) = SingletonSubset(superset(s), indices(s)[idx])
end

# - For an array, which as far as we know has no additional structure, we create
# a small subset by default. For large arrays, a large subset may be more efficient.
subset(s::FunctionSet, indices::Array) = SmallSubset(s, indices)

subset(s::FunctionSet, ::Colon) = s

# Avoid creating nested subsets
subset(s::Subset, idx) = subset(superset(s), indices(s)[idx])
