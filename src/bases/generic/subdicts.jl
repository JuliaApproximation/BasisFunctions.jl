# subdicts.jl

checkbounds(set::Dictionary, idx::Range) =
    (checkbounds(set, first(idx)); checkbounds(set, last(idx)))

checkbounds(set::Dictionary, idx::CartesianRange) =
    (checkbounds(set, first(idx)); checkbounds(set, last(idx)))

function checkbounds(set::Dictionary, idx::Array)
    for i in idx
        checkbounds(set, i)
    end
end

#############################
# An abstract subdictionary
#############################

# We start with a generic description of subdicts and imlement as much
# functionality as possible at the abstract level of Subdictionary.
# This is followed by specialized concrete types of subdicts.

"""
A `Subdictionary` is an abstract subset of a dictionary. It is characterized by
the underlying larger dictionary, and a subcollection of its indices.
"""
abstract type Subdictionary{S,T} <: Dictionary{S,T}
end

const SubdictSpan{A,S,T,D <: Subdictionary} = Span{A,S,T,D}

# We assume that the underlying set is stored in a field called superdict
superdict(s::Subdictionary) = s.superdict

superdict(s::SubdictSpan) = superdict(dictionary(s))

superspan(s::SubdictSpan) = Span(superdict(s), coeftype(s))

# We assume that the underlying indices are stored in a field called indices
indices(s::Subdictionary) = s.indices
indices(s::Subdictionary, i) = s.indices[i]

indices(s::SubdictSpan) = indices(dictionary(s))

# The concrete subdict should implement `similar_subdict`, a routine that
# returns a subdict of a similar type as itself, but with a different underlying set.
dict_promote_domaintype(s::Subdictionary, ::Type{S}) where {S} =
    similar_subdict(s, promote_domaintype(superdict(s), S), indices(s))

apply_map(s::Subdictionary, map) = similar_subdict(s, apply_map(superdict(s), map), indices(s))


name(s::Subdictionary) = "Subdictionary of " * name(superdict(s)) * " with indices " * string(indices(s))

length(s::Subdictionary) = length(indices(s))

size(s::Subdictionary) = size(indices(s))

eachindex(s::Subdictionary) = eachindex(indices(s))

##
# For various properties it is not a priori clear whether a subdict has them or not,
# even if the underlying set does. For each function foo, we invoke a new function
# subdict_foo, that has the underlying set and the range of indices as additional
# arguments. This function call can be intercepted and implemented for specific kinds
# of subdicts, if it is known that foo applies.

resize(s::Subdictionary, n) = subdict_resize(s, n, superdict(s), indices(s))

has_grid(s::Subdictionary) = subdict_has_grid(s, superdict(s), indices(s))
has_derivative(s::Subdictionary) = subdict_has_derivative(s, superdict(s), indices(s))
has_antiderivative(s::Subdictionary) = subdict_has_antiderivative(s, superdict(s), indices(s))
has_transform(s::Subdictionary) = subdict_has_transform(s, superdict(s), indices(s))

derivative_space(s::SubdictSpan, order; options...) = subdict_derivative_space(s, order, superspan(s), indices(s); options...)
antiderivative_space(s::SubdictSpan, order; options...) = subdict_antiderivative_space(s, order, superspan(s), indices(s); options...)

grid(s::Subdictionary) = subdict_grid(s, superdict(s), indices(s))


# By default, we have none of the properties
subdict_has_derivative(s::Subdictionary, superdict, indices) = false
subdict_has_antiderivative(s::Subdictionary, superdict, indices) = false
subdict_has_transform(s::Subdictionary, superdict, indices) = false
subdict_has_grid(s::Subdictionary, superdict, indices) = false




for op in (:isreal, :is_orthogonal, :is_basis)
    @eval $op(dict::Subdictionary) = $op(superdict(dict))
end


native_index(d::Subdictionary, idx::Int) = idx
linear_index(d::Subdictionary, idxn::Int) = idxn

for op in [:left, :right]
    @eval $op(d::Subdictionary) = $op(superdict(d))
end

for op in [:left, :right, :moment, :norm]
    @eval $op(d::Subdictionary, i) = $op(superdict(d), indices(d, i))
end

in_support(d::Subdictionary, i, x) = in_support(superdict(d), indices(d, i), x)

eval_element(d::Subdictionary, i, x) = eval_element(superdict(d), indices(d, i), x)

eval_element_derivative(d::Subdictionary, i, x) = eval_element_derivative(superdict(d), indices(d, i), x)




#####################
# Concrete subdicts
#####################


"""
A `LargeSubdict` is a large subset of a dictionary. Operators associated with the
subset are implemented in terms of corresponding operators on the underlying dictionary.
This often leads to an explicit extension to the full set, but it can take
advantage of possible fast implementations for the underlying set. For large subsets
this is more efficient than iterating over the individual elements.
"""
struct LargeSubdict{SET,IDX,S,T} <: Subdictionary{S,T}
    superdict   ::  SET
    indices     ::  IDX

    function LargeSubdict{SET,IDX,S,T}(dict::Dictionary{S,T}, idx::IDX) where {SET,IDX,S,T}
        checkbounds(dict, idx)
        new(dict, idx)
    end
end

const LargeSubdictSpan{A,S,T,D <: LargeSubdict} = Span{A,S,T,D}

LargeSubdict(dict::Dictionary{S,T}, indices) where {S,T} =
    LargeSubdict{typeof(dict),typeof(indices),S,T}(dict, indices)

similar_subdict(d::LargeSubdict, dict, indices) = LargeSubdict(dict, indices)

grid(d::LargeSubdict) = grid(superdict(d))

function extension_operator(s1::LargeSubdictSpan, s2::Span; options...)
    @assert dictionary(s2) == superdict(s1)
    IndexExtensionOperator(s1, s2, indices(s1))
end

function restriction_operator(s1::Span, s2::LargeSubdictSpan; options...)
    @assert dictionary(s1) == superdict(s2)
    IndexRestrictionOperator(s1, s2, indices(s2))
end

# In general, the derivative set of a subdict can be the whole derivative set
# of the underlying set. We can not know generically whether the derivative set
# can be indexed as well. Same for antiderivative.
# Yet, we can generically define a differentiation_operator by extending the subdict
# to the whole set and then invoking the differentiation operator of the latter,
# and we choose that to be the default.
subdict_has_derivative(s::LargeSubdict, superdict, indices) = has_derivative(superdict)
subdict_has_antiderivative(s::LargeSubdict, superdict, indices) = has_antiderivative(superdict)

subdict_derivative_space(s::LargeSubdictSpan, order, superdict, indices; options...) =
    derivative_space(superdict, order; options...)
subdict_antiderivative_space(s::LargeSubdictSpan, order, superdict, indices; options...) =
    antiderivative_space(superdict, order; options...)

function differentiation_operator(s1::LargeSubdictSpan, s2::Span, order::Int; options...)
    @assert s2 == derivative_space(s1, order)
    D = differentiation_operator(superspan(s1), s2, order; options...)
    E = extension_operator(s1, superspan(s1); options...)
    D*E
end

function antidifferentiation_operator(s1::LargeSubdictSpan, s2::Span, order::Int; options...)
    @assert s2 == antiderivative_space(s1, order)
    D = antidifferentiation_operator(superspan(s1), s2, order; options...)
    E = extension_operator(s1, superspan(s1); options...)
    D*E
end




"""
A `SmallSubdict` is a subset of a dictionary with a small number of indices.
The difference with a regular function subset is that operators on a small set
are implemented by iterating explicitly over the indices, and not in terms of
an operator on the full underlying set.
"""
struct SmallSubdict{SET,IDX,S,T} <: Subdictionary{S,T}
    superdict   ::  SET
    indices     ::  IDX

    function SmallSubdict{SET,IDX,S,T}(dict::Dictionary{S,T}, indices::IDX) where {SET,IDX,S,T}
        checkbounds(dict, indices)
        new(dict, indices)
    end
end

const SmallSubdictSpan{A,S,T,D <: SmallSubdict} = Span{A,S,T,D}

SmallSubdict(dict::Dictionary{S,T}, indices) where {S,T} =
    SmallSubdict{typeof(set),typeof(indices),S,T}(dict, indices)


similar_subdict(d::SmallSubdict, dict, indices) = SmallSubdict(dict, indices)


"""
A SingletonSubdict represent a single element from an underlying set.
"""
struct SingletonSubdict{SET,IDX,S,T} <: Subdictionary{S,T}
    superdict   ::  SET
    index       ::  IDX

    function SingletonSubdict{SET,IDX,S,T}(dict::Dictionary{S,T}, index::IDX) where {SET,IDX,S,T}
        checkbounds(dict, index)
        new(dict, index)
    end
end

const SingletonSubdictSpan{A,S,T,D <: SingletonSubdict} = Span{A,S,T,D}

SingletonSubdict(dict::Dictionary{S,T}, index) where {S,T} =
    SingletonSubdict{typeof(dict),typeof(index),S,T}(dict, index)

similar_subdict(d::SingletonSubdict, dict, index) = SingletonSubdict(dict, index)


# Override the default `indices` because the field has a different name
indices(s::SingletonSubdict) = s.index
indices(s::SingletonSubdict, i) = s.index[i]

index(s::SingletonSubdict) = s.index

eachindex(s::SingletonSubdict) = 1:1

# Internally, we use StaticArrays (SVector) to represent points, except in
# 1d where we use scalars. Here, for convenience, you can call a function with
# x, y, z arguments and so on. These are wrapped into an SVector.
(s::SingletonSubdict)(x) = eval_set_element(superdict(s), index(s), x)
(s::SingletonSubdict)(x, y...) = s(SVector(x, y...))


#########################################
# The default logic of creating subdicts
#########################################

# We define the behaviour of getindex for function sets: `subdict` create a
# suitable subdict based on the type of the indices
getindex(s::Dictionary, idx) = subdict(s, idx)
getindex(s::Dictionary, idx, indices...) = subdict(s, (idx, indices...))

# - By default we generate a large subdict
subdict(s::Dictionary, indices) = LargeSubdict(s, indices)

# - If the type indicates there is only one element we create a singleton subdict
for singleton_type in (:Int, :Tuple, :CartesianIndex, :NativeIndex)
    @eval subdict(s::Dictionary, idx::$singleton_type) = SingletonSubdict(s, idx)
    # Specialize for Subdictionary's in order to avoid ambiguities...
    @eval subdict(s::Subdictionary, idx::$singleton_type) = SingletonSubdict(superdict(s), indices(s)[idx])
end

# - For an array, which as far as we know has no additional structure, we create
# a small subdict by default. For large arrays, a large subdict may be more efficient.
subdict(s::Dictionary, indices::Array) = SmallSubdict(s, indices)

subdict(s::Dictionary, ::Colon) = s

# Avoid creating nested subdicts
subdict(s::Subdictionary, idx) = subdict(superdict(s), indices(s)[idx])

getindex(s::Span, idx) = Span(subdict(dictionary(s), idx), coeftype(s))
