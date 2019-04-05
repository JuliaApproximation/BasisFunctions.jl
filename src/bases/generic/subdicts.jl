
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



# We assume that the underlying set is stored in a field called superdict
superdict(dict::Subdictionary) = dict.superdict

# We assume that the underlying indices are stored in a field called superindices
superindices(dict::Subdictionary) = dict.superindices
superindices(dict::Subdictionary, idx::Int) = dict.superindices[idx]
superindices(dict::Subdictionary, idx::DefaultNativeIndex) = superindices(dict, value(idx))

function similar(d::Subdictionary, ::Type{T}, n::Int) where {T}
    @assert n == length(d)
    similar_subdict(d, similar(superdict(d), T), superindices(d))
end

apply_map(s::Subdictionary, map) = similar_subdict(s, apply_map(superdict(s), map), superindices(s))


hasstencil(dict::Subdictionary) = true
stencilarray(dict::Subdictionary) = [ superdict(dict), "(", string(superindices(dict)), ")" ]

stencil_parentheses(dict::Subdictionary) = true

support(s::Subdictionary) = support(superdict(s))

support(s::Subdictionary, i) = support(superdict(s),superindices(s,i))

size(s::Subdictionary) = size(superindices(s))

##
# For various properties it is not a priori clear whether a subdict has them or not,
# even if the underlying set does. For each function foo, we invoke a new function
# subdict_foo, that has the underlying set and the range of indices as additional
# arguments. This function call can be intercepted and implemented for specific kinds
# of subdicts, if it is known that foo applies.

resize(s::Subdictionary, n) = subdict_resize(s, n, superdict(s), superindices(s))

hasinterpolationgrid(s::Subdictionary) = subdict_hasinterpolationgrid(s, superdict(s), superindices(s))
hasderivative(s::Subdictionary) = subdict_hasderivative(s, superdict(s), superindices(s))
hasantiderivative(s::Subdictionary) = subdict_hasantiderivative(s, superdict(s), superindices(s))
hastransform(s::Subdictionary) = subdict_hastransform(s, superdict(s), superindices(s))

derivative_dict(s::Subdictionary, order; options...) = subdict_derivative_dict(s, order, superdict(s), superindices(s); options...)
antiderivative_dict(s::Subdictionary, order; options...) = subdict_antiderivative_dict(s, order, superdict(s), superindices(s); options...)

interpolation_grid(s::Subdictionary) = subdict_interpolation_grid(s, superdict(s), superindices(s))

subdict_interpolation_grid(s::Subdictionary, gb::GridBasis, indices) =
    interpolation_grid(gb)[indices]

# By default, we have none of the properties
subdict_hasderivative(s::Subdictionary, superdict, superindices) = false
subdict_hasantiderivative(s::Subdictionary, superdict, superindices) = false
subdict_hastransform(s::Subdictionary, superdict, superindices) = false
subdict_hasinterpolationgrid(s::Subdictionary, superdict, superindices) = false




for op in (:isreal, :isbasis)
    @eval $op(dict::Subdictionary) = $op(superdict(dict))
end

for op in (:isreal, :isbiorthogonal, :isorthonormal, :isorthogonal)
    @eval $op(dict::Subdictionary, measure::Measure) = $op(superdict(dict), measure::Measure)
end

for op in (:moment, :norm)
    @eval $op(d::Subdictionary, i) = $op(superdict(d), superindices(d, i))
end

dict_in_support(d::Subdictionary, i, x) = in_support(superdict(d), superindices(d, i), x)

unsafe_eval_element(d::Subdictionary, i, x) = unsafe_eval_element(superdict(d), superindices(d, i), x)

unsafe_eval_element_derivative(d::Subdictionary, i, x) = unsafe_eval_element_derivative(superdict(d), superindices(d, i), x)

hasmeasure(dict::Subdictionary) = hasmeasure(superdict(dict))
measure(dict::Subdictionary) = measure(superdict(dict))

innerproduct1(d1::Subdictionary, i, d2, j, measure; options...) =
    innerproduct(superdict(d1), superindices(d1,i), d2, j, measure; options...)
innerproduct2(d1, i, d2::Subdictionary, j, measure; options...) =
    innerproduct(d1, i, superdict(d2), superindices(d2, j), measure; options...)


#####################
# Concrete subdicts
#####################


"""
A `LargeSubdict` is a large subset of a dictionary. Operators associated with the
subset are implemented in terms of corresponding operators on the underlying dictionary.
This often leads to an explicit extension to the full set, but it can take
advantage of possible fast implementations for the underlying dictionary. For
large subsets this is more efficient than iterating over the individual elements.
"""
struct LargeSubdict{SET,IDX,S,T} <: Subdictionary{S,T}
    superdict       ::  SET
    superindices    ::  IDX

    function LargeSubdict{SET,IDX,S,T}(dict::Dictionary{S,T}, idx::IDX) where {SET,IDX,S,T}
        checkbounds(dict, idx)
        new(dict, idx)
    end
end

LargeSubdict(dict::Dictionary{S,T}, superindices) where {S,T} =
    LargeSubdict{typeof(dict),typeof(superindices),S,T}(dict, superindices)

name(dict::LargeSubdict) = "Large subdictionary"

similar_subdict(d::LargeSubdict, dict, superindices) = LargeSubdict(dict, superindices)

interpolation_grid(d::LargeSubdict) = interpolation_grid(superdict(d))

function extension_operator(s1::LargeSubdict, s2::Dictionary; options...)
    @assert s2 == superdict(s1)
    IndexExtensionOperator(s1, s2, superindices(s1); options...)
end

function restriction_operator(s1::Dictionary, s2::LargeSubdict; options...)
    @assert s1 == superdict(s2)
    IndexRestrictionOperator(s1, s2, superindices(s2); options...)
end

# In general, the derivative set of a subdict can be the whole derivative set
# of the underlying set. We can not know generically whether the derivative set
# can be indexed as well. Same for antiderivative.
# Yet, we can generically define a differentiation_operator by extending the subdict
# to the whole set and then invoking the differentiation operator of the latter,
# and we choose that to be the default.
subdict_hasderivative(s::LargeSubdict, superdict, superindices) = hasderivative(superdict)
subdict_hasantiderivative(s::LargeSubdict, superdict, superindices) = hasantiderivative(superdict)

subdict_derivative_dict(s::LargeSubdict, order, superdict, superindices; options...) =
    derivative_dict(superdict, order; options...)
subdict_antiderivative_dict(s::LargeSubdict, order, superdict, superindices; options...) =
    antiderivative_dict(superdict, order; options...)

function differentiation_operator(s1::LargeSubdict, s2::Dictionary, order; options...)
    @assert s2 == derivative_dict(s1, order)
    D = differentiation_operator(superdict(s1), s2, order; options...)
    E = extension_operator(s1, superdict(s1); options...)
    D*E
end

function antidifferentiation_operator(s1::LargeSubdict, s2::Dictionary, order; options...)
    @assert s2 == antiderivative_dict(s1, order)
    D = antidifferentiation_operator(superdict(s1), s2, order; options...)
    E = extension_operator(s1, superdict(s1); options...)
    D*E
end




"""
A `SmallSubdict` is a subset of a dictionary with a small number of indices.
The difference with a regular function subset is that operators on a small set
are implemented by iterating explicitly over the indices, and not in terms of
an operator on the full underlying set.
"""
struct SmallSubdict{SET,IDX,S,T} <: Subdictionary{S,T}
    superdict       ::  SET
    superindices    ::  IDX

    function SmallSubdict{SET,IDX,S,T}(dict::Dictionary{S,T}, indices::IDX) where {SET,IDX,S,T}
        checkbounds(dict, indices)
        new(dict, indices)
    end
end



SmallSubdict(dict::Dictionary{S,T}, superindices) where {S,T} =
    SmallSubdict{typeof(dict),typeof(superindices),S,T}(dict, superindices)

name(dict::SmallSubdict) = "Small subdictionary"


similar_subdict(d::SmallSubdict, dict, superindices) = SmallSubdict(dict, superindices)


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

SingletonSubdict(dict::Dictionary{S,T}, index) where {S,T} =
    SingletonSubdict{typeof(dict),typeof(index),S,T}(dict, index)

name(dict::SingletonSubdict) = _name(dict, superdict(dict))
_name(dict::SingletonSubdict, superdict::Dictionary) = "Dictionary element (singleton subdictionary)"

similar_subdict(d::SingletonSubdict, dict, index) = SingletonSubdict(dict, index)

Base.length(::SingletonSubdict) = 1

# Override the default `superindices` because the field has a different name
superindices(s::SingletonSubdict) = s.index
superindices(s::SingletonSubdict, i::Int) = s.index[i]

index(s::SingletonSubdict) = s.index

# Internally, we use StaticArrays (SVector) to represent points, except in
# 1d where we use scalars. Here, for convenience, you can call a function with
# x, y, z arguments and so on. These are wrapped into an SVector.
(s::SingletonSubdict)(x) = eval_element(superdict(s), index(s), x)
(s::SingletonSubdict)(x, y...) = s(SVector(x, y...))

innerproduct(f::SingletonSubdict, g::SingletonSubdict, measure; options...) =
    innerproduct(superdict(f), index(f), superdict(g), index(g), measure; options...)

innerproduct(d::SingletonSubdict, f; options...) =
    innerproduct(d, f, measure(superdict(d)); options...)


#########################################
# The default logic of creating subdicts
#########################################

# First some additions to the checkbounds ecosystem

checkbounds(::Type{Bool}, dict::Dictionary, indices::CartesianIndices) =
    checkbounds(Bool, dict, first(indices)) && checkbounds(Bool, dict, last(indices))

function checkbounds(::Type{Bool}, dict::Dictionary, indices::Array)
    result = true
    for idx in indices
        result &= checkbounds(Bool, dict, idx)
    end
    result
end


# We define the behaviour of getindex for function sets: `subdict` creates a
# suitable subdict based on the type of the indices
getindex(dict::Dictionary, idx) = subdict(dict, idx)
# convert multiple indices to a single tuple
getindex(dict::Dictionary, idx, superindices...) = subdict(dict, (idx, superindices...))

# - By default we generate a large subdict
subdict(dict::Dictionary, superindices) = LargeSubdict(dict, superindices)

# - If the type indicates there is only one element we create a singleton subdict
for singleton_type in (:Int, :Tuple, :CartesianIndex, :NativeIndex)
    @eval subdict(s::Dictionary, idx::$singleton_type) = SingletonSubdict(s, idx)
    # Specialize for Subdictionary's in order to avoid ambiguities...
    @eval subdict(s::Subdictionary, idx::$singleton_type) = SingletonSubdict(superdict(s), superindices(s)[idx])
end

# - For an array, which as far as we know has no additional structure, we create
# a small subdict by default. For large arrays, a large subdict may be more efficient.
subdict(s::Dictionary, superindices::Array) = SmallSubdict(s, superindices)

subdict(s::Dictionary, ::Colon) = s

# Avoid creating nested subdicts
subdict(s::Subdictionary, idx) = subdict(superdict(s), superindices(s)[idx])

# A gramoperator of a subdict orthonormal/orthogonal dictionary is also Identity/Diagonal
function gramoperator1(s::Subdictionary, m;
            T = promote_type(coefficienttype(s), domaintype(m)), options...)
    if isorthonormal(s, m)
        return IdentityOperator{T}(s)
    elseif isorthogonal(s, m)
        return DiagonalOperator(s, diag(gramoperator(superdict(s), m; T=T, options...))[superindices(s)]; T=T, options...)
    end
    default_gramoperator(s, m; T=T, options...)
end
