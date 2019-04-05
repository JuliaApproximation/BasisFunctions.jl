
"""
A `MultiDict` is the concatenation of several dictionaries. The elements are contained
in an indexable set, such as a tuple or an array. In case of an array, the number
of dictionaries may be large.

The native representation of a `MultiDict` is a `BlockVector`, of which each element
is the native representation of the corresponding element of the multidict.

Evaluation of an expansion at a point is defined by summing the evaluation of all
functions in the set at that point.
"""
struct MultiDict{DICTS,S,T} <: CompositeDict{S,T}
    dicts   ::  DICTS
    offsets ::  Vector{Int}

    function MultiDict{DICTS,S,T}(dicts) where {DICTS,S,T}
        offsets = compute_offsets(dicts)
        new(dicts, offsets)
    end
end



# Is this constructor type-stable? Probably not, even if T is given, because
# of the use of dimension below.
function MultiDict(dicts, S, T)
    for dict in dicts
        # Is this the right check here?
        @assert promote_type(domaintype(dict), S) == S
        @assert promote_type(codomaintype(dict), T) == T
    end
    MultiDict{typeof(dicts),S,T}(dicts)
end

function MultiDict(dicts)
    S = reduce(promote_type, map(domaintype, dicts))
    T = reduce(promote_type, map(codomaintype, dicts))
    C = reduce(promote_type, map(coefficienttype, dicts))
    MultiDict(map(s->promote_coefficienttype(s,C), dicts), S, T)
end

similar_dictionary(set::MultiDict, dicts) = MultiDict(dicts)

multidict(set::Dictionary) = set

# When manipulating multidicts, we create Array's of dicts by default
multidict(s1::Dictionary, s2::Dictionary) = MultiDict([s1,s2])
multidict(s1::MultiDict, s2::MultiDict) = MultiDict(vcat(elements(s1), elements(s2)))
multidict(s1::MultiDict, s2::Dictionary) = MultiDict(vcat(elements(s1), s2))
multidict(s1::Dictionary, s2::MultiDict) = MultiDict(vcat(s1, elements(s2)))

multidict(s1::Dictionary, s2::Dictionary, s3::Dictionary...) =
    multidict(multidict(s1,s2), s3...)

function multidict(dicts::AbstractArray)
    if length(dicts) == 1
        multidict(dicts[1])
    else
        MultiDict(flatten(MultiDict, dicts, Dictionary{domaintype(dicts[1]),codomaintype(dicts[1])}))
    end
end

multispan(spans::Span...) = Span(multidict(map(dictionary, spans)...))

multispan(spans::AbstractArray) = Span(multidict(map(dictionary, spans)))

# Perhaps we don't want this behaviour, that [b;b] creates a MultiDict, rather
# than an array of Dictionary's
# vcat(s1::Dictionary, s2::Dictionary) = multidict(s1,s2)

⊕(d1::Dictionary, d2::Dictionary) = multidict(d1, d2)
⊕(d1::Span, d2::Span) = multispan(d1, d2)

∪(d1::Dictionary, d2::Dictionary) =
    error("Union of dictionaries is not supported: use ⊕ instead")

name(dict::MultiDict) = "Union of dictionaries"

for op in (:isbasis, :isframe)
    # Redirect the calls to multiple_isbasis with the elements as extra arguments,
    # and that method can decide whether the property holds for the multidict.
    fname = Symbol("multiple_$(op)")
    @eval $op(s::MultiDict) = ($fname)(s, elements(s)...)
    # By default, multidicts do not have these properties:
    @eval ($fname)(s, elements...) = false
end

for op in (:isorthogonal, :isbiorthogonal)
    # Redirect the calls to multiple_isbasis with the elements as extra arguments,
    # and that method can decide whether the property holds for the multidict.
    fname = Symbol("multiple_$(op)")
    @eval $op(s::MultiDict, m::Measure) = ($fname)(s, m, elements(s)...)
    # By default, multidicts do not have these properties:
    @eval ($fname)(s, m::Measure, elements...) = false
end

for op in (:hasinterpolationgrid, :hastransform)
    fname = Symbol("multiple_$(op)")
    @eval $op(s::MultiDict) = ($fname)(s, elements(s)...)
    @eval ($fname)(s, elements...) = false
end


# Try to return ranges of an underlying set, if possible
function subdict(dict::MultiDict, idx::OrdinalRange{Int})
    i1 = multilinear_index(dict, first(idx))
    i2 = multilinear_index(dict, last(idx))
    # Check whether the range lies fully in one set
    if i1[1] == i2[1]
        subdict(element(dict, i1[1]), i1[2]:step(idx):i2[2])
    else
        LargeSubdict(dict, idx)
    end
end



for op in [:support, :moment, :norm]
    @eval $op(set::MultiDict, idx::Int) = $op(set, multilinear_index(set, idx))
    # Pass along a linear or a native index to the subset
    @eval function $op(set::MultiDict, idx::Union{MultilinearIndex,Tuple{Int,Any}})
        i,j = idx
        $op(set.dicts[i], j)
    end
end

support(set::MultiDict) = union(map(support,elements(set))...)

resize(d::MultiDict, n::Int) = resize(d, approx_length(d, n))

approx_length(d::MultiDict, n::Int) = ceil(Int, n/numelements(d)) * ones(Int,numelements(d))

## Differentiation

derivative_dict(s::MultiDict, order; options...) =
    MultiDict(map(b-> derivative_dict(b, order; options...), elements(s)))

antiderivative_dict(s::MultiDict, order; options...) =
    MultiDict(map(b-> antiderivative_dict(b, order; options...), elements(s)))

for op in [:differentiation_operator, :antidifferentiation_operator]
    @eval function $op(s1::MultiDict, s2::MultiDict, order; T=op_eltype(s1,s2), options...)
        if numelements(s1) == numelements(s2)
            BlockDiagonalOperator(DictionaryOperator{T}[$op(element(s1,i), element(s2, i), order; options...) for i in 1:numelements(s1)], s1, s2)
        else
            # We have a situation because the sizes of the multidicts don't match.
            # The derivative set may have been a nested multidict that was flattened. This
            # case occurs for example in multidicts involving an AugmentedSet, because the
            # derivative of a single AugmentedSet may be a MultiDict.
            # The problem is we don't know which elements of s1 to match with which elements of s2.
            # Resolve the situation by looking at the standard derivative sets of each element of s1.
            # This may not be correct if one of the elements has multiple derivative sets, and
            # the user had chosen a non-standard one.
            ops = DictionaryOperator{T}[$op(el; options...) for el in elements(s1)]
            BlockDiagonalOperator(ops, s1, s2)
        end
    end
end

grid_evaluation_operator(dict::MultiDict, gb::GridBasis, grid::AbstractGrid; T = op_eltype(dict, gb), options...) =
    block_row_operator( DictionaryOperator{T}[grid_evaluation_operator(el, gb, grid; T=T, options...) for el in elements(dict)], dict, gb)

## Rescaling

apply_map(s::MultiDict, m) = multidict(map( t-> apply_map(t, m), elements(s)))

## Projecting
function project(s::MultiDict, f::Function; options...)
    Z = BlockArray{T}(undef,[length(e) for e in elements(s)])
    for (i,el) in enumerate(elements(s))
        setblock!(Z, project(el, f; options...), i)
    end
    Z
end


## Printing

string(dict::MultiDict) = "Union of $(numelements(dict)) dictionaries"

function stencilarray(dict::MultiDict)
    A = Any[]
    push!(A, element(dict, 1))
    for i = 2:numelements(dict)
        push!(A, " ⊕ ")
        push!(A, element(dict, i))
    end
    A
end

stencil_parentheses(dict::MultiDict) = true
object_parentheses(dict::MultiDict) = true
