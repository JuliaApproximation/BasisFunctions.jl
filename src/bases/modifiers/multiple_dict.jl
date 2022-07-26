
"""
A `MultiDict` is the concatenation of several dictionaries. The components are contained
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
    MultiDict(map(dict->ensure_coefficienttype(C, dict), dicts), S, T)
end

similardictionary(set::MultiDict, dicts) = MultiDict(dicts)

multidict(dict::Dictionary) = dict

# When manipulating multidicts, we create Array's of dicts by default
multidict(d1::Dictionary, d2::Dictionary) = MultiDict([d1,d2])
multidict(d1::MultiDict, d2::MultiDict) = MultiDict(vcat(components(d1), components(d2)))
multidict(d1::MultiDict, d2::Dictionary) = MultiDict(vcat(components(d1), d2))
multidict(d1::Dictionary, d2::MultiDict) = MultiDict(vcat(d1, components(d2)))

multidict(d1::Dictionary, d2::Dictionary, dicts::Dictionary...) =
    multidict(multidict(d1,d2), dicts...)

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

vcat(dicts::Dictionary...) = multidict(dicts...)
⊕(dicts::Dictionary...) = multidict(dicts...)

for op in (:isbasis, :isframe)
    # Redirect the calls to multiple_isbasis with the elements as extra arguments,
    # and that method can decide whether the property holds for the multidict.
    fname = Symbol("multiple_$(op)")
    @eval $op(s::MultiDict) = ($fname)(s, components(s)...)
    # By default, multidicts do not have these properties:
    @eval ($fname)(s, components...) = false
end

for op in (:isorthogonal, :isbiorthogonal)
    # Redirect the calls to multiple_isbasis with the elements as extra arguments,
    # and that method can decide whether the property holds for the multidict.
    fname = Symbol("multiple_$(op)")
    @eval $op(s::MultiDict, m::Measure) = ($fname)(s, m, components(s)...)
    # By default, multidicts do not have these properties:
    @eval ($fname)(s, m::Measure, components...) = false
end

for op in (:hasinterpolationgrid, :hastransform)
    fname = Symbol("multiple_$(op)")
    @eval $op(s::MultiDict) = ($fname)(s, components(s)...)
    @eval ($fname)(s, components...) = false
end


# Try to return ranges of an underlying set, if possible
function sub(dict::MultiDict, idx::OrdinalRange{Int})
    i1 = multilinear_index(dict, first(idx))
    i2 = multilinear_index(dict, last(idx))
    # Check whether the range lies fully in one set
    if outerindex(i1) == outerindex(i2)
        sub(component(dict, outerindex(i1)), innerindex(i1):step(idx):innerindex(i2))
    else
        defaultsub(dict, idx)
    end
end



for op in (:support, :moment, :norm)
    @eval $op(set::MultiDict, idx::Int) = $op(set, multilinear_index(set, idx))
    # Pass along a linear or a native index to the subset
    @eval function $op(set::MultiDict, idx::Union{MultilinearIndex,Tuple{Int,Any}})
        $op(set.dicts[outerindex(idx)], innerindex(idx))
    end
end

support(set::MultiDict) = union(map(support,components(set))...)

measure(dict::MultiDict) = measure(component(dict, 1))

resize(d::MultiDict, n::Int) = resize(d, approx_length(d, n))

approx_length(d::MultiDict, n::Int) = ceil(Int, n/ncomponents(d)) * ones(Int,ncomponents(d))

## Differentiation

diff(Φ::MultiDict, order; options...) =
    multidict([diff(dict, order; options...) for dict in components(Φ)]...)
derivative_dict(Φ::MultiDict, order; options...) =
    MultiDict(map(d-> derivative_dict(d, order; options...), components(Φ)))

antiderivative_dict(Φ::MultiDict, order; options...) =
    MultiDict(map(d-> antiderivative_dict(d, order; options...), components(Φ)))

for op in (:differentiation, :antidifferentiation)
    @eval function $op(::Type{T}, src::MultiDict, dest::MultiDict, order; options...) where {T}
        if ncomponents(src) == ncomponents(dest)
            BlockDiagonalOperator{T}([$op(component(src,i), component(dest, i), order; options...) for i in 1:ncomponents(src)], src, dest)
        else
            # We have a situation because the sizes of the multidicts don't match.
            # The derivative set may have been a nested multidict that was flattened. This
            # case occurs for example in multidicts involving an AugmentedSet, because the
            # derivative of a single AugmentedSet may be a MultiDict.
            # The problem is we don't know which elements of src to match with which elements of dest.
            # Resolve the situation by looking at the standard derivative sets of each element of src.
            # This may not be correct if one of the elements has multiple derivative sets, and
            # the user had chosen a non-standard one.
            ops = DictionaryOperator{T}[$op(el, order; options...) for el in components(src)]
            BlockDiagonalOperator{T}(ops, src, dest)
        end
    end
end

evaluation(::Type{T}, dict::MultiDict, gb::GridBasis, grid::AbstractGrid; options...) where {T} =
    block_row_operator( DictionaryOperator{T}[evaluation(T, el, gb, grid; options...) for el in components(dict)], dict, gb)



function gramdual(dict::MultiDict, measure::Measure; verbose=false, options...)
    verbose && println("WARN: Are you sure you want `gramdual` and not `weightedsumdual`?")
    default_gramdual(dict, measure; verbose, options...)
end

## Rescaling

mapped_dict(s::MultiDict, m) = multidict(map( t-> mapped_dict(t, m), components(s)))

## Projecting
function project(s::MultiDict, f::Function; options...)
    Z = BlockArray{T}(undef,[length(e) for e in components(s)])
    for (i,el) in enumerate(components(s))
        Z[Block(i)] = project(el, f; options...)
        # setblock!(Z, project(el, f; options...), i)
    end
    Z
end


## Printing

Display.combinationsymbol(d::MultiDict) = Display.Symbol('⊕')
