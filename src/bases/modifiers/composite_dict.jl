
"""
`CompositeDict` is the abstract supertype of a dictionary that consists of
several subdictionaries. The `CompositeDict` type defines common routines for
indexing and iteration.

The representation of a `CompositeDict` is a `BlockVector`. The outer array of this
`BlockVector` adopts the structure of the components of the `CompositeDict`: if the components
are stored in a tuple, the outer array will be a tuple. If the components are
stored in an array, the outer array will be an array as well.

CompositeDict's that store components in an array can have a large number of components.
However, for good efficiency they should have the same type. Tuples are suitable
to hold mixed-type components with good efficiency, however in this case there can't
be too many.

The concrete subtypes differ in what evaluation means. Examples include:
- MultiDict, where evaluation is the sum of the evaluation of the subsets
- PiecewiseDict, where evaluation depends on the location of the evaluation point
"""
abstract type CompositeDict{S,T} <: Dictionary{S,T}
end

# We assume that every subset has an indexable field called dicts
components(set::CompositeDict) = set.dicts

# component(set::CompositeDict, range::AbstractRange) = similardictionary(set, set.dicts[range])
component(set::CompositeDict, range::AbstractRange) = error("this method is deprecated")

tail(set::CompositeDict) = ncomponents(set) == 2 ? component(set, 2) : similardictionary(set, component(set, 2:ncomponents(set)))

# We compute offsets of the individual dicts using a cumulative sum
compute_offsets(dicts::Array) = [0; cumsum(map(length, dicts))]

# Convert a tuple to an array in order to use cumsum like above
compute_offsets(dicts::NTuple{N,Any}) where {N} = compute_offsets([dict for dict in dicts])

"""
Pass the internal `offsets` vector of the composite dict. This is not safe because
the vector is not copied, hence its components could be changed. That would affect
the original composite dictionary. Use with care.
"""
unsafe_offsets(dict::CompositeDict) = dict.offsets

# Implement equality in terms of equality of the components.
==(s1::CompositeDict, s2::CompositeDict) = (components(s1) == components(s2))

size(s::CompositeDict) = (s.offsets[end],)

dimensions(d::CompositeDict) = map(dimensions, components(d))

## Concrete subtypes should override similardictionary and call their own constructor

similar(d::CompositeDict, ::Type{T}, n::Int) where {T} = similar(d, T, composite_size(d, n))

function similar(d::CompositeDict, ::Type{T}, size::Vector{Int}) where {T}
    @assert ncomponents(d) == length(size)
    similardictionary(d, map( (s,l) -> similar(s, T, l), components(d), size))
end

function similar(d::CompositeDict, ::Type{T}, size::Int...) where {T}
    @assert ncomponents(d) == length(size)
    similardictionary(d, map( (s,l) -> similar(s, T, l), components(d), size))
end

resize(d::CompositeDict, size) = similardictionary(d, map(resize, components(d), size))

composite_length(d::CompositeDict) = tuple(map(length, components(d))...)
block_length(dict::CompositeDict) = composite_length(dict)

function composite_size(d::CompositeDict, n::Int)
    if n == length(d)
        map(length, components(d))
    else
        L = ceil(Int, n/ncomponents(d))
        @assert ncomponents(d) * L == n
        L * ones(Int, ncomponents(d))
    end
end

function zeros(::Type{T}, set::CompositeDict) where {T}
    Z = BlockArray{T}(undef,[length(e) for e in components(set)])
    fill!(Z, 0)
    Z
end

tocoefficientformat(a, d::CompositeDict) = BlockVector(a, [length(e) for e in components(set)])

for op in (:isreal, )
    @eval $op(set::CompositeDict) = mapreduce($op, &, components(set))
end

for op in (:hasderivative, :hasantiderivative, :hasextension)
    @eval $op(set::CompositeDict) = mapreduce($op, &, components(set))
end
hasderivative(Φ::CompositeDict, order) =
    mapreduce(hasderivative, &, components(Φ))
hasantiderivative(Φ::CompositeDict, order) =
    mapreduce(hasantiderivative, &, components(Φ))

coefficienttype(dict::CompositeDict) = coefficienttype(component(dict,1))


##################
# Indexing
##################

# The native index is a MultilinearIndex, defined in basis/dictionary/indexing.jl
ordering(d::CompositeDict) = MultilinearIndexList(unsafe_offsets(d))

# For getindex: return indexed basis function of the underlying set
basisfunction(dict::CompositeDict, idx::LinearIndex) =
    basisfunction(dict, native_index(dict, idx))
basisfunction(dict::CompositeDict, idx::MultilinearIndices) =
    dict.dicts[outerindex(idx)][innerindex(idx)]


dict_in_support(set::CompositeDict, idx, x) = _dict_in_support(set, components(set), idx, x)
_dict_in_support(set::CompositeDict, dicts, idx, x) = in_support(dicts[outerindex(idx)], innerindex(idx), x)

eachindex(set::CompositeDict) = MultilinearIndexIterator(map(length, components(set)))


## Extension and restriction

extensionsize(set::CompositeDict) = map(extensionsize, components(set))

for op in [:extension, :restriction]
    @eval $op(::Type{T}, src::CompositeDict, dest::CompositeDict; options...) where {T} =
        BlockDiagonalOperator( DictionaryOperator{T}[$op(component(src,i),component(dest,i); options...) for i in 1:ncomponents(src)], src, dest)
end

# Calling and evaluation

unsafe_eval_element(set::CompositeDict{S,T}, idx::MultilinearIndices, x) where {S,T} =
    convert(T, unsafe_eval_element( component(set, outerindex(idx)), innerindex(idx), x))

unsafe_eval_element_derivative(set::CompositeDict{S,T}, idx::MultilinearIndices, x, order) where {S,T} =
    convert(T, unsafe_eval_element_derivative( component(set, outerindex(idx)), innerindex(idx), x, order))

derivative_dict(s::CompositeDict, order; options...) =
    similardictionary(s,map(u->derivative_dict(u, order; options...), components(s)))

antiderivative_dict(s::CompositeDict, order; options...) =
    similardictionary(s,map(u->antiderivative_dict(u, order; options...), components(s)))

function evaluation_matrix(::Type{T}, dict::CompositeDict, pts) where {T}
    a = BlockArray{T}(undef, [length(pts),], collect(composite_length(dict)))
    evaluation_matrix!(a, dict, pts)
end


dict_innerproduct1(d1::CompositeDict, i, d2, j, measure; options...) =
    dict_innerproduct(component(d1, outerindex(i)), innerindex(i), d2, j, measure; options...)
dict_innerproduct2(d1, i, d2::CompositeDict, j, measure; options...) =
    dict_innerproduct(d1, i, component(d2, outerindex(j)), innerindex(j), measure; options...)


## Printing

Display.displaystencil(d::CompositeDict) = composite_displaystencil(d)
show(io::IO, mime::MIME"text/plain", d::CompositeDict) = composite_show(io, mime, d)
show(io::IO, d::CompositeDict) = composite_show_compact(io, d)
