# subsets.jl

"""
A FunctionSubSet is a subset of a function set. It is characterized by the
underlying larger set, and a collection of indices into that set.
"""
immutable FunctionSubSet{SET, IDX, N, T} <: FunctionSet{N,T}
    set ::  SET
    idx ::  IDX

    FunctionSubSet(set::FunctionSet{N,T}, idx::IDX) = new(set, idx)
end

FunctionSubSet{N,T}(set::FunctionSet{N,T}, idx) =
    FunctionSubSet{typeof(set),typeof(idx),N,T}(set, idx)

set(s::FunctionSubSet) = s.set
indices(s::FunctionSubSet) = s.idx

name(s::FunctionSubSet) = "Subset of " + name(s.set) + " with indices " + string(idx)

promote_eltype{SET,IDX,N,T,S}(s::FunctionSubSet{SET,IDX,N,T}, ::Type{S}) =
    FunctionSubSet(promote_eltype(set(s), S), indices(s))

length(s::FunctionSubSet) = length(s.idx)

# It's not clear in general what to do when attempting to resize a subset. We
# can not simply resize the underlying set, because the indices may no longer
# be correct. Instead, we relegate to a resize_subset routine with set(s) as
# an additional argument. This function can be implemented for specific sets
# if needed.
resize(s::FunctionSubSet, n) = resize_subset(s, set(s), n)

isreal(s::FunctionSubSet) = isreal(set(s))

# A subset of an orthogonal set may not be complete, but it will still be orthogonal.
# The same does not hold in general for biorthogonal sets.
is_orthogonal(s::FunctionSubSet) = is_orthogonal(set(s))

for op in [:left, :right]
    @eval $op(s::FunctionSubSet) = $op(set(s))
end

for op in [:left, :right, :moment, :norm]
    @eval $op(s::FunctionSubSet, i) = $op(s.set, s.idx[i])
end


# We can convert the logical index of a subset to the natural index of the
# underlying set. The other way around is more difficult, because we would need
# the inverse of the s.idx map.
natural_index(s::FunctionSubSet, idx) = natural_index(set(s), s.idx[idx])

call_element(s::FunctionSubSet, i, x...) = call_element(s.set, s.idx[i], x...)

@compat (s::FunctionSubSet)(x...) = (@assert length(s) == 1; call_element(s, 1, x...))

eachindex(s::FunctionSubSet) = eachindex(s.idx)

getindex(s::FunctionSet, idx) = FunctionSubSet(s, idx)

getindex(s::FunctionSubSet, idx) = FunctionSubSet(set(s), s.idx[idx])
