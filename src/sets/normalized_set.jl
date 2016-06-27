#normalized_set.jl

"""
A normalized set represents the normalization of an existing set.
"""
immutable NormalizedSet{S,N,T} <: FunctionSet{N,T}
    set     ::  S

    NormalizedSet(set::FunctionSet{N,T}) = new(set)
end

@compat (b::NormalizedSet)(x...) = call_set(b, x...)

NormalizedSet{N,T}(s::FunctionSet{N,T}) = NormalizedSet{typeof(s),N,T}(s)

set(s::NormalizedSet) = s.set

# Method delegation
for op in (:length,)
    @eval $op(s::NormalizedSet) = $op(s.set)
end

# Delegation of property methods
for op in (:is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op(s::NormalizedSet) = $op(set(s))
end

call_element(b::NormalizedSet, i, x) = call(b.set, i, x) / norm(b.set, i)


normalize(s::FunctionSet) = NormalizedSet(s)

normalize(s::NormalizedSet) = s
