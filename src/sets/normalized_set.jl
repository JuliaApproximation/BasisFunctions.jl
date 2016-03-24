#normalized_set.jl

"""
A normalized set represents the normalization of an existing set.
"""
immutable NormalizedSet{S,N,T} <: FunctionSet{N,T}
    set     ::  S

    NormalizedSet(set::FunctionSet{N,T}) = new(set)
end

NormalizedSet{N,T}(s::FunctionSet{N,T}) = NormalizedSet{typeof(s),N,T}(s)

set(s::AugmentedSet) = s.set

# Method delegation
for op in (:length,)
    @eval $op(s::NormalizedSet) = $op(s.set)
end

# Delegation of type methods
for op in (:is_basis, :is_frame, :is_orthogonal, :is_biorthogonal)
    @eval $op{S,N,T}(::Type{NormalizedSet{S,N,T}}) = $op(S)
end

call_element(b::NormalizedSet, i, x) = call(b.set, i, x) / norm(b.set, i)


normalize(s::FunctionSet) = NormalizedSet(s)

normalize(s::NormalizedSet) = s

