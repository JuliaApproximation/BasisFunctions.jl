# span.jl

"""
The span of a function set is the set of all possible expansions in that set,
with coefficients of a given type `A`.
"""
struct Span{S,A}
    set ::  S
end

Span(set::FunctionSet, ::Type{A} = coefficient_type(s)) where {A} = Span{typeof(s),A}(set)

rangetype(s::Span{S,A}) where {S,A} = typeof(zero(A)*zero(rangetype(s)))
