# generic_op.jl

# Code for a generic orthogonal polynomial, defined in terms of its recurrence coefficients

"""
A generic orthogonal polynomial sequence if determined by its recurrence
coefficients. The `GenericOPS` type stores the coefficients `A_n`, `B_n` and
`C_n` from the recurrence relation in the following form:
```
p_{n+1}(x) = (A_n x + B_n) * p_n(x) - C_n * p_{n-1}(x).
```
"""
struct GenericOPS{T} <: OPS{T}
    moment  ::  T
    rec_a   ::  Vector{T}
    rec_b   ::  Vector{T}
    rec_c   ::  Vector{T}

    function GenericOPS{T}(moment, rec_a, rec_b, rec_c) where {T}
        @assert length(rec_a) == length(rec_b) == length(rec_c)
        new(moment, rec_a, rec_b, rec_c)
    end
end

GenericOPS(moment::T, rec_a::Vector{A}, rec_b::Vector{B}, rec_c::Vector{C}) where {T,A,B,C} =
    GenericOPS{promote_type(T,A,B,C)}(moment, rec_a, rec_b, rec_c)

const GenericOPSpan{A, F <: GenericOPS} = Span{A,F}

length(b::GenericOPS) = length(b.rec_a)

name(b::GenericOPS) = "Generic OPS"

set_promote_domaintype(b::GenericOPS, ::Type{S}) where {S} =
    GenericOPS{S}(b.rec_a, b.rec_b, b.rec_c)

function resize(b::GenericOPS, n)
    @assert n <= length(b)
    GenericOPS(b.moment, b.rec_a[1:n], b.rec_b[1:n], b.rec_c[1:n])
end

first_moment(b::GenericOPS) = b.moment

rec_An(b::GenericOPS, n::Int) = b.rec_a[n+1]

rec_Bn(b::GenericOPS, n::Int) = b.rec_b[n+1]

rec_Cn(b::GenericOPS, n::Int) = b.rec_c[n+1]
