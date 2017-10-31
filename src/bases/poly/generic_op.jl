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
    rec_a   ::  Vector{T}
    rec_b   ::  Vector{T}
    rec_c   ::  Vector{T}

    function GenericOPS{T}(rec_a, rec_b) where {T}
        @assert length(rec_a) == length(rec_b)
        new{T}(rec_a, rec_b)
    end
end

const GenericOPSpan{A, F <: GenericOPS} = Span{A,F}

length(b::GenericOPS) = length(b.α)

name(b::GenericOPS) = "Generic OPS"

set_promote_domaintype(b::GenericOPS, ::Type{S}) where {S} =
    GenericOPS{S}(b.rec_a, b.rec_b, b.rec_c)

function resize(b::GenericOPS, n)
    @assert n <= length(b)
    GenericOPS(b.rec_a[1:n], b.rec_b[1:n])
end

rec_An(b::GenericOPS, n::Int) = b.rec_a[n]

rec_Bn(b::GenericOPS, n::Int) = b.rec_b[n]

rec_Cn(b::GenericOPS, n::Int) = b.rec_c[n]
