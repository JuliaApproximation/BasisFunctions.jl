# Code for a generic orthogonal polynomial, defined in terms of its recurrence coefficients

"""
A generic orthogonal polynomial sequence is determined by its recurrence
coefficients. The `GenericOPS` type stores the coefficients `A_n`, `B_n` and
`C_n` from the recurrence relation in the following form:
```
p_{n+1}(x) = (A_n x + B_n) * p_n(x) - C_n * p_{n-1}(x).
p_{-1} = 0, p_0 = p0
```
"""
struct GenericOPS{T} <: OPS{T}
    moment  ::  T
    p0      ::  T
    rec_a   ::  Vector{T}
    rec_b   ::  Vector{T}
    rec_c   ::  Vector{T}
    support
    weight

    function GenericOPS{T}(moment, rec_a, rec_b, rec_c, support, p0=one(T), weight=nothing) where {T}
        @assert length(rec_a) == length(rec_b) == length(rec_c)
        new(moment, p0, rec_a, rec_b, rec_c, support, weight)
    end
end

GenericOPS(moment::T, rec_a::Vector{A}, rec_b::Vector{B}, rec_c::Vector{C}, support, p0=one(T), weight=nothing) where {T,A,B,C} =
    GenericOPS{promote_type(T,A,B,C)}(moment, rec_a, rec_b, rec_c, support, p0, weight)

function MonicOPSfromQuadrature(n, my_quadrature_rule, other...; options...)
    α, β = adaptive_stieltjes(n,my_quadrature_rule; options...)
    MonicOPSfromMonicCoefficients(α, β, other...)
end

function OrthonormalOPSfromQuadrature(n, my_quadrature_rule, other...; options...)
    α, β = adaptive_stieltjes(n+1,my_quadrature_rule; options...)
    ONPSfromMonicCoefficients(α, β, other...)
end

MonicOPSfromMonicCoefficients(α::Vector{A}, β::Vector{B}, support::Domain{T}, other...) where {T,A,B} =
    GenericOPS{promote_type(T,A,B)}(β[1], ones(T,length(α)), -α, β, support, one(T), other...)

function ONPSfromMonicCoefficients(α::Vector{A}, β::Vector{B}, support::Domain{T}, other...) where {T,A,B}
    a,b,c = monic_to_orthonormal_recurrence_coefficients(α,β)
    GenericOPS{promote_type(T,A,B)}(β[1], a, b, c, support, 1/sqrt(β[1]), other...)
end



support(b::GenericOPS) = b.support

size(b::GenericOPS) = (length(b.rec_a),)

weightfun(b::GenericOPS, x) = (b.weight==nothing) ? error("weight not defined for this Generic OPS") : b.weight(x)

function similar(b::GenericOPS, ::Type{T}, n::Int) where {T}
    @assert n <= length(b)
    GenericOPS{T}(b.moment, b.rec_a[1:n], b.rec_b[1:n], b.rec_c[1:n], b.support, b.p0, b.weight)
end

first_moment(b::GenericOPS) = b.moment

rec_An(b::GenericOPS, n::Int) = b.rec_a[n+1]

rec_Bn(b::GenericOPS, n::Int) = b.rec_b[n+1]

rec_Cn(b::GenericOPS, n::Int) = b.rec_c[n+1]

p0(b::GenericOPS) = b.p0
