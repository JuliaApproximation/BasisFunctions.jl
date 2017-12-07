# generic_op.jl

# Code for a generic orthogonal polynomial, defined in terms of its recurrence coefficients

"""
A generic orthogonal polynomial sequence if determined by its recurrence
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
    left    ::  T
    right   ::  T

    function GenericOPS{T}(moment, rec_a, rec_b, rec_c, left, right, p0=one(T)) where {T}
        @assert length(rec_a) == length(rec_b) == length(rec_c)
        new(moment, p0, rec_a, rec_b, rec_c, left, right)
    end
end

GenericOPS(moment::T, rec_a::Vector{A}, rec_b::Vector{B}, rec_c::Vector{C}, left, right, p0=one(T)) where {T,A,B,C} =
    GenericOPS{promote_type(T,A,B,C)}(moment, rec_a, rec_b, rec_c, left, right, p0)

function GenericOPSfromQuadrature(n, my_quadrature_rule, left, right; options...)
    α, β = adaptive_stieltjes(n,my_quadrature_rule; options...)
    A = eltype(α); B = eltype(β);
    T = promote_type(A,B)
    GenericOPS{T}(β[1], ones(T,n), -α, β, left, right)
end

function OrthonormalOPSfromQuadrature(n, my_quadrature_rule, left, right; options...)
    α, β = adaptive_stieltjes(n+1,my_quadrature_rule; options...)
    A = eltype(α); B = eltype(β);
    a,b,c = monic_to_orthonormal_recurrence_coefficients(α,β)
    T = promote_type(A,B)
    GenericOPS{T}(β[1], a, b, c, left, right, one(T)/sqrt(β[1]))
end

const GenericOPSpan{A, F <: GenericOPS} = Span{A,F}

left(b::GenericOPS) = b.left
left(b::GenericOPS, idx) = left(b)

right(b::GenericOPS) = b.right
right(b::GenericOPS, idx) = right(b)

length(b::GenericOPS) = length(b.rec_a)

name(b::GenericOPS) = "Generic OPS"

set_promote_domaintype(b::GenericOPS, ::Type{S}) where {S} =
    GenericOPS{S}(b.rec_a, b.rec_b, b.rec_c)

function resize(b::GenericOPS, n)
    @assert n <= length(b)
    GenericOPS(b.moment, b.rec_a[1:n], b.rec_b[1:n], b.rec_c[1:n], b.left, b.right, b.p0)
end

first_moment(b::GenericOPS) = b.moment

rec_An(b::GenericOPS, n::Int) = b.rec_a[n+1]

rec_Bn(b::GenericOPS, n::Int) = b.rec_b[n+1]

rec_Cn(b::GenericOPS, n::Int) = b.rec_c[n+1]

p0(b::GenericOPS) = b.p0

"""
Creates the (normalized) half range Chebyshev polynomials of the first kind.

These orthonormal polynomials `(p_k(t))` on [-1,1] are orthogonal with respect to the weight
```
w(t) = [(1-t)(t-m(T))]^{-1/2}
```
with
```
m(T) = 1-2cosec^2(pi/(2T))
```

They appear in the context of Fourier extensions. There, an even function `f(x)` on
`[-1,1]` is approximated with a cosine series in `x` on `[-T,T]`. By the cosine mapping
```
    y = m(x) = cos(pi/T x)
```
this problem is transformed. The Fourier Extension problem on `f` is then equivalent with
the approximation of `f(m^{-1}(y))` with orthogonal polynomials `(q_k(y))` on `[cos(pi/T),1]`

By another mapping the interval `[cos(pi/T),1]` is mapped to `[-1,1]` and the polynomials `(q_k(y))` are mapped
to the polynomials `(p_k(t))`

See Huybrechs 2010, On the Fourier Extension of nonperiodic functions
    Adcock, Huybrechs, Vaquero 2014, On the numerical stability of Fourier Extension

The indicator function refers to the indicator function of the domain the function
`f(x)` is approximated on. By default it is the indicator function of the interval [-1,1]

See also `HalfRangeChebyshevIkind`
"""
HalfRangeChebyshevIkind(n::Int, T::ELT, indicator_function_nodes::Vector{ELT}=default_indicator_nodes(ELT); options...) where ELT =
    HalfRangeChebyshev(n, ELT(-1//2), T, indicator_function_nodes; options...)

"""
Creates the (normalized) half range Chebyshev polynomials of the second kind.

These orthonormal polynomials `(p_k(t))` on [-1,1] are orthogonal with respect to the weight
```
w(t) = [(1-t)(t-m(T))]^{1/2}
```
with
```
m(T) = 1-2cosec^2(pi/(2T))
```

They appear in the context of Fourier extensions. There, an odd function `f(x)` on
`[-1,1]` is approximated with a sine series in `x` on `[-T,T]`.

See also `HalfRangeChebyshevIkind`
"""
HalfRangeChebyshevIIkind(n::Int, T::ELT, indicator_function_nodes::Vector{ELT}=default_indicator_nodes(ELT); options...) where ELT =
    HalfRangeChebyshev(n, ELT(1//2), T, indicator_function_nodes; options...)

m = (x,T)->2*(cos(pi/T*x)-cos(pi/T))/(1-cos(pi/T))-1
m_inv = (t,T)->T/pi*acos(cos(pi/T)+(1-cos(pi/T))/2*(t+1))
weight_of_indicator(T,indicator) = t->.5*(indicator(-m_inv(t,T))+indicator(m_inv(t,T)))
default_indicator_nodes(ELT) = [-ELT(1),ELT(1)]

using FastGaussQuadrature
function HalfRangeChebyshev(n::Int, α, T::ELT, indicator_function_nodes::Vector{ELT}; options...) where ELT
    my_quadrature_rule = n->_halfrangechebyshevweights(n, α, T, indicator_function_nodes)
    OrthonormalOPSfromQuadrature(n, my_quadrature_rule, -one(ELT), one(ELT); options...)
end

function _halfrangechebyshevweights(n, α::ELT, T::ELT, indicator_function_nodes::Vector{ELT}) where {ELT}
    @assert indicator_function_nodes[1] == -1 && indicator_function_nodes[end] == 1
    @assert reduce(&, true, indicator_function_nodes[1:end-1] .< indicator_function_nodes[2:end])
    @assert iseven(length(indicator_function_nodes))

    if length(indicator_function_nodes)==2
        nodes, weights = gaussjacobi(n, α, 0)
        Λ = weight_of_indicator(T,x->1)
        modified_weights = weights.*((nodes.-m(T,T)).^α).*Λ.(nodes)*2T/pi
        nodes, modified_weights
    else
        indicator = indicator_function(indicator_function_nodes)
        mapped_nodes = sort(unique(map(x->m(x,T), push!(indicator_function_nodes,0))))
        pop!(indicator_function_nodes)
        no_intervals = length(mapped_nodes)-1
        n_interval = n
        N = n_interval*no_intervals
        nodes = zeros(ELT,N)
        weights = zeros(ELT,N)


        for i in 1:no_intervals
            nodes_interval, weights_interval = gaussjacobi(n_interval, 0., 0.)
            a = mapped_nodes[i]
            b = mapped_nodes[i+1]
            nodes_interval[:] .= a + (b-a)/2*(nodes_interval+1)
            weights_interval[:] .= weights_interval*(b-a)/2
            Δ = weight_of_indicator(T,indicator)
            weights_interval[:] .= 2T/pi*weights_interval.*(1-nodes_interval).^α.*(nodes_interval-m(T,T)).^α.*Δ.(nodes_interval)

            nodes[1+(i-1)*n_interval:i*n_interval] .= nodes_interval[:]
            weights[1+(i-1)*n_interval:i*n_interval] .= weights_interval[:]
        end
        nodes, weights
    end
end

indicator_function(nodes) = x-> reduce(|, false, nodes[1:2:end] .<= x .<= nodes[2:2:end])
