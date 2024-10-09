
"""
`OrthogonalPolynomials` is the abstract supertype of all univariate orthogonal
polynomials.
"""
abstract type OrthogonalPolynomials{T} <: PolynomialBasis{T}
end

"Supertype of orthogonal polynomials on the interval `[-1,1]`."
abstract type IntervalOrthogonalPolynomials{T} <: OrthogonalPolynomials{T} end

support(b::IntervalOrthogonalPolynomials{T}) where {T} = ChebyshevInterval{T}()

const OPS{T} = OrthogonalPolynomials{T}
const IntervalOPS{T} = IntervalOrthogonalPolynomials{T}

abstract type OrthogonalPolynomial{T} <: Polynomial{T} end

approx_length(b::OPS, n::Int) = n

# Taking the derivative of a polynomial lowers its degree. This results in a
# rectangular differentiation matrix. With the reducedegree option set to false,
# the degree is maintained (resulting in a square differentiation matrix).
function derivative_dict(dict::OPS, order::Int; reducedegree = true, options...)
	@assert order >= 0
	if reducedegree
		@assert length(dict)-order >= 0
		resize(dict, length(dict)-order)
	else
		dict
	end
end

function antiderivative_dict(s::OPS, order::Int; options...)
	@assert order >= 0
	resize(s, length(s)+order)
end

size(o::OrthogonalPolynomials) = (o.n,)

p0(::OPS{T}) where {T} = one(T)

hasconstant(b::OPS) = true
coefficients_of_one(b::OPS) = (c=zeros(b); c[1]=1; c)

hasx(b::OPS) = length(b) > 1
function coefficients_of_x(b::OPS)
	b2 = similar(b, 2)
	E = matrix(conversion(Monomials{domaintype(b)}(2), b2))
	c = zeros(b)
	c[1] = E[1,2]
	c[2] = E[2,2]
	c
end

hasextension(b::OPS) = true

extension(::Type{T}, src::O, dest::O; options...) where {T,O <: OPS} = IndexExtension{T}(src, dest, 1:length(src))
restriction(::Type{T}, src::O, dest::O; options...) where {T,O <: OPS} = IndexRestriction{T}(src, dest, 1:length(dest))

include("recurrence.jl")

# Default evaluation of an orthogonal polynomial: invoke the recurrence relation
unsafe_eval_element(b::OPS, idx::PolynomialDegree, x) =
    recurrence_eval(b, idx, x)
function unsafe_eval_element_derivative(b::OPS, idx::PolynomialDegree, x, order)
	if orderiszero(order)
		unsafe_eval_element(b, idx, x)
	else
		@assert order == 1
		recurrence_eval_derivative(b, idx, x)
	end
end

## Some routines for OPS on [-1,1] are implemented for Chebyshev polynomials

promote_convertible(d1::IntervalOPS, d2::IntervalOPS) =
    iscompatible(d1,d2) ? (d1, d2) : (to_chebyshev_dict(d1), to_chebyshev_dict(d2))

expansion_roots(dict::IntervalOPS, coef) = roots(to_chebyshev(dict, coef))

expansion_multiply(src1::IntervalOPS, src2::IntervalOPS, coef1, coef2) =
    to_chebyshev_expansion_multiply(src1, src2, coef1, coef2)
function to_chebyshev_expansion_multiply(src1::IntervalOPS, src2::IntervalOPS, coef1, coef2)
    result = to_chebyshev(src1, coef1) * to_chebyshev(src2, coef2)
    dictionary(result), coefficients(result)
end
function expansion_multiply(src1::I, src2::I, coef1, coef2) where {I <: IntervalOPS}
    result1 = to_chebyshev(src1, coef1) * to_chebyshev(src2, coef2)
    result2 = conversion(dictionary(result1), resize(src1, length(result1))) * result1
    dictionary(result2), coefficients(result2)
end


hasmeasure(dict::OPS) = true


measure(ϕ::OrthogonalPolynomial) = measure(dictionary(ϕ))

weightfun(b::OPS, x) = weightfun(measure(b), x)


function gram1(T, dict::OPS, m; options...)
    isorthonormal(dict, m) && return IdentityOperator{T}(dict)
	if isorthogonal(dict, m)
		diagonal_gram(T, dict, m; options...)
	else
		default_gram(T, dict, m; options...)
	end
end

diagonal_gram(::Type{T}, dict::OPS, measure; options...) where {T} =
    default_diagonal_gram(T, dict::OPS, measure; options...)

function symmetric_jacobi_matrix(b::OPS)
    T = codomaintype(b)
    n = length(b)
    α, β = monic_recurrence_coefficients(b)
    J = zeros(T, n, n)
    for k in 1:n
        J[k,k] = α[k]
        if k > 1
            J[k,k-1] = sqrt(β[k])
        end
        if k < n
            J[k,k+1] = sqrt(β[k+1])
        end
    end
    SymTridiagonal(J)
end

"Return the roots of the (N+1)st orthogonal polynomial."
function ops_roots(b::OPS{T}) where {T<:Number}
    J = symmetric_jacobi_matrix(b)
    eigen(J).values
end

function ops_roots(b::OPS{T}) where {T<:BigFloat}
    J = symmetric_jacobi_matrix(b)
    # assuming the user has imported GenericLinearAlgebra.jl
    sort(real(eigvals!(J)))
end

interpolation_grid(b::OPS) = ops_roots(b)
hasinterpolationgrid(dict::OPS) = true
opsorthogonal(dict, measure) = length(dict) -issymmetric(dict) <= length(points(measure))

"Return the first moment, i.e., the integral of the weight function."
function first_moment(b::OPS)
    @warn "implement first_moment for $b"
end

"""
Compute the Gaussian quadrature rule using the roots of the orthogonal polynomial.
"""
function gauss_rule(b::OPS{T}) where {T <: Real}
    J = symmetric_jacobi_matrix(b)
    x,v = eigen(J)
    b0 = first_moment(b)
    # In the real-valued case it is sufficient to use the first element of the
    # eigenvector. See e.g. Gautschi's book, "Orthogonal Polynomials and Computation".
    w = b0 * v[1,:].^2
    x,w
end

# In the complex-valued case, we have to compute the weights by summing explicitly
# over the full eigenvector.
function gauss_rule(b::OPS{T}) where {T <: Complex}
    J = symmetric_jacobi_matrix(b)
    x,v = eigen(J)
    b0 = first_moment(b)
    w = similar(x)
    for i in 1:length(w)
        w[i] = b0 * v[1,i]^2 / sum(v[:,i].^2)
    end
    x,w
end

function gauss_rule(b::GenericMappedDict{T,T,<:OPS{T},ScalarAffineMap{T}}) where T
	x,w = gauss_rule(superdict(b))
	m = forward_map(b)
	discretemeasure(map_grid(m, x), w*matrix(m))
end

import FastGaussQuadrature: gaussjacobi
using GaussQuadrature: jacobi
gaussjacobi(n::Integer,α::T,β::T) where T<:AbstractFloat =
    jacobi(n, α, β)

"""
Compute the weights of Gaussian quadrature rule from the given roots of the
orthogonal polynomial and using the formula:
```
w_j = 1 / \\sum_{k=0}^{n-1} p_k(x_j)^2
```
This formula only holds for the orthonormal polynomials.
"""
function gauss_weights_from_points(b::OPS, roots)
    pk = zeros(eltype(roots), length(b))
    w = zeros(eltype(roots), length(b))
    for j in 1:length(w)
        recurrence_eval_orthonormal!(pk, b, roots[j])
        w[j] = 1/sum(pk.^2)
    end
    w
end

leading_order_coefficient(b::OPS, idx::PolynomialDegree) =
	leading_order_coefficient(b, value(idx)+1)

function leading_order_coefficient(b::OPS, idx::Int)
    @assert 1 <= idx <= length(b)
	if idx == 1
		unsafe_eval_element(b, idx, 0)
	else
		prod(rec_An(b, k) for k in 0:idx-2)
	end
end

"""
Stieltjes Algorithm

Given nodes (t_k) and weights (w_k) of a quadrature rule, the Stieltjes Algorithm
calculates the N first recurrence coefficients of the monic polynomials (p_k) satisfying

```
p_{k+1}(t) = (t-a_k)p_k(t)-b_kp_{k-1}(t)
p_0(t) = 1, p_{-1}(t) = 0
```

and orthogonal with respect to the inner product induced by the quadrature

```
\\sum w_k p_m(t_k)q_n(t_k) = \\delta_{m,n}
```

See e.g. Gautschi's book, "Orthogonal Polynomials and Computation" and its
accompanying code `https://www.cs.purdue.edu/archives/2002/wxg/codes/OPQ.html`

Note: The vectors used have indices starting at `1` (such that `a_n` above is
given by `a[n+1]`). The last value is `a_{n-1} = a[n]`.
"""
function stieltjes(N, x::AbstractArray{RT}, w::AbstractArray{T}) where {T,RT}
    @assert real(T) == RT
    # ####
    # Remove data with zero weights done in, e.g, Gautschi's book, "Orthogonal Polynomials and Computation".
    # ####
    # indices = sortperm(w)
    # x = x[indices]
    # w = w[indices]
    # index = minimum(find(!(w==0)))
    # w = w[index:length(w)]
    # x = x[index:length(w)]
    #
    # indices = sortperm(x)
    # x = x[indices]
    # w = w[indices]

    Ncap=length(x)
    # ####
    if N < 1 || N > Ncap
        error("Too little quadrature points")
    end

    a = zeros(T,N)
    b = zeros(T,N)

    p1 = zeros(T,Ncap)
    p2 = ones(T,Ncap)
    # The following scaling has to be adopted in case the
    # orthogonal polynomials underflow or overflow. In the
    # former case, c has to be chosen sufficiently large
    # positive, in the latter case sufficiently small
    # positive. (The example below relates to Table2_9.m.)
    #
    # if N==320
    #     c=1e50;
    #     p2=c*p2;
    #     s0=c^2*s0;
    # end
    p0 = similar(p1)
    scratch = similar(p1)
    stieltjes!(a,b,N,x,w,p0,p1,p2,scratch)
    a,b
end

"""
In-place Stieltjes Algorithm

see `stieltjes`

`p0`, `p1`, `p2`, and `scratch` are vectors of size `N`
"""
function stieltjes!(a::Array{T}, b::Array{T},N,
		x::AbstractArray{RT}, w::AbstractArray{T}, p0::Array{T},p1::Array{T},p2::Array{T},scratch::Array{T}) where {T, RT}
    @assert real(T) == RT
    tiny = 10*typemin(real(T))
    huge = .1*typemax(real(T))

    s0 = sum(w)
    a[1] = dot(x,w)/s0
    b[1] = s0
    for k=1:N-1
        copyto!(p0,p1)
        copyto!(p1,p2)

        p2 .= (x .- a[k]) .* p1 .- b[k] .* p0
        scratch .= (p2 .^ 2) .* w
        s1 = sum(scratch)
        scratch .= w .* (p2 .^ 2) .* x
        s2 = sum(scratch)
        scratch .= abs.(p2)
        if (maximum(abs.(scratch))>huge || abs(s2)>huge)
            error("impending overflow in stieltjes for k=$(k)")
        end
        if abs(s1)<tiny
            error("impending underflow in stieltjes for k=$(k)")
        end
        a[k+1] = s2/s1
        b[k+1] = s1/s0
        s0 = s1
    end
end

"""
Modified Chebyshev Algorithm

Implements the map of the modified moments

```
\\mu_k = \\int p_k(x) dλ(x), k=0,...,2N-1
```
where (p_k) are some monic orthogonal polynomials satisfying the recurrence relation
```
p_{k+1}(t) = a_k(t_k)p_k(t) - b_kp{k-1}(t)
p_0(t) = 1, p_{-1}(t) = 0
```
to the first recurrence coefficients (α_k), (β_k), with k=0,...,N-1, such that
the monic polynomials (q_k) satisfying
```
q_{k+1}(t) = α_k(t_k)q_k(t) - β_kq{k-1}(t)
q_0(t) = 1, q_{-1}(t) = 0
```
are orthogonal with respect to the measure λ.

By default a, and b are zero vectors and therefore the moments are assumed to be
```
\\mu_k = \\int x^k dλ(x), k=0,...,2N-1
```
This procedure is however not stable.


See e.g. Gautschi's book, "Orthogonal Polynomials and Computation". Algorithm 2.1, p77

Note: The vectors used have indices starting at `1` (such that `α_n` above is
given by `α[n+1]`). The last value is `α_{n-1} = α[n]`.
"""
function modified_chebyshev(m::Array{T}, a=zeros(T,length(m)), b=zeros(T,length(m))) where {T}
    L = length(m)
    n = L>>1
    α = Array{T}(undef, n)
    β = Array{T}(undef, n)
    σmone = Array{T}(undef, L)
    σzero = Array{T}(undef, L)
    σ = Array{T}(undef, L)

    modified_chebyshev!(α,β,m,a,b,σ,σzero,σmone,n,1)
    α, β
end

"""
Modified Chebyshev Algorithm

see `chebyshev`

`σ`, `σzero`, and `σmone` are vectors of size or larger then `2n`
`n` is the number of coefficients required and `os` is the offset, i.e.,
the initial index that can be used in `σ`, `σzero`, and `σmone`.
"""
function modified_chebyshev!(α,β,m,a,b,σ,σzero,σmone,n=length(α),os=1)
    # Initialisation
    α[1] = a[1]+m[2]/m[1]
    β[1] = m[1]
    fill!(σmone,0)
    copyto!(σzero,os,m,os,2n)
    # continue
    for k=1:n-1
        for l in k+os:2n-k-1+os
            σ[l] = σzero[l+1]-(α[k]-a[l]).*σzero[l]-β[k].*σmone[l]+b[l].*σzero[l-1]
        end
        α[k+1] = a[k+1]
        α[k+1] += σ[k+2]/σ[k+1]
        α[k+1] = a[k+1]+σ[k+2]/σ[k+1]-σzero[k+1]/σzero[k]
        β[k+1] = σ[k+1]/σzero[k]
        copyto!(σmone,os,σzero,os,2n)
        copyto!(σzero,os,σ,os,2n)
    end
    nothing
end

"""
Calculates the first `n` recurrence coefficients (`α_k`) and (`β_k`) of the monic orthogonal polynomials
up to the tolerance `tol`
given a quadrature rule `my_quadrature_rule` with a relative degree of exactness
```
d(M)/M = δ + O(1/M) as M goes to infinity
```
with d(M) the degree of exactness. Thus δ is 2 for Gaussian quadrature.

See equation page 101 from Gautschi's book, "Orthogonal Polynomials and Computation"
"""
function adaptive_stieltjes(n,my_quadrature_rule::Function; tol = 1e-12, δ = 1, maxits = 20, quadrature_size=false)
    M = 1 + floor(Int, (2n-1)/δ)
    nodes, weights = my_quadrature_rule(M)
    ELT = eltype(nodes)

    α0,β0 = stieltjes(n,nodes,weights)

    M = M+1
    nodes, weights = my_quadrature_rule(M)
    α1,β1 = stieltjes(n,nodes,weights)

    no_its = 2
    while reduce(&, abs.(β1-β0) .> tol.*abs.(β1))
        if no_its > maxits
            @warn("accuracy of Stieltjes is not obtained, degree of Quadrature is $(M) and error is $(maximum(abs.(β1-β0)./abs.(β1)))")
            break
        end
        M = M+2^floor(Int,no_its/5)
        no_its = no_its + 1
        copyto!(α0,α1)
        copyto!(β0,β1)

        nodes, weights = my_quadrature_rule(M)
        α1,β1 = stieltjes(n,nodes,weights)
    end
    quadrature_size ? (α1, β1, M) : (α1, β1)
end

"""
Transforms the `N` recurrence coefficients (α_k) and (β_k) of monic
orthogonal polynomials (p_k), which satisfy the three-term recurrence relation
```
p_{k+1}(t) = (t-α_k)p_k(t)-β_kp_{k-1}(t)
```
to the `N-1` first recurrence coefficients (a_k), (b_k), and (c_k) the associated
monic polynomials (q_k), such that

```
q_{k+1}(t) = (a_kt+b_k)q_k(t)-c_kq_{k-1}(t)
```
"""
function monic_to_orthonormal_recurrence_coefficients(α::Array{T}, β::Array{T}) where {T}
    n = length(α)
    a = Array{T}(undef, n-1)
    b = Array{T}(undef, n-1)
    c = Array{T}(undef, n-1)
    monic_to_orthonormal_recurrence_coefficients!(a,b,c,α,β)
    a,b,c
end

"""
See `monic_to_orthonormal_recurrence_coefficients`
"""
function monic_to_orthonormal_recurrence_coefficients!(a::Array{T},b::Array{T},c::Array{T},α::Array{T},β::Array{T}) where {T}
    a .= 1 ./ sqrt.(β[2:length(β)])
    b .= -one(T).*α[1:length(α)-1]./sqrt.(β[2:length(β)])
    c .= sqrt.(β[1:length(β)-1]./β[2:length(β)])
    a,b,c
end
