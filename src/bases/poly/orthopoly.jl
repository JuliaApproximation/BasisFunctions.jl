
"""
`OrthogonalPolynomials` is the abstract supertype of all univariate orthogonal
polynomials.
"""
abstract type OrthogonalPolynomials{S,T} <: PolynomialBasis{S,T}
end

const OPS{S,T} = OrthogonalPolynomials{S,T}



is_orthogonal(b::OPS) = true
is_biorthogonal(b::OPS) = true

approx_length(b::OPS, n::Int) = n

derivative_dict(s::OPS, order::Int; options...) = resize(s, length(s)-order)
antiderivative_dict(s::OPS, order::Int; options...) = resize(s, length(s)+order)

size(o::OrthogonalPolynomials) = (o.n,)

p0(::OPS{T}) where {T} = one(T)

function dot(s::OPS, f1::Function, f2::Function, nodes::Array=native_nodes(dictionary(s)); options...)
    T = real(coefficienttype(s))
	# To avoid difficult points at the ends of the domain.
	dot(x->weight(s,x)*f1(x)*f2(x), clip_and_cut(nodes, -T(1)+eps(real(T)), +T(1)-eps(real(T))); options...)
end

clip(a::Real, low::Real, up::Real) = min(max(low, a), up)

function clip_and_cut(a::Array{T,1}, low, up) where {T <: Real}
	clipped = clip.(a,low, up)
	t = clipped[1]
	s = 1
	for i in 2:length(a)
		t != clipped[i] && break
		t = clipped[i]
		s += 1
	end
	t = clipped[end]
	e = length(a)
	for i in length(a)-1:-1:1
		t != clipped[i] && break
		t = clipped[i]
		e -= 1
	end
	clipped[s:e]
end

has_extension(b::OPS) = true

# CAVE: we have to add D <: OrthogonalPolynomials at the end, otherwise

# Using OPS as types of the arguments, i.e. without parameters, is fine and
# only matches with polynomial sets. But here we use parameters to enforce that
# the two spaces have the same type of set, and same type of coefficients.
function extension_operator(s1::OPS, s2::OPS; options...)
    @assert length(s2) >= length(s1)
    IndexExtensionOperator(s1, s2, 1:length(s1))
end

function restriction_operator(s1::OPS, s2::OPS; options...)
    @assert length(s2) <= length(s1)
    IndexRestrictionOperator(s1, s2, 1:length(s2))
end



# Default evaluation of an orthogonal polynomial: invoke the recurrence relation
unsafe_eval_element(b::OPS, idx::PolynomialDegree, x) =
    recurrence_eval(b, idx, x)
unsafe_eval_element_derivative(b::OPS, idx::PolynomialDegree, x) =
    recurrence_eval_derivative(b, idx, x)


"""
Evaluate an orthogonal polynomial using the three term recurrence relation.
The recurrence relation is assumed to have the form:
'''
    p_{n+1}(x) = (A_n x + B_n) * p_n(x) - C_n * p_{n-1}(x)
    p_{-1} = 0; p_0 = p0
'''
with the coefficients implemented by the `rec_An`, `rec_Bn` and `rec_Cn`
functions and with the initial value implemented with the `p0` function.
This is the convention followed by the DLMF, see `http://dlmf.nist.gov/18.9#i`.
"""
function recurrence_eval(b::OPS, idx::PolynomialDegree, x)
	T = codomaintype(b)
    z0 = T(p0(b))
    z1 = convert(T, rec_An(b, 0) * x + rec_Bn(b, 0))*z0

    d = degree(idx)
    if d == 0
        return z0
    end
    if d == 1
        return z1
    end

    z = z1
    for i = 1:d-1
        z = (rec_An(b, i)*x + rec_Bn(b, i)) * z1 - rec_Cn(b, i) * z0
        z0 = z1
        z1 = z
    end
    z
end

recurrence_eval(b::OPS, idx::LinearIndex, x) = recurrence_eval(b, native_index(b, idx), x)

"""
Evaluate all the polynomials in the orthogonal polynomial sequence in the given
point `x` using the three-term recurrence relation.
"""
function recurrence_eval!(result, b::OPS, x)
    @assert length(result) == length(b)

    result[1] = p0(b)
    if length(b) > 1
        result[2] = rec_An(b, 0)*x + rec_Bn(b, 0)
        for i = 2:length(b)-1
            result[i+1] = (rec_An(b, i-1)*x + rec_Bn(b, i-1)) * result[i] - rec_Cn(b, i-1) * result[i-1]
        end
    end
    result
end


"""
Evaluate all the orthonormal polynomials in the sequence in the given
point `x`, using the three-term recurrence relation.

The implementation is based on Gautschi, 2004, Theorem 1.29, p. 12.
"""
function recurrence_eval_orthonormal!(result, b::OPS, x)
    @assert length(result) == length(b)

    α, β = monic_recurrence_coefficients(b)
    # Explicit formula for the constant orthonormal polynomial
    result[1] = 1/sqrt(first_moment(b))
    if length(result) > 1
        # We use an explicit formula for the first degree polynomial too
        result[2] = 1/sqrt(β[2]) * (x-α[1]) * result[1]
        # The rest follows (1.3.13) in the book of Gautschi
        for i = 2:length(b)-1
            result[i+1] = 1/sqrt(β[i+1]) * ( (x-α[i])*result[i] - sqrt(β[i])*result[i-1])
        end
    end
    result
end

"""
Evaluate the derivative of an orthogonal polynomial, based on taking the
derivative of the three-term recurrence relation (see `recurrence_eval`).
"""
function recurrence_eval_derivative(b::OPS, idx::PolynomialDegree, x)
	T = codomaintype(b)
    z0 = one(p0(b))
    z1 = convert(T, rec_An(b, 0) * x + rec_Bn(b, 0))*z0
    z0_d = zero(T)
    z1_d = convert(T, rec_An(b, 0))

    d = degree(idx)
    if d == 0
        return z0_d
    end
    if d == 1
        return z1_d
    end

    z = z1
    z_d = z1_d
    for i = 1:d-1
        z = (rec_An(b, i)*x + rec_Bn(b, i)) * z1 - rec_Cn(b, i) * z0
        z_d = (rec_An(b, i)*x + rec_Bn(b, i)) * z1_d + rec_An(b, i)*z1 - rec_Cn(b, i) * z0_d
        z0 = z1
        z1 = z
        z0_d = z1_d
        z1_d = z_d
    end
    z_d
end

recurrence_eval_derivative(b::OPS, idx::LinearIndex, x) =
    recurrence_eval_derivative(b, native_index(b, idx), x)

"""
Return the recurrence coefficients `α_n` and `β_n` for the monic orthogonal
polynomials (i.e., with leading order coefficient `1`). The recurrence relation
is
```
p_{n+1}(x) = (x-α_n)p_n(x) - β_n p_{n-1}(x).
```
The result is a vector with indices starting at `1` (such that `α_n` above is
given by `α[n+1]`). The last value is `α_{n-1} = α[n]`.
"""
function monic_recurrence_coefficients(b::OPS)
    T = codomaintype(b)
    # n is the maximal degree polynomial
    n = length(b)
    α = zeros(T, n)
    β = zeros(T, n)

    # We keep track of the leading order coefficients of p_{k-1}, p_k and
    # p_{k+1} in γ_km1, γ_k and γ_kp1 respectively. The recurrence relation
    # becomes, with p_k(x) = γ_k q_k(x):
    #
    # γ_{k+1} q_{k+1}(x) = (A_k x + B_k) γ_k q_k(x) - C_k γ_{k-1} q_{k-1}(x).
    #
    # From this we derive the formulas below.

    # We start with k = 1 and recall that p_0(x) = 1 and p_1(x) = A_0 x + B_0,
    # hence γ_0 = 1 and γ_1 = A_0.
    γ_km1 = one(T)
    γ_k = rec_An(b, 0)
    α[1] = -rec_Bn(b, 0) / rec_An(b, 0)
    # see equation (1.3.6) from Gautschi's book, "Orthogonal Polynomials and Computation"
    β[1] = first_moment(b)

    # We can now loop for k from 1 to n.
    for k in 1:n-1
        γ_kp1 = rec_An(b, k) * γ_k
        α[k+1] = -rec_Bn(b, k) * γ_k / γ_kp1
        β[k+1] =  rec_Cn(b, k) * γ_km1 / γ_kp1
        γ_km1 = γ_k
        γ_k = γ_kp1
    end
    α, β
end


"""
Evaluate the three-term recurrence relation for monic orthogonal polynomials
```
p_{n+1}(x) = (x-α_n)p_n(x) - β_n p_{n-1}(x).
```
"""
function monic_recurrence_eval(α, β, idx, x)
    @assert idx > 0
    T = eltype(α)
    # n is the maximal degree polynomial, we want to evaluate p_n(x)
    n = length(α)

    # We store the values p_{k-1}(x), p_k(x) and p_{k+1}(x) in z_km1, z_k and
    # z_kp1 respectively.
    z_km1 = zero(T)
    z_k = one(T)
    z_kp1 = z_k
    if idx == 1
        z_k
    else
        for k in 0:idx-2
            z_kp1 = (x-α[k+1])*z_k - β[k+1]*z_km1
            z_km1 = z_k
            z_k = z_kp1
        end
        z_k
    end
end


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
    J
end

function roots(b::OPS{T}) where {T<:Number}
    J = symmetric_jacobi_matrix(b)
    eigen(J).values
end

function roots(b::OPS{T}) where {T<:Union{BigFloat}}
    J = symmetric_jacobi_matrix(b)
    # assuming the user has imported GenericLinearAlgebra.jl
    sort(real(eigvals!(J)))
end

gauss_points(b::OPS) = roots(b)

# We say that has_grid is true only for Float64 because it relies on an
# eigenvalue decomposition and that is currently not (natively) supported in
# BigFloat
has_grid(b::OPS{Float64}) = true

grid(b::OPS{Float64}) = ScatteredGrid(roots(b))

"Return the first moment, i.e., the integral of the weight function."
function first_moment(b::OPS)
    # To be implemented by the concrete subtypes
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
    x,v = eig(J)
    b0 = first_moment(b)
    w = similar(x)
    for i in 1:length(w)
        w[i] = b0 * v[1,i]^2 / sum(v[:,i].^2)
    end
    x,w
end

function gauss_rule(b::OPS{T}) where {T <: Union{BigFloat,Complex{BigFloat}}}
    x = gauss_points(b)
    w = gauss_weights_from_points(b, x)
    x,w
end

function sorted_gauss_rule(b::OPS)
    x,w = gauss_rule(b)
    idx = sortperm(real(x))
    x[idx], w[idx]
end

function gaussjacobi(n::Int,α::T,β::T) where {T<:BigFloat}
    x, w = sorted_gauss_rule(JacobiPolynomials(n,α,β))
    @assert norm(imag(x)) < eps(T)
    @assert norm(imag(w)) < eps(T)
    real(x), real(w)
end

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

function leading_order_coefficient(b::OPS{T}, idx) where {T}
    @assert 1 <= idx <= length(b)
    γ = one(T)
    for k in 0:idx-2
        γ *= rec_An(b, k)
    end
    γ
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
function stieltjes(N, x::Array{RT},w::Array{T}) where {T,RT}
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
function stieltjes!(a::Array{T},b::Array{T},N,x::Array{RT},w::Array{T},p0::Array{T},p1::Array{T},p2::Array{T},scratch::Array{T}) where {T, RT}
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
    a .= 1 ./ sqrt.(view(β,2:length(β)))
    b .= -1.0.*view(α,1:length(α)-1)./sqrt.(view(β,2:length(β)))
    c .= sqrt.(view(β,1:length(β)-1)./view(β,2:length(β)))
    a,b,c
end
