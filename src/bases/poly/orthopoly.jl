# orthopoly.jl

"""
`OrthogonalPolynomials` is the abstract supertype of all univariate orthogonal
polynomials.
"""
abstract type OrthogonalPolynomials{T} <: PolynomialBasis{T}
end

const OPS{T} = OrthogonalPolynomials{T}

const OPSpan{A,F <: OrthogonalPolynomials} = Span{A,F}

is_orthogonal(b::OPS) = true
is_biorthogonal(b::OPS) = true

approx_length(b::OPS, n::Int) = n

derivative_space(s::OPSpan, order::Int; options...) = resize(s, length(s)-order)
antiderivative_space(s::OPSpan, order::Int; options...) = resize(s, length(s)+order)

length(o::OrthogonalPolynomials) = o.n

function dot(s::OPSpan, f1::Function, f2::Function, nodes::Array=native_nodes(set(s)); options...)
    T = coeftype(s)
	# To avoid difficult points at the ends of the domain.
	dot(x->weight(set(s),x)*f1(x)*f2(x), clip_and_cut(nodes, -T(1)+eps(real(T)), +T(1)-eps(real(T))); options...)
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

# CAVE: we have to add F <: OrthogonalPolynomials at the end, otherwise
# OPSpan{A,F} also seems to match non-polynomial sets F (in Julia 0.6).
# Using OPSpan as types of the arguments, i.e. without parameters, is fine and
# only matches with polynomial sets. But here we use parameters to enforce that
# the two spaces have the same type of set, and same type of coefficients.
function extension_operator(s1::OPSpan{A,F}, s2::OPSpan{A,F}; options...) where {A,F <: OrthogonalPolynomials}
    @assert length(s2) >= length(s1)
    IndexExtensionOperator(s1, s2, 1:length(s1))
end

function restriction_operator(s1::OPSpan{A,F}, s2::OPSpan{A,F}; options...) where {A,F <: OrthogonalPolynomials}
    @assert length(s2) <= length(s1)
    IndexRestrictionOperator(s1, s2, 1:length(s2))
end



# Default evaluation of an orthogonal polynomial: invoke the recurrence relation
eval_element(b::OPS, idx::Int, x) = recurrence_eval(b, idx, x)
eval_element_derivative(b::OPS, idx::Int, x) = recurrence_eval_derivative(b, idx, x)


"""
Evaluate an orthogonal polynomial using the three term recurrence relation.
The recurrence relation is assumed to have the form:
'''
    p_{n+1}(x) = (A_n x + B_n) * p_n(x) - C_n * p_{n-1}(x)
'''
with the coefficients implemented by the `rec_An`, `rec_Bn` and `rec_Cn`
functions.
This is the convention followed by the DLMF, see `http://dlmf.nist.gov/18.9#i`.
"""
function recurrence_eval(b::OPS, idx::Int, x)
	T = rangetype(b)
    z0 = one(T)
    z1 = convert(T, rec_An(b, 0) * x + rec_Bn(b, 0))

    if idx == 1
        return z0
    end
    if idx == 2
        return z1
    end

    z = z1
    for i = 1:idx-2
        z = (rec_An(b, i)*x + rec_Bn(b, i)) * z1 - rec_Cn(b, i) * z0
        z0 = z1
        z1 = z
    end
    z
end


"""
Evaluate the derivative of an orthogonal polynomial, based on taking the
derivative of the three-term recurrence relation (see `recurrence_eval`).
"""
function recurrence_eval_derivative(b::OPS, idx::Int, x)
	T = rangetype(b)
    z0 = one(T)
    z1 = convert(T, rec_An(b, 0) * x + rec_Bn(b, 0))
    z0_d = zero(T)
    z1_d = convert(T, rec_An(b, 0))

    if idx == 1
        return z0_d
    end
    if idx == 2
        return z1_d
    end

    z = z1
    z_d = z1_d
    for i = 1:idx-2
        z = (rec_An(b, i)*x + rec_Bn(b, i)) * z1 - rec_Cn(b, i) * z0
        z_d = (rec_An(b, i)*x + rec_Bn(b, i)) * z1_d + rec_An(b, i)*z1 - rec_Cn(b, i) * z0_d
        z0 = z1
        z1 = z
        z0_d = z1_d
        z1_d = z_d
    end
    z_d
end


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
    T = rangetype(b)
    # n is the maximal degree polynomial
    n = length(b)
    α = zeros(T, n+1)
    β = zeros(T, n+1)

    # We keep track of the leading order coefficients of p_{k-1}, p_k and
    # p_{k+1} in γ_km1, γ_k and γ_kp1 respectively. The recurrence relation
    # becomes, with p_k(x) = γ_k q_k(x):
    #
    # γ_{k+1} q_{k+1}(x) = (A_k x + B_k) γ_k q_k(x) - C_k γ_{k-1} q_{k-1}(x).
    #
    # From this we derive the formulas below.

    # We start with k = 1 and recall that p_0(x) = 1 and p_1(x) = A_0 x + B_0,
    # hence γ_0 = 1 and γ_1 = A_0.
    γ_km1 = 1
    γ_k = rec_An(b, 0)
    α[1] = -rec_Bn(b, 0) / rec_An(b, 0)
    # We arbitrarily choose β_0 = β[1] = 0. TODO: change later.
    β[1] = 0

    # We can now loop for k from 1 to n.
    for k in 1:n
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

function jacobi_matrix(b::OPS)
    T = rangetype(b)
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

function roots(b::OPS)
    J = jacobi_matrix(b)
    eig(J)[1]
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
    J = jacobi_matrix(b)
    x,v = eig(J)
    b0 = first_moment(b)
    # In the real-valued case it is sufficient to use the first element of the
    # eigenvector. See e.g. Gautschi's book, "Orthogonal Polynomials and Computation".
    w = b0 * v[1,:].^2
    x,w
end

# In the complex-valued case, we have to compute the weights by summing explicitly
# over the full eigenvector.
function gauss_rule(b::OPS{T}) where {T <: Complex}
    J = jacobi_matrix(b)
    x,v = eig(J)
    b0 = first_moment(b)
    w = similar(x)
    for i in 1:length(w)
        w[i] = b0/sum(v[:,i].^2)
    end
    x,w
end

function leading_order_coefficient(b::OPS{T}, idx) where {T}
    @assert 1 <= idx <= length(b)
    γ = one(T)
    for k in 0:idx-2
        γ *= rec_An(b, k)
    end
    γ
end
