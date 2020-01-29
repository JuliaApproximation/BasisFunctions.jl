
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


"Iterate over the values of an orthogonal polynomial sequence using the recurrence relation."
struct OPSValueIterator{T,O<:OPS{T}} <: DictionaryValueIterator{T}
	dict	::	O
	x		::	T
end

pointvalues(ops::OPS, x) = OPSValueIterator(ops, x)

# The state vector is (z0,z1,d) as follows:
# - z0 is the value of the polynomial of degree d-1
# - z1 is the value of the polynomial of degree d
function iterate(iter::OPSValueIterator)
	ops = dictionary(iter)
	val = p0(ops)
	if length(ops) > 0
		(1,val), (zero(val),val,0)
	else
		nothing
	end
end

# Here we have to compute the polynomial p_{d+1}(x) using p_d and p_{d-1}
function iterate(iter::OPSValueIterator, state)
	z0, z1, degree = state
	dict = dictionary(iter)
	x = point(iter)
	if degree == 0
		# We avoid evaluating rec_Cn with degree=0
		z = (rec_An(dict, degree)*x + rec_Bn(dict, degree)) * z1
		(degree+2,z), (z1, z, degree+1)
	elseif degree < length(dict)-1
		z = (rec_An(dict, degree)*x + rec_Bn(dict, degree)) * z1 - rec_Cn(dict, degree) * z0
		(degree+2,z), (z1, z, degree+1)
	else
		nothing
	end
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
