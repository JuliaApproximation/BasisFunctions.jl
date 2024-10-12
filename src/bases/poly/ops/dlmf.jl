# Formulas from the Digital Library of Mathematical Functions.

"""
Pochhammer's Symbol.

See [DLMF](https://dlmf.nist.gov/5.2#iii) formula (5.2.4).
"""
pochhammer(a, n::Int) = n == 0 ? one(a) : prod(a+i for i in 0:n-1)
# or in terms of the gamma function (DLMF, 5.2.5):  gamma(n+a) / gamma(a)

##############################
# DLMF Table 18.3.1
# https://dlmf.nist.gov/18.3
##############################
# Formulas for squared norm hn and leading order coefficient kn

# The formula for An and A0
dlmf_18d3d1_A(n::Int, α, β) = 2^(α+β+1)*gamma(n+α+1)*gamma(n+β+1) / (factorial(n)*(2n+α+β+1)*gamma(n+α+β+1))
dlmf_18d3d1_A0(α, β) = 2^(α+β+1)*gamma(α+1)*gamma(β+1) / gamma(α+β+2)

jacobi_hn(n::Int, α, β) = n == 0 ? dlmf_18d3d1_A0(α, β) : dlmf_18d3d1_A(n, α, β)
jacobi_kn(n::Int, α, β) = pochhammer(n+α+β+1, n) / (2^n * factorial(n))

ultraspherical_hn(n::Int, λ::T) where T = 2^(1-2λ)*T(π)*gamma(n+2λ) / ((n+λ)*gamma(λ)^2*factorial(n))
ultraspherical_kn(n::Int, λ) = 2^n*pochhammer(λ, n) / factorial(n)

chebyshev_1st_hn(n::Int, ::Type{T} = Float64) where T = n == 0 ? T(π) : T(π)/2
chebyshev_1st_kn(n::Int, ::Type{T} = Float64) where T = n == 0 ? one(T) : T(2)^(n-1)

chebyshev_2nd_hn(n::Int, ::Type{T} = Float64) where T = T(π)/2
chebyshev_2nd_kn(n::Int, ::Type{T} = Float64) where T = T(2)^n

chebyshev_3rd_hn(n::Int, ::Type{T} = Float64) where T = T(π)
chebyshev_3rd_kn(n::Int, ::Type{T} = Float64) where T = T(2)^n

chebyshev_4rd_hn(n::Int, ::Type{T} = Float64) where T = T(π)
chebyshev_4rd_kn(n::Int, ::Type{T} = Float64) where T = T(2)^n

legendre_hn(n::Int, ::Type{T} = Float64) where T = 2 / T(2n+1)
legendre_kn(n::Int, ::Type{T} = Float64) where T = 2^n*pochhammer(one(T)/2, n) / factorial(n)

laguerre_hn(n::Int, α) = gamma(n+α+1) / factorial(n)
laguerre_kn(n::Int, α::T) where T = (-one(T))^n / factorial(n)

hermite_hn(n::Int, ::Type{T} = Float64) where T = sqrt(T(π)) * 2^n * factorial(n)
hermite_kn(n::Int, ::Type{T} = Float64) where T = T(2)^n


##############################
# DLMF Table 18.9.1
# http://dlmf.nist.gov/18.9#i
##############################
# Recurrence coefficients in the form
# p_{n+1}(x) = (A_n x + B_n) p_n(x) - C_n p_{n-1}(x)
# with initial values
# p_0(x) = 1 and p_1(x) = A_0x + B_0


# Jacobi, (18.9.2)
function jacobi_rec_An(n::Int, α, β)
    if (n == 0) && (α + β + 1 == 0)
        (α+β)/2+1
    else
        (2*n + α + β + 1) * (2n + α + β + 2) / (2 * (n+1) * (n + α + β + 1))
    end
end
function jacobi_rec_Bn(n::Int, α, β)
    if (n == 0) && ((α + β + 1 == 0) || (α+β == 0))
        (α-β)/2
    else
        (α^2 - β^2) * (2*n + α + β + 1) / (2 * (n+1) * (n + α + β + 1) * (2*n + α + β))
    end
end
function jacobi_rec_Cn(n::Int, α, β)
    (n + α) * (n + β) * (2*n + α + β + 2) / ((n+1) * (n + α + β + 1) * (2*n + α + β))
end

function jacobi_eval(n::Int, x, α, β)
    @assert n >= 0
    z0 = one(x)
    if n == 0
        return z0
    end
    z1 = jacobi_rec_An(0, α, β)*x + jacobi_rec_Bn(0, α, β)
    if n == 1
        return z1
    end
    z = z1
    for i in 2:n
        z = (jacobi_rec_An(i, α, β)*x + jacobi_rec_Bn(i, α, β))*z1 - jacobi_rec_Cn(i, α, β)*z0
        z0 = z1
        z1 = z
    end
    z
end


# Legendre
legendre_rec_An(n::Int, ::Type{T} = Float64) where T = (2*n+1)/T(n+1)
legendre_rec_Bn(n::Int, ::Type{T} = Float64) where T = zero(T)
legendre_rec_Cn(n::Int, ::Type{T} = Float64) where T = n/T(n+1)

# Ultraspherical
ultraspherical_rec_An(n::Int, λ) = 2(n+λ) / (n+1)
ultraspherical_rec_Bn(n::Int, λ) = zero(λ)
ultraspherical_rec_Cn(n::Int, λ) = (n+2λ-1) / (n+1)

# Chebyshev
chebyshev_1st_rec_An(n::Int, ::Type{T} = Float64) where T = n == 0 ? one(T) : T(2)
chebyshev_1st_rec_Bn(n::Int, ::Type{T} = Float64) where T = zero(T)
chebyshev_1st_rec_Cn(n::Int, ::Type{T} = Float64) where T = one(T)
chebyshev_2nd_rec_An(n::Int, ::Type{T} = Float64) where T = T(2)
chebyshev_2nd_rec_Bn(n::Int, ::Type{T} = Float64) where T = zero(T)
chebyshev_2nd_rec_Cn(n::Int, ::Type{T} = Float64) where T = one(T)

# Laguerre
laguerre_rec_An(n::Int, α::T) where T = -one(T)/(n+1)
laguerre_rec_Bn(n::Int, α) = (2n+α+1) / (n+1)
laguerre_rec_Cn(n::Int, α) = (n+α) / (n+1)

# Hermite
hermite_rec_An(n::Int, ::Type{T} = Float64) where T = T(2)
hermite_rec_Bn(n::Int, ::Type{T} = Float64) where T = zero(T)
hermite_rec_Cn(n::Int, ::Type{T} = Float64) where T = T(2n)




##############################
# DLMF section 18.7
# https://dlmf.nist.gov/18.7
##############################

"DLMF formula (18.7.1)."
ultraspherical_to_jacobi(n::Int, λ) = pochhammer(2λ, n) / pochhammer(λ+one(λ)/2, n)
"DLMF formula (18.7.2)."
jacobi_to_ultraspherical_(n::Int, α) = pochhammer(α+1, n) / pochhammer(2α+1, n)

"DLMF formula (18.7.3)."
chebyshevt_to_jacobi(n::Int, ::Type{T} = Float64) where T =
    1 / jacobi_eval(n, one(T), -one(T)/2, -one(T)/2)

"DLMF formula (18.7.4)."
chebyshevu_to_jacobi(n::Int, ::Type{T} = Float64) where T =
    (n+1_) / jacobi_eval(n, one(T), one(T)/2, one(T)/2)
