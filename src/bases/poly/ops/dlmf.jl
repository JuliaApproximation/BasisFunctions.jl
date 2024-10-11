# Formulas from the Digital Library of Mathematical Functions.

"""
Pochhammer's Symbol.

See [DLMF](https://dlmf.nist.gov/5.2#iii) formula (5.2.4).
"""
pochhammer(a, n) = n == 0 ? one(a) : prod(a+i for i in 0:n-1)
# or in terms of the gamma function (DLMF, 5.2.5):  gamma(n+a) / gamma(a)

##############################
# DLMF Table 18.3.1
# https://dlmf.nist.gov/18.3
##############################
# Formulas for squared norm hn and leading order coefficient kn

# The formula for An and A0
dlmf_18d3d1_A(α, β, n) = 2^(α+β+1)*gamma(n+α+1)*gamma(n+β+1) / (factorial(n)*(2n+α+β+1)*gamma(n+α+β+1))
dlmf_18d3d1_A0(α, β) = 2^(α+β+1)*gamma(α+1)*gamma(β+1) / gamma(α+β+2)

jacobi_hn(α, β, n) = n == 0 ? dlmf_18d3d1_A0(α, β) : dlmf_18d3d1_A(α, β, n)
jacobi_kn(α, β, n) = pochhammer(n+α+β+1, n) / (2^n * factorial(n))

ultraspherical_hn(λ::T, n) where T = 2^(1-2λ)*T(π)*gamma(n+2λ) / ((n+λ)*gamma(λ)^2*factorial(n))
ultraspherical_kn(λ, n) = 2^n*pochhammer(λ, n) / factorial(n)

chebyshev_1st_hn(::Type{T}, n) where T = n == 0 ? T(π) : T(π)/2
chebyshev_1st_kn(::Type{T}, n) where T = n == 0 ? one(T) : T(2)^(n-1)

chebyshev_2nd_hn(::Type{T}, n) where T = T(π)/2
chebyshev_2nd_kn(::Type{T}, n) where T = T(2)^n

chebyshev_3rd_hn(::Type{T}, n) where T = T(π)
chebyshev_3rd_kn(::Type{T}, n) where T = T(2)^n

chebyshev_4rd_hn(::Type{T}, n) where T = T(π)
chebyshev_4rd_kn(::Type{T}, n) where T = T(2)^n

legendre_hn(::Type{T}, n) where T = 2 / T(2n+1)
legendre_kn(::Type{T}, n) where T = 2^n*pochhammer(one(T)/2, n) / factorial(n)

laguerre_hn(α, n) = gamma(n+α+1) / factorial(n)
laguerre_kn(α::T, n) where T = (-one(T))^n / factorial(n)

hermite_hn(::Type{T}, n) where T = sqrt(T(π)) * 2^n * factorial(n)
hermite_kn(::Type{T}, n) where T = T(2)^n


##############################
# DLMF Table 18.9.1
# http://dlmf.nist.gov/18.9#i
##############################
# Recurrence coefficients in the form
# p_{n+1}(x) = (A_n x + B_n) p_n(x) - C_n p_{n-1}(x)
# with initial values
# p_0(x) = 1 and p_1(x) = A_0x + B_0


# Jacobi, (18.9.2)
function jacobi_rec_An(α, β, n)
    if (n == 0) && (α + β + 1 == 0)
        (α+β)/2+1
    else
        (2*n + α + β + 1) * (2n + α + β + 2) / (2 * (n+1) * (n + α + β + 1))
    end
end
function jacobi_rec_Bn(α, β, n)
    if (n == 0) && ((α + β + 1 == 0) || (α+β == 0))
        (α-β)/2
    else
        (α^2 - β^2) * (2*n + α + β + 1) / (2 * (n+1) * (n + α + β + 1) * (2*n + α + β))
    end
end
function jacobi_rec_Cn(α, β, n)
    (n + α) * (n + β) * (2*n + α + β + 2) / ((n+1) * (n + α + β + 1) * (2*n + α + β))
end

# Legendre
legendre_rec_An(::Type{T}, n) where T = (2*n+1)/T(n+1)
legendre_rec_Bn(::Type{T}, n) where T = zero(T)
legendre_rec_Cn(::Type{T}, n) where T = n/T(n+1)

# Ultraspherical
ultraspherical_rec_An(λ, n) = 2(n+λ) / (n+1)
ultraspherical_rec_Bn(λ, n) = zero(λ)
ultraspherical_rec_Cn(λ, n) = (n+2λ-1) / (n+1)

# Chebyshev
chebyshev_1st_rec_An(::Type{T}, n) where T = n == 0 ? one(T) : T(2)
chebyshev_1st_rec_Bn(::Type{T}, n) where T = zero(T)
chebyshev_1st_rec_Cn(::Type{T}, n) where T = one(T)
chebyshev_2nd_rec_An(::Type{T}, n) where T = T(2)
chebyshev_2nd_rec_Bn(::Type{T}, n) where T = zero(T)
chebyshev_2nd_rec_Cn(::Type{T}, n) where T = one(T)

# Laguerre
laguerre_rec_An(α::T, n) where T = -one(T)/(n+1)
laguerre_rec_Bn(α, n) = (2n+α+1) / (n+1)
laguerre_rec_Cn(α, n) = (n+α) / (n+1)

# Hermite
hermite_rec_An(::Type{T}, n) where T = T(2)
hermite_rec_Bn(::Type{T}, n) where T = zero(T)
hermite_rec_Cn(::Type{T}, n) where T = T(2n)
