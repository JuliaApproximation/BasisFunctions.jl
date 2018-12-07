
"""
A basis of Legendre polynomials on the interval `[-1,1]`. These classical
polynomials are orthogonal with respect to the weight function `w(x) = 1`.
"""
struct LegendrePolynomials{T} <: OPS{T,T}
    n   ::  Int
end



name(b::LegendrePolynomials) = "Legendre OPS"

LegendrePolynomials(n::Int) = LegendrePolynomials{Float64}(n)

instantiate(::Type{LegendrePolynomials}, n, ::Type{T}) where {T} = LegendrePolynomials{T}(n)

similar(b::LegendrePolynomials, ::Type{T}, n::Int) where {T} = LegendrePolynomials{T}(n)

support(b::LegendrePolynomials{T}) where {T} = ChebyshevInterval{T}()

first_moment(b::LegendrePolynomials{T}) where {T} = T(2)

jacobi_α(b::LegendrePolynomials{T}) where {T} = T(0)
jacobi_β(b::LegendrePolynomials{T}) where {T} = T(0)

weight(b::LegendrePolynomials{T}, x) where {T} = T(1)

function gramdiagonal!(result, ::LegendrePolynomials; options...)
    T = eltype(result)
    for i in 1:length(result)
        result[i] = T(2//(2(i-1)+1))
    end
end

# See DLMF, Table 18.9.1
# http://dlmf.nist.gov/18.9#i
rec_An(b::LegendrePolynomials{T}, n::Int) where {T} = T(2*n+1)/T(n+1)

rec_Bn(b::LegendrePolynomials{T}, n::Int) where {T} = zero(T)

rec_Cn(b::LegendrePolynomials{T}, n::Int) where {T} = T(n)/T(n+1)
