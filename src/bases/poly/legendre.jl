# legendre.jl

"""
A basis of Legendre polynomials on the interval `[-1,1]`. These classical
polynomials are orthogonal with respect to the weight function `w(x) = 1`.
"""
struct LegendrePolynomials{T} <: OPS{T,T}
    n   ::  Int
end

const LegendreSpan{A,S,T,D <: LegendrePolynomials} = Span{A,S,T,D}

name(b::LegendrePolynomials) = "Legendre OPS"

# Constructor with a default numeric type
LegendrePolynomials(n::Int, ::Type{T} = Float64) where {T} = LegendrePolynomials{T}(n)

instantiate(::Type{LegendrePolynomials}, n, ::Type{T}) where {T} = LegendrePolynomials{T}(n)

dict_promote_domaintype(b::LegendrePolynomials, ::Type{S}) where {S} = LegendrePolynomials{S}(b.n)

resize(b::LegendrePolynomials{T}, n) where {T} = LegendrePolynomials{T}(n)

support(b::LegendrePolynomials{T}) where {T} = ChebyshevInterval{T}()

#grid(b::LegendrePolynomials) = LegendreGrid(b.n)

first_moment(b::LegendrePolynomials{T}) where {T} = T(2)

jacobi_α(b::LegendrePolynomials{T}) where {T} = T(0)
jacobi_β(b::LegendrePolynomials{T}) where {T} = T(0)

weight(b::LegendrePolynomials{T}, x) where {T} = T(1)

function gramdiagonal!(result, ::LegendreSpan; options...)
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
