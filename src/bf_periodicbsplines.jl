immutable PeriodicBSplineBasis{K,T} <: SplineBasis{K,T}
  n   :: Int

  gramcolumn :: Array{T}
  dualgramcolumn :: Array{T}

  PeriodicBSplineBasis{T}(n::Int, degree, ::Type{T}) = new{degree,T}(n,gramcolumn(degree,n,T),dualgramcolumn(degree,n,T))
end

name(b::PeriodicBSplineBasis) = "Periodic squeezed B-splines of degree $(degree(b))"

left{K,T}(::PeriodicBSplineBasis{K,T}) = real(T)(0)

right{K,T}(::PeriodicBSplineBasis{K,T}) = real(T)(1)

include("util/bsplines.jl")

PeriodicBSplineBasis{T}(n::Int, K::Int, ::Type{T} = Float64) =   PeriodicBSplineBasis{K,T}(n, K, T)

import Base: ==
=={K1,K2,T1,T2}(b1::PeriodicBSplineBasis{K1,T1}, b2::PeriodicBSplineBasis{K2,T2}) = T1==T2 && K1==K2 && length(b1)==length(b2)
instantiate{T}(::Type{PeriodicBSplineBasis}, n::Int, ::Type{T}) = PeriodicBSplineBasis(n,3,T)

set_promote_eltype{K,T,S}(b::PeriodicBSplineBasis{K,T}, ::Type{S}) = PeriodicBSplineBasis(length(b),K, S)

resize{K,T}(b::PeriodicBSplineBasis{K,T}, n::Int) = PeriodicBSplineBasis{K,T}(n, degree(b), T)

has_grid(b::PeriodicBSplineBasis) = true

length(b::PeriodicBSplineBasis) = b.n

grid(b::PeriodicBSplineBasis) = PeriodicEquispacedGrid(length(b), left(b), right(b))

splinescaling(n) = sqrt(n)

# Indices of periodic splines naturally range from 0 to n-1
native_index(b::PeriodicBSplineBasis, idx::Int) = idx-1
linear_index(b::PeriodicBSplineBasis, idxn::Int) = idxn+1

stepsize{K,T}(b::PeriodicBSplineBasis{K,T}) = real(T)(1)/length(b)

period{K,T}(b::PeriodicBSplineBasis{K,T}) = real(T)(1)
# return only one interval, but because of periodicity two parts of this interval may lay in [0,1]
left{K}(b::PeriodicBSplineBasis{K}, j::Int) = (j - 1) * stepsize(b)

right{K}(b::PeriodicBSplineBasis{K}, j::Int) = (j - 1 + K + 1) * stepsize(b)

function in_support{K}(b::PeriodicBSplineBasis{K}, idx::Int, x)
	period = right(b) - left(b)

	A = left(b) <= x <= right(b)
	B = (left(b, idx) <= x <= right(b, idx)) || (left(b, idx) <= x-period <= right(b, idx)) ||
		(left(b, idx) <= x+period <= right(b, idx))
	A && B
end

function eval_element{K,T}(b::PeriodicBSplineBasis{K,T}, idx::Int, x)
  if !in_support(b, idx, x)
    return T(0)
  end
  splinescaling(length(b))*Cardinal_b_splines.evaluate_periodic_Bspline(K, length(b)*x-(idx-1), length(b), T)
end

function eval_expansion{K,T <: Number}(b::PeriodicBSplineBasis{K}, coef, x::T)
	i = interval(b, x)
	n = length(b)

	z = zero(T)
	for idxn = i:i+K+1
		idx = linear_index(b, mod(idxn,n))
		z = z + coef[idx] * eval_element(b, idx, x)
	end

	z
end


function gramcolumn(degree, n, T)
    result = zeros(n)
    for i in 1:degree+1
        I = quadgk(x->BasisFunctions.Cardinal_b_splines.evaluate_Bspline(degree, x, T)*BasisFunctions.Cardinal_b_splines.evaluate_Bspline(degree, x-(i-1), T),linspace(0,degree+i,degree+i+1)...)[1]

        result[i] = I
        i > 1 &&(result[n-(i-2)] = I)
    end
    splinescaling(n)^2/n*result
end

function dualgramcolumn(degree, n, T)
    primalcolumn = gramcolumn(degree, n, T)
    e1 = zeros(n); e1[1] = 1;
    d = 1./fft(primalcolumn)
    real(ifft(Diagonal(d)*fft(e1)))
end

function Gram{K,T}(b::PeriodicBSplineBasis{K,T}; options...)
  C = ComplexifyOperator(b)
  R = inv(C)
  b = set_promote_eltype(b,complex(T))
  F = FastFourierTransformFFTW(b, b; options...)
  iF = InverseFastFourierTransformFFTW(b, b; options...)
  R*iF*DiagonalOperator(BasisFunctions.fftw_operator(b,b,1:1,FFTW.MEASURE)*b.gramcolumn)*F*C
end

function DualGram(b::PeriodicBSplineBasis; options...)
  inv(Gram(b, options...))
end


grammatrix(b::PeriodicBSplineBasis) = full(Circulant(b.gramcolumn))
dualgrammatrix(b::PeriodicBSplineBasis) = full(Circulant(b.dualgramcolumn))

eval_dualelement{K,T}(b::PeriodicBSplineBasis{K,T}, idx::Int, x) = eval_expansion(b,circshift(b.dualgramcolumn,(idx-1)),x)

function innerproduct{K,T}(b::PeriodicBSplineBasis{K,T}, f::Function, idx::Int; options...)
  n = length(b)
  if idx > n-K
    quadgk(x->b[idx](x)*f(x), linspace(left(b, idx), right(b), n-idx+2)...; options...)[1] +
    quadgk(x->b[idx](x)*f(x), linspace(left(b), right(b, idx), K+idx-n+1)...; options...)[1]
  else
    quadgk(x->b[idx](x)*f(x), linspace(left(b,idx), right(b,idx), K+2)...; options...)[1]
  end
end
