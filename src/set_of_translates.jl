"""
  Set consisting of n translates of a function.
"""
abstract SetOfTranslates{T} <: FunctionSet1d{T}

is_biorthogonal(::SetOfTranslates) = true

name(b::SetOfTranslates) = "Set of translates of function $(fun(b))"

fun(b::SetOfTranslates) = b.fun

# Indices set of translates naturally range from 0 to n-1
native_index(b::SetOfTranslates, idx::Int) = idx-1
linear_index(b::SetOfTranslates, idxn::Int) = idxn+1

"""
  Set consisting of n translates of a periodic function.

  The set can be written as ``\left\{T_k f\right\}_{k=0}^n``, where ``T_k f(x) = f(x-p/n)``.
  ``p`` is the period of the set, ``n`` is the number of elements.
"""
abstract PeriodicSetOfTranslates{T} <: SetOfTranslates{T}

left{T}(set::PeriodicSetOfTranslates{T}) = real(T)(set.a)

right{T}(set::PeriodicSetOfTranslates{T}) = real(T)(set.b)

left(set::PeriodicSetOfTranslates, j::Int) = left(set)

right(set::PeriodicSetOfTranslates, j::Int) = right(set)

has_grid(::PeriodicSetOfTranslates) = true

grid(set::PeriodicSetOfTranslates) = MidpointEquispacedGrid(length(set), left(set), right(set))

period(set::PeriodicSetOfTranslates) = right(set)-left(set)

stepsize(set::PeriodicSetOfTranslates) = period(set)/length(set)
"Return the index of the interval between two knots in which x lies, starting from index 0."
interval(b::PeriodicSetOfTranslates, x) = round(Int, floor( (x-left(b))/stepsize(b) ))

# All inner products between elements of PeriodicSetOfTranslates are known by the first column of the (circulant) gram matrix.
# The dual gram matrix is the inverse of the gram matrix and this function returns its first column.
dualgramcolumn(b::PeriodicSetOfTranslates) = dualgramcolumn(primalgramcolumn(b))

eval_element{T}(b::PeriodicSetOfTranslates{T}, idx::Int, x) = fun(T(x)-native_index(b, idx)*stepsize(b))

eval_dualelement{T}(b::PeriodicSetOfTranslates{T}, idx::Int, x) = eval_expansion(b, circshift(dualgramcolumn(b),native_index(b, idx)), x)

grammatrix(b::PeriodicSetOfTranslates) = full(Circulant(primalgramcolumn(b)))

dualgrammatrix(b::PeriodicSetOfTranslates) = full(Circulant(dualgramcolumn(b)))

# All inner products between elements of PeriodicSetOfTranslates are known by the first column of the (circulant) gram matrix.
function primalgramcolumn{T}(set::PeriodicSetOfTranslates{T}; options...)
  n = length(set)
  result = zeros(real(T), n)
  for i in 1:length(result)
    result[i] = dot(set, 1, i; options...)
  end
  result
end

dualgramcolumn(set::PeriodicSetOfTranslates; options...) =
    dualgramcolumn(primalgramcolumn(set; options...))

# The inverse of a circulant matrix is calculated in O(n logn) by use of an fft.
function dualgramcolumn{T}(primalgramcolumn::Array{T,1})
    n = length(primalgramcolumn)
    e1 = zeros(T,n); e1[1] = 1;
    d = 1./fft(primalgramcolumn)
    real(ifft(Diagonal(d)*fft(e1)))
end

"""
  Set consisting of n translates of a compact and periodic function.

  The support of the function is [0,c], where c∈R and 0 < c < ∞.
"""
abstract CompactPeriodicSetOfTranslates{T} <: PeriodicSetOfTranslates{T}

"""
  Support of the function of a CompactPeriodicSetOfTranslates.
"""
function length_compact_support end

left(b::CompactPeriodicSetOfTranslates, j::Int) = native_index(j) * stepsize(b)

right(b::CompactPeriodicSetOfTranslates, j::Int) = (native_index(j)+length_compact_support(b)) * stepsize(b)

# return whether x lays in the support of the idxth element of set. x should lay in the support of the set.
function in_support{K}(set::CompactPeriodicSetOfTranslates{K}, idx::Int, x)
	per = period(set)
	A = left(set) <= x <= right(set)
	B = (left(set, idx) <= x <= right(set, idx)) || (left(set, idx) <= x-per <= right(set, idx)) ||
		(left(set, idx) <= x+per <= right(set, idx))
	A && B
end

eval_element{T}(b::CompactPeriodicSetOfTranslates{T}, idx::Int, x) = !in_support(b, idx, x) ?
  T(0) :
  fun(T(x)-native_index(idx)*stepsize(b))

"""

"""
immutable BSplineTranslatesBasis{K,T} <: SetOfTranslates{T}
  n               :: Int
  a               :: T
  b               :: T
  fun             :: Function
end

is_biorthogonal(::PeriodicBSplineBasis) = true

# include("util/bsplines.jl")
BSplineTranslatesBasis{T}(n::Int, K::Int, ::Type{T} = Float64) = BSplineTranslatesBasis{K,T}(n, T(0), T(1), x->sqrt(T(n))*Cardinal_b_splines.evaluate_periodic_Bspline(degree, n*x, n, T), T)

import Base: ==
=={K1,K2,T1,T2}(b1::BSplineTranslatesBasis{K1,T1}, b2::BSplineTranslatesBasis{K2,T2}) = T1==T2 && K1==K2 && length(b1)==length(b2)

instantiate{T}(::Type{BSplineTranslatesBasis}, n::Int, ::Type{T}) = BSplineTranslatesBasis(n,3,T)

set_promote_eltype{K,T,S}(b::BSplineTranslatesBasis{K,T}, ::Type{S}) = BSplineTranslatesBasis(length(b),K, S)

resize{K,T}(b::BSplineTranslatesBasis{K,T}, n::Int) = BSplineTranslatesBasis{K,T}(n, degree(b), T)

function eval_expansion{K,T <: Number}(b::BSplineTranslatesBasis{K}, coef, x::T)
	i = interval(b, x)
	n = length(b)
	z = zero(T)
	for idxn = i-K:i
		idx = linear_index(b, mod(idxn,n))
		z = z + coef[idx] * eval_element(b, idx, x)
	end
	z
end

function Gram{K,T}(b::PeriodicBSplineBasis{K,T}; options...)
  C = ComplexifyOperator(b)
  R = inv(C)
  b = dest(C)
  F = forward_fourier_operator(b, b, eltype(b); options...)
  iF = backward_fourier_operator(b, b, eltype(b); options...)
  R*iF*DiagonalOperator(BasisFunctions.fftw_operator(b,b,1:1,FFTW.MEASURE)*b.gramcolumn)*F*C
end



function innerproduct{K,T}(basis::PeriodicBSplineBasis{K,T}, f::Function, idx::Int, a::Real, b::Real; options...)
  n = length(basis)
  if idx > n-K
    quadgkwrap(x->basis[idx](x)*f(x), nodes(left(basis, idx), b, n); options...) +
    quadgkwrap(x->basis[idx](x)*f(x), nodes(a, right(basis,idx)-period(basis), n); options...)
  else
    quadgkwrap(x->basis[idx](x)*f(x), nodes(a, b, n); options...)
  end
end

function nodes(left::Real, right::Real, n::Int)
    if left >= right
        return [NaN]
    end
    left *= n
    right *= n
    low = ceil(Int,left)
    up = floor(Int,right)
    l = linspace(low, up, up-low+1)
    if up ≈ right
        if low ≈ left
            result = [l...]
        else
            result = [left, l...]
        end
    else
        if low ≈ left
            result = [l..., right]
        else
            result = [left, l..., right]
        end
    end
    result/n
end

quadgkwrap(f::Function, range; options...) =
  length(range) == 1? 0. : quadgk(f, range...; options...)[1]
