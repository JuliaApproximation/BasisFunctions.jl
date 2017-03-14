# translates_of_bsplines.jl

"""
  Basis consisting of dilated, translated, and periodized cardinal B splines on the interval [0,1].
"""
immutable BSplineTranslatesBasis{K,T} <: CompactPeriodicSetOfTranslates{T}
  n               :: Int
  a               :: T
  b               :: T
  fun             :: Function
end

degree{K}(b::BSplineTranslatesBasis{K}) = K
# BSplineTranslatesBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64) =
#     BSplineTranslatesBasis{DEGREE,T}(n, T(0), T(1), x->sqrt(T(n))*Cardinal_b_splines.evaluate_periodic_Bspline(DEGREE, n*x, n, T))
# BSplineTranslatesBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64) =
#     BSplineTranslatesBasis{DEGREE,T}(n, T(0), T(1), x->T(n)*Cardinal_b_splines.evaluate_periodic_Bspline(DEGREE, n*x, n, T))
BSplineTranslatesBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64) =
    BSplineTranslatesBasis{DEGREE,T}(n, T(0), T(1), x->Cardinal_b_splines.evaluate_periodic_Bspline(DEGREE, n*x, n, T))

length_compact_support(b::BSplineTranslatesBasis) = stepsize(b)*(degree(b)+1)

=={K1,K2,T1,T2}(b1::BSplineTranslatesBasis{K1,T1}, b2::BSplineTranslatesBasis{K2,T2}) = T1==T2 && K1==K2 && length(b1)==length(b2)

instantiate{T}(::Type{BSplineTranslatesBasis}, n::Int, ::Type{T}) = BSplineTranslatesBasis(n,3,T)

set_promote_eltype{K,T,S}(b::BSplineTranslatesBasis{K,T}, ::Type{S}) = BSplineTranslatesBasis(length(b),K, S)

resize{K,T}(b::BSplineTranslatesBasis{K,T}, n::Int) = BSplineTranslatesBasis(n, degree(b), T)

Gram(b::BSplineTranslatesBasis; options...) = CirculantOperator(b, b, primalgramcolumn(b; options...); options...)
# TODO find an explination for this (splines are no Chebyshev system)
# For the B spline with degree 1 (hat functions) the MidpointEquispacedGrid does not lead to evaluation_matrix that is non singular
compatible_grid{K}(b::BSplineTranslatesBasis{K}, grid::MidpointEquispacedGrid) = iseven(K) &&
    (1+(left(b) - left(grid))≈1) && (1+(right(b) - right(grid))≈1) && (length(b)==length(grid))
compatible_grid{K}(b::BSplineTranslatesBasis{K}, grid::PeriodicEquispacedGrid) = isodd(K) &&
    (1+(left(b) - left(grid))≈1) && (1+(right(b) - right(grid))≈1) && (length(b)==length(grid))
# we use a PeriodicEquispacedGrid in stead
grid{K}(b::BSplineTranslatesBasis{K}) = isodd(K) ?
    PeriodicEquispacedGrid(length(b), left(b), right(b)) :
    MidpointEquispacedGrid(length(b), left(b), right(b))

# function restriction_operator(::BSplineTranslatesBasis, ::BSplineTranslatesBasis; options...)
#   println("Method does not exists for splines of different degrees")
#   throw(InexactError())
# end
#
# function extension_operator(::BSplineTranslatesBasis, ::BSplineTranslatesBasis; options...)
#   println("Method does not exists for splines of different degrees")
#   throw(InexactError())
# end

function extension_operator{K,T}(s1::BSplineTranslatesBasis{K,T}, s2::BSplineTranslatesBasis{K,T}; options...)
  @assert 2*length(s1) == length(s2)
  _binomial_circulant(s2)*IndexExtensionOperator(s1, s2, 1:2:length(s2))
end

function _binomial_circulant{K,T}(s::BSplineTranslatesBasis{K,T})
  c = zeros(T, length(s))
  for k in 1:K+2
    c[k] = binomial(K+1, k-1)
  end
  T(1)/(1<<(degree(s)))*CirculantOperator(s, c)
end

# The calculation done in this function is equivalent to finding the pseudoinverse of the extension_operator.
# TODO find out why exactly.
function restriction_operator{K,T}(s1::BSplineTranslatesBasis{K,T}, s2::BSplineTranslatesBasis{K,T}; options...)
    @assert length(s1) == 2*length(s2)
    # t = zeros(s2)
    # t[1] = 1
    # IndexRestrictionOperator(s1,s2,1:2:length(s1))*CirculantOperator(s1, matrix(extension_operator(s2, s1; options...))'\t)'
    r = BasisFunctions._binomial_circulant(s1)
    e = BasisFunctions.eigenvalues(r)
    n = length(e)
    d = similar(e)
    d[1] = T(2)/e[1]
    d[n>>1+1] = 0
    iseven(n>>1) && (d[n>>2+1] = 1/e[n>>2+1]; d[n-n>>2+1] = 1/e[n-n>>2+1])
    n>>2
    I = 2:n>>2
    quarter = n>>2
    isodd(n>>1) && (quarter = n>>2+1)
    for i in 2:quarter
      x = e[i]
      y = e[n>>1+2-i]
      a = 2*(abs(x)^2)/(abs(y)^2+abs(x)^2)
      d[i] = a/x
      d[n>>1+2-i] = (2-a)/y
      d[n+2-i] = conj(a/x)
      d[n>>1+i] = conj((2-a)/y)
    end
    IndexRestrictionOperator(s1,s2,1:2:length(s1))*CirculantOperator(s1, s1, PseudoDiagonalOperator(d))
end



function restriction_operator2{K,T}(s1::BSplineTranslatesBasis{K,T}, s2::BSplineTranslatesBasis{K,T}; options...)
    @assert length(s1) == 2*length(s2)
    inv(evaluation_operator(s2; options...))*evaluation_operator(s1, grid(s2); options...)
end
