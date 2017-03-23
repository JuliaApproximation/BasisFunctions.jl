# translates_of_bsplines.jl
abstract PeriodicBSplineBasis{K,T} <: CompactPeriodicSetOfTranslates{T}

degree{K}(b::PeriodicBSplineBasis{K}) = K

Gram(b::PeriodicBSplineBasis; options...) = CirculantOperator(b, b, primalgramcolumn(b; options...); options...)

function _binomial_circulant{K,T}(s::PeriodicBSplineBasis{K,T})
  c = zeros(T, length(s))
  for k in 1:K+2
    c[k] = binomial(K+1, k-1)
  end
  T(1)/(1<<(degree(s)))*CirculantOperator(s, c)
end

function bspline_extension_operator{K,T}(s1::PeriodicBSplineBasis{K,T}, s2::PeriodicBSplineBasis{K,T}; options...)
  @assert 2*length(s1) == length(s2)
  _binomial_circulant(s2)*IndexExtensionOperator(s1, s2, 1:2:length(s2))
end

# The calculation done in this function is equivalent to finding the pseudoinverse of the extension_operator.
function bspline_restriction_operator{K,T}(s1::PeriodicBSplineBasis{K,T}, s2::PeriodicBSplineBasis{K,T}; options...)
    @assert length(s1) == 2*length(s2)
    r = BasisFunctions._binomial_circulant(s1)
    e = BasisFunctions.eigenvalues(r)
    n = length(e)
    d = similar(e)
    eabs = map(abs, e)
    for i in 1:n>>1
      a = 2*(eabs[i]^2)/(eabs[i+n>>1]^2+eabs[i]^2)
      d[i] = a
      d[i+n>>1] = (2-a)
    end
    d = d ./ e
    d[map(isnan,d)] = 0

    IndexRestrictionOperator(s1,s2,1:2:length(s1))*CirculantOperator(s1, s1, PseudoDiagonalOperator(d))
end

# TODO check the properties of this one
function bspline_restriction_operator2{K,T}(s1::PeriodicBSplineBasis{K,T}, s2::PeriodicBSplineBasis{K,T}; options...)
    @assert length(s1) == 2*length(s2)
    inv(evaluation_operator(s2; options...))*evaluation_operator(s1, grid(s2); options...)
end

# function restriction_operator{K,T}(s1::PeriodicBSplineBasis{K,T}, s2::PeriodicBSplineBasis{K,T}; options...)
#     @assert length(s1) == 2*length(s2)
#     t = zeros(s2)
#     t[1] = 1
#     IndexRestrictionOperator(s1,s2,1:2:length(s1))*CirculantOperator(s1, matrix(extension_operator(s2, s1; options...))'\t)'
# end

"""
  Basis consisting of dilated, translated, and periodized cardinal B splines on the interval [0,1].
"""
immutable BSplineTranslatesBasis{K,T} <: PeriodicBSplineBasis{K,T}
  n               :: Int
  a               :: T
  b               :: T
  fun             :: Function
end

BSplineTranslatesBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64) =
    BSplineTranslatesBasis{DEGREE,T}(n, T(0), T(1), x->Cardinal_b_splines.evaluate_periodic_Bspline(DEGREE, n*x, n, T))

left_of_compact_function{K,T}(b::BSplineTranslatesBasis{K,T}) = T(0)

right_of_compact_function{K,T}(b::BSplineTranslatesBasis{K,T}) = stepsize(b)*(degree(b)+1)


=={K1,K2,T1,T2}(b1::BSplineTranslatesBasis{K1,T1}, b2::BSplineTranslatesBasis{K2,T2}) = T1==T2 && K1==K2 && length(b1)==length(b2)

instantiate{T}(::Type{BSplineTranslatesBasis}, n::Int, ::Type{T}) = BSplineTranslatesBasis(n,3,T)

set_promote_eltype{K,T,S}(b::BSplineTranslatesBasis{K,T}, ::Type{S}) = BSplineTranslatesBasis(length(b),K, S)

resize{K,T}(b::BSplineTranslatesBasis{K,T}, n::Int) = BSplineTranslatesBasis(n, degree(b), T)

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

# TODO  can be added to PeriodicBSplineBasis in julia 0.6
# extension_operator{K,T,B<:PeriodicBSplineBasis{K,T}}(s1::B, s2::B; options...) =
extension_operator{K,T}(s1::BSplineTranslatesBasis{K,T}, s2::BSplineTranslatesBasis{K,T}; options...) =
    bspline_extension_operator(s1, s2; options...)

restriction_operator{K,T}(s1::BSplineTranslatesBasis{K,T}, s2::BSplineTranslatesBasis{K,T}; options...) =
    bspline_restriction_operator(s1, s2; options...)

"""
  Basis consisting of symmetric, dilated, translated, and periodized cardinal B splines on the interval [0,1].
"""
immutable SymBSplineTranslatesBasis{K,T} <: PeriodicBSplineBasis{K,T}
  n               :: Int
  a               :: T
  b               :: T
  fun             :: Function
end

SymBSplineTranslatesBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64) =
    SymBSplineTranslatesBasis{DEGREE,T}(n, T(0), T(1), x->Cardinal_b_splines.evaluate_symmetric_periodic_Bspline(DEGREE, n*x, n, T))

left_of_compact_function{K,T}(b::SymBSplineTranslatesBasis{K,T}) = -right_of_compact_function(b)

right_of_compact_function{K,T}(b::SymBSplineTranslatesBasis{K,T}) = stepsize(b)*(degree(b)+1)/2


=={K1,K2,T1,T2}(b1::SymBSplineTranslatesBasis{K1,T1}, b2::SymBSplineTranslatesBasis{K2,T2}) = T1==T2 && K1==K2 && length(b1)==length(b2)

instantiate{T}(::Type{SymBSplineTranslatesBasis}, n::Int, ::Type{T}) = SymBSplineTranslatesBasis(n,3,T)

set_promote_eltype{K,T,S}(b::SymBSplineTranslatesBasis{K,T}, ::Type{S}) = SymBSplineTranslatesBasis(length(b),K, S)

resize{K,T}(b::SymBSplineTranslatesBasis{K,T}, n::Int) = SymBSplineTranslatesBasis(n, degree(b), T)

# TODO  can be added to PeriodicBSplineBasis in julia 0.6
# extension_operator{K,T,B<:PeriodicBSplineBasis{K,T}}(s1::B, s2::B; options...) =
extension_operator{K,T}(s1::SymBSplineTranslatesBasis{K,T}, s2::SymBSplineTranslatesBasis{K,T}; options...) =
    bspline_extension_operator(s1, s2; options...)

restriction_operator{K,T}(s1::SymBSplineTranslatesBasis{K,T}, s2::SymBSplineTranslatesBasis{K,T}; options...) =
    bspline_restriction_operator(s1, s2; options...)
