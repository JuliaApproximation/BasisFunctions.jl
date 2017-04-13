# translates_of_bsplines.jl
abstract PeriodicBSplineBasis{K,T} <: CompactPeriodicSetOfTranslates{T}

degree{K}(b::PeriodicBSplineBasis{K}) = K

Gram(b::PeriodicBSplineBasis; options...) = CirculantOperator(b, b, primalgramcolumn(b; options...); options...)

function bspline_extension_operator{K,T}(s1::PeriodicBSplineBasis{K,T}, s2::PeriodicBSplineBasis{K,T}; options...)
  @assert 2*length(s1) == length(s2)
  _binomial_circulant(s2)*IndexExtensionOperator(s1, s2, 1:2:length(s2))
end

# The calculation done in this function is equivalent to finding the pseudoinverse of the bspline_extension_operator.
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
# function bspline_restriction_operator2{K,T}(s1::PeriodicBSplineBasis{K,T}, s2::PeriodicBSplineBasis{K,T}; options...)
#     @assert length(s1) == 2*length(s2)
#     inv(evaluation_operator(s2; options...))*evaluation_operator(s1, grid(s2); options...)
# end

# function restriction_operator{K,T}(s1::PeriodicBSplineBasis{K,T}, s2::PeriodicBSplineBasis{K,T}; options...)
#     @assert length(s1) == 2*length(s2)
#     t = zeros(s2)
#     t[1] = 1
#     IndexRestrictionOperator(s1,s2,1:2:length(s1))*CirculantOperator(s1, matrix(extension_operator(s2, s1; options...))'\t)'
# end

"""
  Basis consisting of dilated, translated, and periodized cardinal B splines on the interval [0,1].
"""
immutable BSplineTranslatesBasis{K,T,SCALED} <: PeriodicBSplineBasis{K,T}
  n               :: Int
  a               :: T
  b               :: T
  fun             :: Function
end

BSplineTranslatesBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64; scaled = false) = scaled?
    BSplineTranslatesBasis{DEGREE,T,true}(n, T(0), T(1), x->sqrt(n)*Cardinal_b_splines.evaluate_periodic_Bspline(DEGREE, n*x, n, real(T))) :
    BSplineTranslatesBasis{DEGREE,T,false}(n, T(0), T(1), x->Cardinal_b_splines.evaluate_periodic_Bspline(DEGREE, n*x, n, real(T)))

name(b::BSplineTranslatesBasis) = name(typeof(b))*" (B spline of degree $(degree(b)))"

left_of_compact_function{K,T}(b::BSplineTranslatesBasis{K,T})::real(T) = real(T)(0)

right_of_compact_function{K,T}(b::BSplineTranslatesBasis{K,T})::real(T) = stepsize(b)*real(T)(degree(b)+1)


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

function _binomial_circulant{K,T,SCALED}(s::BSplineTranslatesBasis{K,T,SCALED})
  c = zeros(T, length(s))
  for k in 1:K+2
    c[k] = binomial(K+1, k-1)
  end
  if SCALED
    sqrt(T(2))/(1<<(degree(s)))*CirculantOperator(s, c)
  else
    T(1)/(1<<(degree(s)))*CirculantOperator(s, c)
  end
end

# TODO extension_operator/restriction_operator can be added to PeriodicBSplineBasis in julia 0.6
# extension_operator{K,T,B<:PeriodicBSplineBasis{K,T}}(s1::B, s2::B; options...) =
extension_operator{K,T}(s1::BSplineTranslatesBasis{K,T}, s2::BSplineTranslatesBasis{K,T}; options...) =
    bspline_extension_operator(s1, s2; options...)

restriction_operator{K,T}(s1::BSplineTranslatesBasis{K,T}, s2::BSplineTranslatesBasis{K,T}; options...) =
    bspline_restriction_operator(s1, s2; options...)

"""
  Basis consisting of symmetric, dilated, translated, and periodized cardinal B splines on the interval [0,1].

  There degree should be odd to use extension or restriction.
"""
immutable SymBSplineTranslatesBasis{K,T} <: PeriodicBSplineBasis{K,T}
  n               :: Int
  a               :: T
  b               :: T
  fun             :: Function
end

SymBSplineTranslatesBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64) =
    SymBSplineTranslatesBasis{DEGREE,T}(n, T(0), T(1), x->Cardinal_b_splines.evaluate_symmetric_periodic_Bspline(DEGREE, n*x, n, real(T)))

name(b::SymBSplineTranslatesBasis) = name(typeof(b))*" (symmetric B spline of degree $(degree(b)))"

left_of_compact_function{K,T}(b::SymBSplineTranslatesBasis{K,T})::real(T) = -right_of_compact_function(b)

right_of_compact_function{K,T}(b::SymBSplineTranslatesBasis{K,T})::real(T) = stepsize(b)*real(T)((degree(b)+1))/2


=={K1,K2,T1,T2}(b1::SymBSplineTranslatesBasis{K1,T1}, b2::SymBSplineTranslatesBasis{K2,T2}) = T1==T2 && K1==K2 && length(b1)==length(b2)

instantiate{T}(::Type{SymBSplineTranslatesBasis}, n::Int, ::Type{T}) = SymBSplineTranslatesBasis(n,3,T)

set_promote_eltype{K,T,S}(b::SymBSplineTranslatesBasis{K,T}, ::Type{S}) = SymBSplineTranslatesBasis(length(b),K, S)

resize{K,T}(b::SymBSplineTranslatesBasis{K,T}, n::Int) = SymBSplineTranslatesBasis(n, degree(b), T)

function _binomial_circulant{K,T}(s::SymBSplineTranslatesBasis{K,T})
  if iseven(degree(s))
    warn("Extension and restriction work with odd degrees only.")
    throw(MethodError())
  end
  c = zeros(T, length(s))
  c[1] = binomial(K+1, (K+1)>>1)
  for (i,k) in enumerate((K+1)>>1+1:K+1)
    c[i+1] = binomial(K+1, k)
    c[end+1-i] = binomial(K+1, k)
  end
  T(1)/(1<<(degree(s)))*CirculantOperator(s, c)
end

# TODO extension_operator/restriction_operator can be added to PeriodicBSplineBasis in julia 0.6
# extension_operator{K,T,B<:PeriodicBSplineBasis{K,T}}(s1::B, s2::B; options...) =
extension_operator{K,T}(s1::SymBSplineTranslatesBasis{K,T}, s2::SymBSplineTranslatesBasis{K,T}; options...) =
    bspline_extension_operator(s1, s2; options...)

restriction_operator{K,T}(s1::SymBSplineTranslatesBasis{K,T}, s2::SymBSplineTranslatesBasis{K,T}; options...) =
    bspline_restriction_operator(s1, s2; options...)

"""
  Basis consisting of orthonormal basis function in the spline space of degree K.
"""
immutable OrthonormalSplineBasis{K,T} <: LinearCombinationOfPeriodicSetOfTranslates{BSplineTranslatesBasis,T}
  superset     ::    BSplineTranslatesBasis{K,T}
  coefficients ::    Array{T,1}
  OrthonormalSplineBasis{K,T}(b::BSplineTranslatesBasis{K,T}; options...) =
    new(b, coeffs_in_other_basis(b, OrthonormalSplineBasis; options...))
end

degree{K,T}(::OrthonormalSplineBasis{K,T}) = K

superset(b::OrthonormalSplineBasis) = b.superset
coeffs(b::OrthonormalSplineBasis) = b.coefficients

OrthonormalSplineBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64; options...) =
    OrthonormalSplineBasis{DEGREE,T}(BSplineTranslatesBasis(n,DEGREE,T); options...)

name(b::OrthonormalSplineBasis) = name(b.superset)*" (orthonormalized)"

instantiate{T}(::Type{OrthonormalSplineBasis}, n::Int, ::Type{T}) = OrthonormalSplineBasis(n,3,T)

set_promote_eltype{K,T,S}(b::OrthonormalSplineBasis{K,T}, ::Type{S}) = OrthonormalSplineBasis(length(b),K, S)

resize{K,T}(b::OrthonormalSplineBasis{K,T}, n::Int) = OrthonormalSplineBasis(n, degree(b), T)

Gram(b::OrthonormalSplineBasis) = IdentityOperator(b, b)

change_of_basis{B<:OrthonormalSplineBasis}(b::BSplineTranslatesBasis, ::Type{B}; options...) = sqrt(DualGram(b; options...))


"""
  Basis consisting of orthonormal (w.r.t. a discrete inner product) basis function in the spline space of degree K.
"""
immutable DiscreteOrthonormalSplineBasis{K,T} <: LinearCombinationOfPeriodicSetOfTranslates{BSplineTranslatesBasis,T}
  superset     ::    BSplineTranslatesBasis{K,T}
  coefficients ::    Array{T,1}

  oversampling ::   T
  DiscreteOrthonormalSplineBasis{K,T}(b::BSplineTranslatesBasis{K,T}; oversampling=1, options...) =
    new(b, coeffs_in_other_basis(b, DiscreteOrthonormalSplineBasis; oversampling=oversampling, options...), oversampling)
end

degree{K,T}(::DiscreteOrthonormalSplineBasis{K,T}) = K

superset(b::DiscreteOrthonormalSplineBasis) = b.superset
coeffs(b::DiscreteOrthonormalSplineBasis) = b.coefficients

==(b1::DiscreteOrthonormalSplineBasis, b2::DiscreteOrthonormalSplineBasis) =
    superset(b1)==superset(b2) && coeffs(b1) ≈ coeffs(b2) && b1.oversampling == b2.oversampling

DiscreteOrthonormalSplineBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64; options...) =
    DiscreteOrthonormalSplineBasis{DEGREE,T}(BSplineTranslatesBasis(n,DEGREE,T); options...)

name(b::DiscreteOrthonormalSplineBasis) = name(b.superset)*" (orthonormalized, discrete)"

instantiate{T}(::Type{DiscreteOrthonormalSplineBasis}, n::Int, ::Type{T}) = DiscreteOrthonormalSplineBasis(n,3,T)

set_promote_eltype{K,T,S}(b::DiscreteOrthonormalSplineBasis{K,T}, ::Type{S}) = DiscreteOrthonormalSplineBasis(length(b),K, S)

resize{K,T}(b::DiscreteOrthonormalSplineBasis{K,T}, n::Int) = DiscreteOrthonormalSplineBasis(n, degree(b), T; oversampling=b.oversampling)

# DiscreteGram(b::DiscreteOrthonormalSplineBasis; oversampling=b.oversampling) = IdentityOperator(b, b)

change_of_basis{B<:DiscreteOrthonormalSplineBasis}(b::BSplineTranslatesBasis, ::Type{B}; options...) = sqrt(DiscreteDualGram(b; options...))
