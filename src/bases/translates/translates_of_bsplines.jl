# translates_of_bsplines.jl

abstract type PeriodicBSplineBasis{K,T} <: CompactPeriodicSetOfTranslates{T}
end

const PeriodicBSplineSpan{A,F <: PeriodicBSplineBasis} = Span{A,F}

degree(b::B) where {K,T, B<:PeriodicBSplineBasis{K,T}} = K

Gram(b::PeriodicBSplineSpan; options...) = CirculantOperator(b, b, primalgramcolumn(b; options...); options...)

function extension_operator(s1::PeriodicBSplineSpan, s2::PeriodicBSplineSpan; options...)
    @assert degree(set(s1)) == degree(set(s2))
    bspline_extension_operator(s1, s2; options...)
end

function restriction_operator(s1::PeriodicBSplineSpan, s2::PeriodicBSplineSpan; options...)
    @assert degree(set(s1)) == degree(set(s2))
    bspline_restriction_operator(s1, s2; options...)
end

function bspline_extension_operator(s1::PeriodicBSplineSpan, s2::PeriodicBSplineSpan; options...)
  @assert 2*length(s1) == length(s2)
  _binomial_circulant(s2)*IndexExtensionOperator(s1, s2, 1:2:length(s2))
end

# The calculation done in this function is equivalent to finding the pseudoinverse of the bspline_extension_operator.
function bspline_restriction_operator(s1::PeriodicBSplineSpan, s2::PeriodicBSplineSpan; options...)
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
struct BSplineTranslatesBasis{K,T,SCALED} <: PeriodicBSplineBasis{K,T}
  n               :: Int
  a               :: T
  b               :: T
  fun             :: Function
end

const BSplineTranslatesSpan{A,F <: BSplineTranslatesBasis} = Span{A,F}

BSplineTranslatesBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64; scaled = false) = scaled?
    BSplineTranslatesBasis{DEGREE,T,true}(n, T(0), T(1), x->sqrt(n)*Cardinal_b_splines.evaluate_periodic_Bspline(DEGREE, n*x, n, real(T))) :
    BSplineTranslatesBasis{DEGREE,T,false}(n, T(0), T(1), x->Cardinal_b_splines.evaluate_periodic_Bspline(DEGREE, n*x, n, real(T)))

name(b::BSplineTranslatesBasis) = name(typeof(b))*" (B spline of degree $(degree(b)))"

left_of_compact_function{K,T}(b::BSplineTranslatesBasis{K,T})::real(T) = real(T)(0)

right_of_compact_function{K,T}(b::BSplineTranslatesBasis{K,T})::real(T) = stepsize(b)*real(T)(degree(b)+1)


=={K1,K2,T1,T2}(b1::BSplineTranslatesBasis{K1,T1}, b2::BSplineTranslatesBasis{K2,T2}) = T1==T2 && K1==K2 && length(b1)==length(b2)

instantiate{T}(::Type{BSplineTranslatesBasis}, n::Int, ::Type{T}) = BSplineTranslatesBasis(n,3,T)

set_promote_domaintype{K,T,S}(b::BSplineTranslatesBasis{K,T}, ::Type{S}) = BSplineTranslatesBasis(length(b),K, S)

resize{K,T}(b::BSplineTranslatesBasis{K,T}, n::Int) = BSplineTranslatesBasis(n, degree(b), T)

# TODO find an explination for this (splines are no Chebyshev system)
# For the B spline with degree 1 (hat functions) the MidpointEquispacedGrid does not lead to evaluation_matrix that is non singular
compatible_grid(b::BSplineTranslatesBasis{K}, grid::MidpointEquispacedGrid) where {K} = iseven(K) &&
    (1+(left(b) - leftendpoint(grid))≈1) && (1+(right(b) - rightendpoint(grid))≈1) && (length(b)==length(grid))
compatible_grid(b::BSplineTranslatesBasis{K}, grid::PeriodicEquispacedGrid) where {K}= isodd(K) &&
    (1+(left(b) - leftendpoint(grid))≈1) && (1+(right(b) - rightendpoint(grid))≈1) && (length(b)==length(grid))
# we use a PeriodicEquispacedGrid in stead
grid(b::BSplineTranslatesBasis{K}) where {K} = isodd(K) ?
    PeriodicEquispacedGrid(length(b), left(b), right(b)) :
    MidpointEquispacedGrid(length(b), left(b), right(b))

function _binomial_circulant(s::Span{A,BSplineTranslatesBasis{K,T,SCALED}}) where {A,K,T,SCALED}
  c = zeros(A, length(s))
  for k in 1:K+2
    c[k] = binomial(K+1, k-1)
  end
  if SCALED
    sqrt(A(2))/(1<<K)*CirculantOperator(s, c)
  else
    A(1)/(1<<K)*CirculantOperator(s, c)
  end
end

function primalgramcolumnelement(span::Span{A,BSplineTranslatesBasis{K,T,SCALED}}, i::Int; options...) where {A,K,T,SCALED}
  r = 0
  # If size of functionspace is too small there is overlap and we can not use the
  # function squared_spline_integral which assumes no overlap.
  # Use integration as long as there is no more efficient way is implemented.
  if length(span) <= 2K+1
    return defaultprimalgramcolumnelement(span, i; options...)
  else
    # squared_spline_integral gives the exact integral (in a rational number)
    if i==1
      r = BasisFunctions.Cardinal_b_splines.squared_spline_integral(K)
    elseif 1 < i <= K+1
      r = BasisFunctions.Cardinal_b_splines.shifted_spline_integral(K,i-1)
    elseif i > length(span)-K
      r = BasisFunctions.Cardinal_b_splines.shifted_spline_integral(K,length(span)-i+1)
    end
  end
  if SCALED
    A(r)
  else
    A(r)/length(span)
  end
end

"""
  Basis consisting of symmetric, dilated, translated, and periodized cardinal B splines on the interval [0,1].

  There degree should be odd to use extension or restriction.
"""
struct SymBSplineTranslatesBasis{K,T} <: PeriodicBSplineBasis{K,T}
  n               :: Int
  a               :: T
  b               :: T
  fun             :: Function
end

const SymBSplineTranslatesSpan{A,F <: SymBSplineTranslatesBasis} = Span{A,F}

SymBSplineTranslatesBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64) =
    SymBSplineTranslatesBasis{DEGREE,T}(n, T(0), T(1), x->Cardinal_b_splines.evaluate_symmetric_periodic_Bspline(DEGREE, n*x, n, real(T)))

name(b::SymBSplineTranslatesBasis) = name(typeof(b))*" (symmetric B spline of degree $(degree(b)))"

left_of_compact_function{K,T}(b::SymBSplineTranslatesBasis{K,T})::real(T) = -right_of_compact_function(b)

right_of_compact_function{K,T}(b::SymBSplineTranslatesBasis{K,T})::real(T) = stepsize(b)*real(T)((degree(b)+1))/2


=={K1,K2,T1,T2}(b1::SymBSplineTranslatesBasis{K1,T1}, b2::SymBSplineTranslatesBasis{K2,T2}) = T1==T2 && K1==K2 && length(b1)==length(b2)

instantiate{T}(::Type{SymBSplineTranslatesBasis}, n::Int, ::Type{T}) = SymBSplineTranslatesBasis(n,3,T)

set_promote_domaintype{K,T,S}(b::SymBSplineTranslatesBasis{K,T}, ::Type{S}) = SymBSplineTranslatesBasis(length(b),K, S)

resize{K,T}(b::SymBSplineTranslatesBasis{K,T}, n::Int) = SymBSplineTranslatesBasis(n, degree(b), T)

function _binomial_circulant(s::Span{A,SymBSplineTranslatesBasis{K,T}}) where {A,K,T}
  if iseven(K)
    warn("Extension and restriction work with odd degrees only.")
    throw(MethodError())
  end
  c = zeros(T, length(s))
  c[1] = binomial(K+1, (K+1)>>1)
  for (i,k) in enumerate((K+1)>>1+1:K+1)
    c[i+1] = binomial(K+1, k)
    c[end+1-i] = binomial(K+1, k)
  end
  T(1)/(1<<K)*CirculantOperator(s, c)
end

function testprimalgramcolumnelement{K,T}(set::SymBSplineTranslatesBasis{K,T}, i::Int; options...)
  r = 0
  if length(set) <= 2degree(set)+1
    return defaultprimalgramcolumnelement(set, i; options...)
  else
    if i==1
      r = BasisFunctions.Cardinal_b_splines.squared_spline_integral(K)
    elseif 1 < i <= degree(set)+1
      r = BasisFunctions.Cardinal_b_splines.shifted_spline_integral(K,i-1)
    elseif i > length(set)-degree(set)
      r = BasisFunctions.Cardinal_b_splines.shifted_spline_integral(K,length(set)-i+1)
    end
  end
  T(r)/length(set)
end

# # TODO erator/restriction_operator can be added to PeriodicBSplineBasis in julia 0.6
# # erator{K,T,B<:PeriodicBSplineBasis{K,T}}(s1::B, s2::B; options...) =
# extension_operator{K,T}(s1::SymBSplineTranslatesBasis{K,T}, s2::SymBSplineTranslatesBasis{K,T}; options...) =
#     bspline_extension_operator(s1, s2; options...)
#
# restriction_operator{K,T}(s1::SymBSplineTranslatesBasis{K,T}, s2::SymBSplineTranslatesBasis{K,T}; options...) =
#     bspline_restriction_operator(s1, s2; options...)

"""
  Basis consisting of orthonormal basis function in the spline space of degree K.
"""
struct OrthonormalSplineBasis{K,T} <: LinearCombinationOfPeriodicSetOfTranslates{BSplineTranslatesBasis,T}
  superset     ::    BSplineTranslatesBasis{K,T}
  coefficients ::    Array{T,1}

  OrthonormalSplineBasis{K,T}(b::BSplineTranslatesBasis{K,T}; options...) where {K,T} =
    new(b, coefficients_in_other_basis(b, OrthonormalSplineBasis; options...))
end

const OrthonormalSplineSpan{A,F <: OrthonormalSplineBasis} = Span{A,F}

degree{K,T}(::OrthonormalSplineBasis{K,T}) = K

superset(b::OrthonormalSplineBasis) = b.superset
coefficients(b::OrthonormalSplineBasis) = b.coefficients

OrthonormalSplineBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64; options...) =
    OrthonormalSplineBasis{DEGREE,T}(BSplineTranslatesBasis(n,DEGREE,T); options...)

name(b::OrthonormalSplineBasis) = name(b.superset)*" (orthonormalized)"

instantiate{T}(::Type{OrthonormalSplineBasis}, n::Int, ::Type{T}) = OrthonormalSplineBasis(n,3,T)

set_promote_domaintype{K,T,S}(b::OrthonormalSplineBasis{K,T}, ::Type{S}) = OrthonormalSplineBasis(length(b),K, S)

resize{K,T}(b::OrthonormalSplineBasis{K,T}, n::Int) = OrthonormalSplineBasis(n, degree(b), T)

Gram(b::OrthonormalSplineSpan) = IdentityOperator(b, b)

change_of_basis{B<:OrthonormalSplineBasis}(b::BSplineTranslatesBasis, ::Type{B}; options...) = sqrt(DualGram(Span(b); options...))


"""
  Basis consisting of orthonormal (w.r.t. a discrete inner product) basis function in the spline space of degree K.
"""
struct DiscreteOrthonormalSplineBasis{K,T} <: LinearCombinationOfPeriodicSetOfTranslates{BSplineTranslatesBasis,T}
  superset     ::    BSplineTranslatesBasis{K,T}
  coefficients ::    Array{T,1}

  oversampling ::   T

  DiscreteOrthonormalSplineBasis{K,T}(b::BSplineTranslatesBasis{K,T}; oversampling=default_oversampling(b), options...) where {K,T} =
    new(b, coefficients_in_other_basis(b, DiscreteOrthonormalSplineBasis; oversampling=oversampling, options...), oversampling)

end

const DiscreteOrthonormalSplineSpan{A,F <: DiscreteOrthonormalSplineBasis} = Span{A,F}

degree{K,T}(::DiscreteOrthonormalSplineBasis{K,T}) = K

superset(b::DiscreteOrthonormalSplineBasis) = b.superset
coefficients(b::DiscreteOrthonormalSplineBasis) = b.coefficients
default_oversampling(b::DiscreteOrthonormalSplineBasis) = b.oversampling

==(b1::DiscreteOrthonormalSplineBasis, b2::DiscreteOrthonormalSplineBasis) =
    superset(b1)==superset(b2) && coefficients(b1) ≈ coefficients(b2) && default_oversampling(b1) == default_oversampling(b2)

DiscreteOrthonormalSplineBasis{T}(n::Int, DEGREE::Int, ::Type{T} = Float64; options...) =
    DiscreteOrthonormalSplineBasis{DEGREE,T}(BSplineTranslatesBasis(n,DEGREE,T); options...)

name(b::DiscreteOrthonormalSplineBasis) = name(superset(b))*" (orthonormalized, discrete)"

instantiate{T}(::Type{DiscreteOrthonormalSplineBasis}, n::Int, ::Type{T}) = DiscreteOrthonormalSplineBasis(n,3,T)

set_promote_domaintype{K,T,S}(b::DiscreteOrthonormalSplineBasis{K,T}, ::Type{S}) = DiscreteOrthonormalSplineBasis(length(b),K, S)

resize{K,T}(b::DiscreteOrthonormalSplineBasis{K,T}, n::Int) = DiscreteOrthonormalSplineBasis(n, degree(b), T; oversampling=default_oversampling(b))

change_of_basis{B<:DiscreteOrthonormalSplineBasis}(b::BSplineTranslatesBasis, ::Type{B}; options...) = sqrt(DiscreteDualGram(Span(b); options...))
