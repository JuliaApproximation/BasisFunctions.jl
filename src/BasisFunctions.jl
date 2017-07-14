# BasisFunctions

__precompile__()

module BasisFunctions

using StaticArrays
using RecipesBase
using QuadGK

using SpecialMatrices
using FastTransforms
using Domains

import Base: +, *, /, ==, |, &, -, \, ^, .+, .*, .-, .\, ./, .^
import Base: ≈
import Base: ∘

import Base: promote, promote_rule, convert, promote_eltype, widen

import Base: length, size, start, next, done, ind2sub, sub2ind, eachindex,
        range, collect, endof, checkbounds, first, last

import Base: cos, sin, exp, log

import Base: zeros, ones, one, zero, fill!

import Base: getindex, setindex!, unsafe_getindex, eltype

import Base: isreal, iseven, isodd, real, complex

import Base: ctranspose, transpose, inv, hcat, vcat, ndims

import Base: show, showcompact, convert, similar

import Base: dct, idct

import Base: indices, normalize

import Base: broadcast

import Base: cross, ×

import Base.LinAlg: dot

# import Wavelets: primal, dual, scaling, filter, support, evaluate_periodic, evaluate_periodic_in_dyadic_points
# import Wavelets.DWT: primal, dual, scaling, wavelet, Side, Kind, DiscreteWavelet, full_dwt, full_idwt, perbound
# import Wavelets.DWT: DaubechiesWavelet, CDFWavelet, name, wavelet_index, coefficient_index
# import Wavelets.Sequences: support
# import Wavelets.Util: isdyadic


## Imports from Domains

# For intervals
import Domains: interval, leftendpoint, rightendpoint
# For maps
import Domains: matrix, vector

# composite type interface
import Domains: element, elements, nb_elements
# cartesian product utility functions
import Domains: cartesianproduct, ×, product_eltype
import Domains: mapping



## Exhaustive list of exports

# from maps/partition.jl
export PiecewiseInterval, Partition
export partition
export split_interval

# from src/tensorproducts.jl
export tensorproduct, ⊗

# from grid/grid.jl
export AbstractGrid, AbstractGrid1d, AbstractGrid2d, AbstractGrid3d,
        AbstractEquispacedGrid, EquispacedGrid, PeriodicEquispacedGrid,
        DyadicPeriodicEquispacedGrid, MidpointEquispacedGrid, RandomEquispacedGrid,
        AbstractIntervalGrid, eachelement, stepsize, ChebyshevGrid, ScatteredGrid,
        ChebyshevNodeGrid, ChebyshevExtremaGrid
export Point
export leftendpoint, rightendpoint, range, sample

# from grid/productgrid.jl
export ProductGrid

# from grid/subgrid.jl
export AbstractSubGrid, IndexSubGrid, subindices, supergrid, is_subindex,
    similar_subgrid

# from grid/mappedgrid.jl
export MappedGrid, mapped_grid, apply_map

# from operator/dimop.jl
export DimensionOperator, dimension_operator

# from bases/sets/functionset.jl
export FunctionSet, FunctionSet1d, FunctionSet2d, FunctionSet3d
export numtype, grid, left, right, support, eval_expansion, eval_set_element, eval_element
export name
export instantiate, promote_eltype, set_promote_eltype, resize
export native_index, linear_index, multilinear_index, native_size, linear_size
export is_composite
export is_basis, is_frame, is_orthogonal, is_biorthogonal, is_orthonormal
export in_support
export True, False
export approx_length, extension_size
export has_transform, has_unitary_transform, has_extension, has_derivative, has_antiderivative, has_grid
export linearize_coefficients, delinearize_coefficients, linearize_coefficients!,
    delinearize_coefficients!
export moment

# from bases/sets/span.jl
export Span
export coefficient_type, coeftype

# from bases/sets/subsets.jl
export Subset, LargeSubset, SmallSubset, SingletonSubset, indices

# from bases/sets/tensorproductset.jl
export TensorProductSet

# from bases/sets/derived_set.jl
export DerivedSet

# from bases/sets/mapped_set.jl
export MappedSet, mapped_set, mapping, superset, rescale

#from bases/sets/expansions.jl
export SetExpansion, TensorProductExpansion, coefficients, set, roots,
        random_expansion, differentiate, antidifferentiate, call_set_expansion,
        ∂x, ∂y, ∂z, ∫∂x, ∫∂y, ∫∂z, ∫, is_compatible

# from operator/operator.jl
export AbstractOperator, operator, src, dest, apply!,
        apply, apply_multiple, apply_inplace!
export matrix, diagonal, is_diagonal, is_inplace

# from operator/derived_op.jl
export ConcreteDerivedOperator

# from operator/composite_operator.jl
export CompositeOperator, compose

# from operator/special_operators.jl
export IdentityOperator, ScalingOperator, DiagonalOperator, inv_diagonal,
        CoefficientScalingOperator, MatrixOperator, FunctionOperator,
        MultiplicationOperator, WrappedOperator, UnevenSignFlipOperator, ZeroOperator,
        IndexRestrictionOperator, IndexExtensionOperator, RealifyOperator, ComplexifyOperator
# from operator/circulant_operator.jl
export CirculantOperator
# from operator/pseudo_diagonal.jl
export PseudoDiagonalOperator

# from generic/transform.jl
export transform_operator, transform_set, full_transform_operator,
    transform_operator_pre, transform_operator_post,
    transform_to_grid, transform_to_grid_pre, transform_to_grid_post,
    transform_from_grid, transform_from_grid_pre, transform_from_grid_post,
    transform_operators
# from generic/gram.jl
export grammatrix, dualgrammatrix, mixedgrammatrix, Gram, DualGram, MixedGram, eval_dualelement
export DiscreteGram, DiscreteDualGram, DiscreteMixedGram

# from generic/extension
export extension_operator, default_extension_operator, extension_size, extend,
    restriction_operator, default_restriction_operator, restriction_size, restrict,
    Extension, Restriction

# from generic/evaluation.jl
export evaluation_operator, grid_evaluation_operator, default_evaluation_operator,
    evaluation_matrix, discrete_dual_evaluation_operator

# from generic/interpolation.jl
export interpolation_operator, default_interpolation_operator, interpolation_matrix

# from generic/leastsquares.jl
export leastsquares_operator, default_leastsquares_operator, leastsquares_matrix

# from generic/approximation.jl
export approximation_operator, default_approximation_operator, approximate, discrete_approximation_operator, continuous_approximation_operator, project

# from generic/differentiation.jl
export differentiation_operator, antidifferentiation_operator, derivative_set,
    antiderivative_set, Differentiation, Antidifferentiation

# from tensorproducts.jl
export is_homogeneous, basetype, tensorproduct

# from operator/tensorproductoperator.jl
export TensorProductOperator

# from operator/block_operator.jl
export BlockOperator, block_row_operator, block_column_operator, composite_size

# from grid/discretegridspace.jl
export DiscreteGridSpace, DiscreteGridSpace1d, DiscreteGridSpaceNd, left, right,
    gridspace

# from util/functors.jl
export Cos, Sin, Exp, Log, PowerFunction, IdentityFunction

# from bases/sets/weighted_set.jl
export WeightedSet, WeightedSet1d, WeightedSet2d, WeightedSet3d,
    weightfunction, weightfun_scaling_operator


# from bases/sets/composite_set.jl
export tail

# from bases/sets/multiple_set.jl
export MultiSet, multiset, ⊕

# from bases/sets/piecewise_set.jl
export PiecewiseSet

# from bases/sets/operated_set.jl
export OperatedSet, derivative

# from bases/sets/euclidean.jl
export Cn, Rn

# from bases/fourier/fourier.jl
export FourierBasis, FourierBasisEven, FourierBasisOdd, FourierBasisNd,
    FastFourierTransform, InverseFastFourierTransform,
    FastFourierTransformFFTW, InverseFastFourierTransformFFTW,
    frequency2idx, idx2frequency,
    fourier_basis_even, fourier_basis_odd

# from bases/fourier/(co)sineseries.jl
export CosineSeries, SineSeries


# from bases/poly/chebyshev.jl
export ChebyshevBasis, ChebyshevBasisSecondKind,
    FastChebyshevTransform, InverseFastChebyshevTransform,
    FastChebyshevTransformFFTW, InverseFastChebyshevTransformFFTW

# from util/plots.jl
# export plot, plot_expansion, plot_samples, plot_error
# from util/recipes.jl
export plotgrid, postprocess

#from util/MultiArray.jl
export MultiArray

# from bases/poly/polynomials.jl and friends
export LegendreBasis, JacobiBasis, LaguerreBasis, HermiteBasis, MonomialBasis, RationalBasis

# # from bases/wavelets/bf_wavelets.jl
# export DaubechiesWaveletBasis, CDFWaveletBasis
# from bases/splines/bf_splines.jl
export SplineBasis, FullSplineBasis, PeriodicSplineBasis, NaturalSplineBasis, SplineDegree
# from bases/translates/set_of_translates.jl
export CompactPeriodicSetOfTranslates, dual, discrete_dual
# from bases/translates/translates_of_bsplines.jl
export BSplineTranslatesBasis, SymBSplineTranslatesBasis, OrthonormalSplineBasis, DiscreteOrthonormalSplineBasis

export degree, interval


using Base.Cartesian


include("util/common.jl")
include("util/composite_index.jl")
include("util/multiarray.jl")
include("util/slices.jl")
include("util/functors.jl")
include("util/domain_extensions.jl")

include("maps/partition.jl")

include("grid/grid.jl")
include("grid/productgrid.jl")
include("grid/derived_grid.jl")

include("bases/sets/functionset.jl")
include("bases/sets/span.jl")
include("generic/gram.jl")

include("operator/operator.jl")
include("operator/derived_op.jl")
include("operator/composite_operator.jl")

include("grid/discretegridspace.jl")
include("grid/mappedgrid.jl")
include("grid/intervalgrids.jl")
include("grid/scattered_grid.jl")
include("grid/subgrid.jl")

include("bases/sets/derived_set.jl")
include("bases/sets/tensorproductset.jl")
include("bases/sets/mapped_set.jl")

include("bases/sets/euclidean.jl")

include("operator/dimop.jl")

include("operator/basic_operators.jl")
include("operator/special_operators.jl")
include("operator/tensorproductoperator.jl")
include("operator/block_operator.jl")
include("operator/pseudo_diagonal.jl")
include("operator/circulant_operator.jl")



include("bases/sets/expansions.jl")


include("tensorproducts.jl")

include("generic/generic_operators.jl")

include("bases/sets/subsets.jl")
include("bases/sets/composite_set.jl")
include("bases/sets/multiple_set.jl")
include("bases/sets/piecewise_set.jl")
include("bases/sets/operated_set.jl")
include("bases/sets/weighted_set.jl")


include("bases/fourier/fouriertransforms.jl")
include("bases/fourier/fourier.jl")
include("bases/fourier/cosineseries.jl")
include("bases/fourier/sineseries.jl")

include("bases/splines/bf_splines.jl")

include("util/bsplines.jl")

include("bases/translates/set_of_translates.jl")
include("bases/translates/translates_of_bsplines.jl")

# # include("bases/wavelets/bf_wavelets.jl")

include("bases/poly/polynomials.jl")

include("bases/poly/chebyshev.jl")
include("bases/poly/legendre.jl")
include("bases/poly/jacobi.jl")
include("bases/poly/laguerre.jl")
include("bases/poly/hermite.jl")
include("bases/poly/rational.jl")


include("util/recipes.jl")


end # module
