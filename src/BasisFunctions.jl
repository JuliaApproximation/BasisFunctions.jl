# BasisFunctions

__precompile__(true)

module BasisFunctions

using StaticArrays
using RecipesBase
using QuadGK

using SpecialMatrices
using FastTransforms
using Domains

using LinearAlgebra
using AbstractTrees

import Base: +, *, /, ==, |, &, -, \, ^
import Base: <, <=, >, >=
import Base: ≈, norm
import Base: ∘

import Base: promote, promote_rule, convert, promote_eltype, widen

import Base: length, size, start, next, done, ind2sub, sub2ind, eachindex,
        range, collect, endof, first, last
import Base: checkbounds, checkbounds_indices, checkindex

import Base: cos, sin, exp, log

import Base: zeros, ones, one, zero, fill!, rand

import Base: getindex, setindex!, unsafe_getindex, eltype

import Base: isreal, iseven, isodd, real, complex

import Base: ctranspose, transpose, inv, pinv, hcat, vcat

import Base: show, showcompact, convert, similar, string

import Base: dct, idct

import Base: indices, normalize

import Base: broadcast

import Base: cross, ×

import Base.LinAlg: dot

import WaveletsCopy.DWT: isdyadic, value, name, support, wavelet, kind
using WaveletsCopy.DWT: Prl, Dul, Side, filter, Wvl, Scl
using WaveletsCopy.DWT: EvalPeriodicScratchSpace, evaluate_periodic_in_dyadic_points!, evaluate_periodic_in_dyadic_points, dwt!
using WaveletsCopy.DWT: _evaluate_periodic_scaling_basis_in_dyadic_points!, SFilterBank, WaveletBoundary, evaluate_in_dyadic_points, idwt!
using WaveletsCopy.DWT: WaveletIndex, wavelet_indices, offset, level, wavelet_index, scaling_indices, scaling_value, scaling_index
using WaveletsCopy.DWT: Primal, Dual, scaling, filter, evaluate_periodic, evaluate_periodic_wavelet_basis_in_dyadic_points
using WaveletsCopy.DWT: Primal, Dual, scaling, Side, Kind, DiscreteWavelet, full_dwt, full_idwt, perbound, dwt, idwt
using WaveletsCopy.DWT: DaubechiesWavelet, CDFWavelet
using WaveletsCopy.Filterbanks: Filterbank
using WaveletsCopy: Sequences
using WaveletsCopy.Sequences: CompactSequence, shifted_conv, Sequence, firstindex, lastindex, PeriodicExtension
using WaveletsCopy: support_length

## Imports from Domains

import Domains: domaintype, codomaintype, dimension, domain
# For intervals
import Domains: interval, leftendpoint, rightendpoint
# For maps
import Domains: matrix, vector, tensorproduct

# composite type interface
import Domains: element, elements, nb_elements
# cartesian product utility functions
import Domains: cartesianproduct, ×, product_eltype

import Domains: forward_map, inverse_map

import FastGaussQuadrature: gaussjacobi

import AbstractTrees: children

## Exhaustive list of exports

# from util/indexing.jl
export LinearIndex, NativeIndex
export DefaultNativeIndex, DefaultIndexList
export value

# from maps/partition.jl
export PiecewiseInterval, Partition
export partition
export split_interval

# from src/products.jl
export tensorproduct, ⊗
export element, elements, nb_elements

# from grid/grid.jl
export AbstractGrid, AbstractGrid1d, AbstractGrid2d, AbstractGrid3d,
        AbstractEquispacedGrid, EquispacedGrid, PeriodicEquispacedGrid,
        DyadicPeriodicEquispacedGrid, MidpointEquispacedGrid, RandomEquispacedGrid,
        AbstractIntervalGrid, eachelement, stepsize, ScatteredGrid
export ChebyshevNodeGrid, ChebyshevGrid, ChebyshevPoints, ChebyshevExtremaGrid
export Point
export leftendpoint, rightendpoint, range

# from grid/productgrid.jl
export ProductGrid

# from grid/subgrid.jl
export AbstractSubGrid, IndexSubGrid, subindices, supergrid, is_subindex,
    similar_subgrid

# from grid/mappedgrid.jl
export MappedGrid, mapped_grid, apply_map

# from spaces/spaces.jl
export FunctionSpace

# from operator/dimop.jl
export DimensionOperator, dimension_operator

# from bases/generic/dictionary.jl
export Dictionary, Dictionary1d, Dictionary2d, Dictionary3d
export domaintype, codomaintype, coefficient_type
export promote_domaintype, promote_domainsubtype
export grid, left, right, support, domain, codomain
export eval_expansion, eval_element, eval_element_derivative
export name
export instantiate, promote_eltype, set_promote_eltype, resize
export ordering
export native_index, linear_index, multilinear_index, native_size, linear_size, native_coefficients
export is_composite
export is_basis, is_frame, is_orthogonal, is_biorthogonal, is_orthonormal
export in_support
export approx_length, extension_size
export has_transform, has_unitary_transform, has_extension, has_derivative, has_antiderivative, has_grid
export linearize_coefficients, delinearize_coefficients, linearize_coefficients!,
    delinearize_coefficients!
export moment, norm

# from bases/generic/span.jl
export Span
export coefficient_type, coeftype

# from bases/generic/subdicts.jl
export Subdictionary, LargeSubdict, SmallSubdict, SingletonSubdict
export subdict, superindices

# from bases/generic/tensorproduct_dict.jl
export TensorProductDict, TensorProductDict1, TensorProductDict2,
        TensorProductDict3, ProductIndex
export recursive_native_index

# from bases/generic/derived_dict.jl
export DerivedDict

# from bases/generic/mapped_dict.jl
export MappedDict, mapped_dict, mapping, superdict, rescale

#from bases/generic/expansions.jl
export Expansion, TensorProductExpansion
export expansion, coefficients, dictionary, roots,
        random_expansion, differentiate, antidifferentiate,
        ∂x, ∂y, ∂z, ∫∂x, ∫∂y, ∫∂z, ∫, is_compatible

# from operator/operator.jl
export AbstractOperator, DictionaryOperator
export src, dest, src_space, dest_space
export apply!, apply, apply_multiple, apply_inplace!
export matrix, diagonal, is_diagonal, is_inplace, sparse_matrix

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
#export PseudoDiagonalOperator
# from operator/generic_operators.jl
export GenericIdentityOperator

# from generic/transform.jl
export transform_operator, transform_dict, full_transform_operator,
    transform_operator_pre, transform_operator_post,
    transform_to_grid, transform_to_grid_pre, transform_to_grid_post,
    transform_from_grid, transform_from_grid_pre, transform_from_grid_post,
    transform_operators

# from generic/gram.jl
export grammatrix, dualgrammatrix, mixedgrammatrix, Gram, DualGram, MixedGram, eval_dualelement
export DiscreteGram, DiscreteDualGram, DiscreteMixedGram
export dual

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
export differentiation_operator, antidifferentiation_operator, derivative_dict,
    antiderivative_dict, Differentiation, Antidifferentiation

# from products.jl
export is_homogeneous, basetype, tensorproduct

# from operator/tensorproductoperator.jl
export TensorProductOperator

# from operator/block_operator.jl
export BlockOperator, block_row_operator, block_column_operator, composite_size

# from util/functors.jl
export Cos, Sin, Exp, Log, PowerFunction, IdentityFunction

# from bases/generic/weighted_dict.jl
export WeightedDict, WeightedDict1d, WeightedDict2d, WeightedDict3d,
    weightfunction, weightfun_scaling_operator


# from bases/generic/composite_dict.jl
export tail, numelements

# from bases/generic/multiple_dict.jl
export MultiDict, multidict, ⊕

# from bases/generic/piecewise_dict.jl
export PiecewiseDict, dictionaries

# from bases/generic/vector_dict.jl
export VectorvaluedDict

# from bases/generic/operated_dict.jl
export OperatedDict
export derivative, src_dictionary, dest_dictionary

# from bases/generic/discrete_sets.jl
export DiscreteDictionary, DiscreteVectorDictionary, DiscreteArrayDictionary
export is_discrete

# from bases/generic/gridbasis.jl
export GridBasis
export gridbasis, grid_multiplication_operator

# from sampling/sampling_operator.jl
export GridSamplingOperator
export sample

# from sampling/platform.jl
export Platform
export primal, dual, sampler, A, Zt, dual_sampler

# from bases/fourier/fourier.jl
export FourierBasis, FourierBasisEven, FourierBasisOdd, FourierBasisNd,
    FastFourierTransform, InverseFastFourierTransform,
    FastFourierTransformFFTW, InverseFastFourierTransformFFTW,
    frequency2idx, idx2frequency,
    fourier_basis_even, fourier_basis_odd

# from bases/fourier/(co)sineseries.jl
export CosineSeries, SineSeries


# from bases/poly/chebyshev.jl
export ChebyshevBasis, ChebyshevT, ChebyshevU,
    FastChebyshevTransform, InverseFastChebyshevTransform,
    FastChebyshevTransformFFTW, InverseFastChebyshevTransformFFTW

# from util/recipes.jl
export plotgrid, postprocess

# from util/MultiArray.jl
export MultiArray

# from util/domain_extensions.jl
export float_type, dimension

# from bases/poly/orthopoly.jl and friends
export LegendrePolynomials, JacobiPolynomials, LaguerrePolynomials, HermitePolynomials
export Monomials, RationalBasis, GenericOPS
export recurrence_eval, recurrence_eval_derivative, monic_recurrence_eval, monic_recurrence_coefficients
export symmetric_jacobi_matrix, roots, gauss_rule, sorted_gauss_rule, first_moment
export leading_order_coefficient

# from specialOPS.jl
export HalfRangeChebyshevIkind, HalfRangeChebyshevIIkind, WaveOPS

# # from bases/wavelets/bf_wavelets.jl
export DaubechiesWaveletBasis, CDFWaveletBasis, WaveletIndex, WaveletBasis, DaubechiesScalingBasis, CDFScalingBasis, ScalingBasis
export scaling_platform
# from bases/translates/translation_dict.jl
export CompactPeriodicTranslationDict, dual, discrete_dual
# from bases/translates/translates_of_bsplines.jl
export BSplineTranslatesBasis, SymBSplineTranslatesBasis, OrthonormalSplineBasis, DiscreteOrthonormalSplineBasis
export bspline_platform

export degree, interval

export gaussjacobi


using Base.Cartesian


include("util/common.jl")
include("util/indexing.jl")
include("util/multiarray.jl")
include("util/slices.jl")
include("util/functors.jl")
include("util/domain_extensions.jl")

include("maps/partition.jl")

include("grid/grid.jl")
include("grid/productgrid.jl")
include("grid/derived_grid.jl")

include("spaces/spaces.jl")

include("bases/generic/dictionary.jl")
include("bases/generic/span.jl")
include("generic/gram.jl")

include("bases/generic/discrete_sets.jl")
include("bases/generic/gridbasis.jl")

include("operator/operator.jl")
include("operator/derived_op.jl")
include("operator/composite_operator.jl")

include("grid/mappedgrid.jl")
include("grid/intervalgrids.jl")
include("grid/scattered_grid.jl")
include("grid/subgrid.jl")

include("bases/generic/derived_dict.jl")
include("bases/generic/complexified_dict.jl")
include("bases/generic/tensorproduct_dict.jl")
include("bases/generic/mapped_dict.jl")

include("operator/dimop.jl")

include("operator/basic_operators.jl")
include("operator/banded_operators.jl")
include("operator/special_operators.jl")
include("operator/tensorproductoperator.jl")
include("operator/block_operator.jl")
#include("operator/pseudo_diagonal.jl")
include("operator/circulant_operator.jl")

include("operator/generic_operators.jl")


include("bases/generic/expansions.jl")


include("products.jl")

include("generic/generic_operators.jl")

################################################################
# Generic dictionaries
################################################################

include("bases/generic/subdicts.jl")
include("bases/generic/composite_dict.jl")
include("bases/generic/multiple_dict.jl")
include("bases/generic/piecewise_dict.jl")
include("bases/generic/operated_dict.jl")
include("bases/generic/weighted_dict.jl")
include("bases/generic/vector_dict.jl")

################################################################
# Sampling
################################################################

include("sampling/synthesis.jl")
include("sampling/sampling_operator.jl")
include("sampling/platform.jl")

################################################################
# Trigonometric sets: Fourier, cosines and sines
################################################################

include("bases/fourier/fouriertransforms.jl")
include("bases/fourier/fourier.jl")
include("bases/fourier/cosineseries.jl")
include("bases/fourier/sineseries.jl")

################################################################
# Wavelets
################################################################

include("bases/wavelets/bf_wavelets.jl")

################################################################
# Splines: natural splines, periodic splines, B-splines, ....
################################################################


include("bases/translates/translation_dict.jl")
include("bases/translates/translates_of_bsplines.jl")


################################################################
# Polynomials: monomials and (classical) orthogonal polynomials
################################################################

include("bases/poly/polynomials.jl")
include("bases/poly/monomials.jl")
include("bases/poly/orthopoly.jl")
include("bases/poly/chebyshev.jl")
include("bases/poly/legendre.jl")
include("bases/poly/jacobi.jl")
include("bases/poly/laguerre.jl")
include("bases/poly/hermite.jl")
include("bases/poly/generic_op.jl")
include("bases/poly/specialOPS.jl")
include("bases/poly/rational.jl")

include("operator/prettyprint.jl")

include("util/recipes.jl")


end # module
