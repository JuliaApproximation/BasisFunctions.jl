
module BasisFunctions

using StaticArrays, RecipesBase, QuadGK, DomainSets, AbstractTrees
using FFTW, LinearAlgebra, SparseArrays, FastTransforms, GenericLinearAlgebra
using Base.Cartesian

## Some specific functions of Base we merely use

using Base: IteratorSize
using SpecialFunctions: gamma
using DSP: conv
using Base.Sys: isapple

## Imports from Base of functions we extend

import Base: +, *, /, ==, |, &, -, \, ^
import Base: <, <=, >, >=
import Base: ≈
import Base: ∘

import Base: promote, promote_rule, convert, promote_eltype, widen, convert

import Base: length, size, eachindex, firstindex, lastindex,
        range, collect, first, last, copyto!
import Base: transpose, inv, hcat, vcat
import Base: checkbounds, checkbounds_indices, checkindex
import Base: getindex, setindex!, unsafe_getindex, eltype, @propagate_inbounds,
    IndexStyle, axes, axes1
import Base: broadcast, similar

import Base: cos, sin, exp, log
import Base: zeros, ones, one, zero, fill!, rand

import Base: isreal, iseven, isodd, real, complex

import Base: show, string


## Imports from LinearAlgebra
import LinearAlgebra: norm, pinv, normalize, cross, ×, dot, adjoint


## Imports from DomainSets

import DomainSets: domaintype, codomaintype, dimension, domain
# For intervals
import DomainSets: leftendpoint, rightendpoint
# For maps
import DomainSets: matrix, vector, tensorproduct

# composite type interface
import DomainSets: element, elements, numelements
# cartesian product utility functions
import DomainSets: cartesianproduct, ×, product_eltype

import DomainSets: forward_map, inverse_map

import FastGaussQuadrature: gaussjacobi

import AbstractTrees: children


## Exhaustive list of exports

# from util/indexing.jl
export LinearIndex, NativeIndex
export DefaultNativeIndex, DefaultIndexList
export value

# from util/domain_extensions.jl
export interval, circle, sphere, disk, ball, rectangle, cube, simplex
export Domain1d, Domain2d, Domain3d, Domain4d

# from maps/partition.jl
export PiecewiseInterval, Partition
export partition
export split_interval

# from src/products.jl
export tensorproduct, ⊗
export element, elements, numelements

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

# from operators/banded_operators.jl
export HorizontalBandedOperator, VerticalBandedOperator

export SparseOperator

# from bases/generic/dictionary.jl
export Dictionary, Dictionary1d, Dictionary2d, Dictionary3d
export domaintype, codomaintype, coefficient_type
export promote_domaintype, promote_domainsubtype
export grid, left, right, support, domain, codomain
export measure
export eval_expansion, eval_element, eval_element_derivative
export name
export instantiate, resize
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
export QR_solver, SVD_solver, operator
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
export Gram, DualGram, MixedGram, gram_entry
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

# from sampling/quadrature.jl
export clenshaw_curtis, fejer_first_rule, fejer_second_rule
export trapezoidal_rule, rectangular_rule

# from bases/fourier/fourier.jl
export FourierBasis, FourierBasisEven, FourierBasisOdd, FourierBasisNd,
    FastFourierTransform, InverseFastFourierTransform,
    FastFourierTransformFFTW, InverseFastFourierTransformFFTW,
    frequency2idx, idx2frequency,
    fourier_basis_even, fourier_basis_odd, pseudodifferential_operator

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

export gaussjacobi




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

include("spaces/measure.jl")
include("spaces/spaces.jl")
include("spaces/integral.jl")

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
include("sampling/quadrature.jl")

################################################################
# Trigonometric sets: Fourier, cosines and sines
################################################################

include("bases/fourier/fouriertransforms.jl")
include("bases/fourier/fourier.jl")
include("bases/fourier/cosineseries.jl")
include("bases/fourier/sineseries.jl")


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

include("test/Test.jl")


end # module
