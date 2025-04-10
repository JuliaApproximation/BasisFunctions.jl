module BasisFunctions

# This file lists dependencies, imports, exports and includes.

## Dependencies

using StaticArrays, BandedMatrices, BlockArrays, SparseArrays, FillArrays
using ToeplitzMatrices, LinearAlgebra, GenericLinearAlgebra
using FFTW, GenericFFT
using DomainSets, DomainIntegrals
using CompositeTypes, CompositeTypes.Display

using QuadGK, Base.Cartesian

using Reexport


## Some specific functions we merely use

using Base: IteratorSize
using IterativeSolvers: lsqr, lsmr
using SpecialFunctions: gamma
using MacroTools: @forward

## Imports

import Base:
    +, *, /, ==, |, &, -, \, ^,
    <, <=, >, >=,
    ≈,
    ∘, ∘

import Base:
    promote, promote_rule, convert, promote_eltype, widen, convert,
    # array methods
    length, size, eachindex, iterate, firstindex, lastindex, range, collect,
    issubset,
    first, last, copyto!, tail,
    transpose, inv, hcat, vcat, hvcat,
    getindex, setindex!, eltype, @propagate_inbounds,
    IndexStyle, axes, axes1,
    broadcast, similar,
    checkbounds, checkbounds_indices, checkindex,
    zeros, ones, one, zero, fill!, rand,
    # math functions
    cos, sin, exp, log, sqrt,
    # numbers
    isreal, iseven, isodd, real, complex, conj, diff,
    # io
    show, string

import Base.Broadcast: broadcasted, DefaultArrayStyle, broadcast_shape


import LinearAlgebra:
    norm, pinv, normalize, cross, ×, dot, adjoint, mul!, rank,
    diag, isdiag, eigvals, issymmetric, svdvals
import SparseArrays: sparse


import BlockArrays.BlockVector
export BlockVector

import CompositeTypes:
    iscomposite, component, components, ncomponents

import DomainSets:
    domain, domaintype, codomaintype, dimension,
    indomain, approx_indomain,
    # intervals
    leftendpoint, rightendpoint, endpoints,
    # maps
    matrix, vector,
    forward_map, inverse_map,
    applymap, jacobian,
    ScalarAffineMap,
    # composite types
    factors, factor, nfactors,
    # products
    ×,
    # utils
    prectype, numtype

import DomainIntegrals:
    integral, integrate,
    support, moment,
    weightfun, weightfunction,
    unsafe_weightfun, unsafe_weightfunction,
    weight, unsafe_weight,
    points, weights,
    iscontinuous, isdiscrete,
    isnormalized, isuniform,
    ismappedmeasure,
    jacobi_α, jacobi_β, laguerre_α
using DomainIntegrals:
    ProductWeight,
    AbstractJacobiWeight,
    UltrasphericalWeight,
    islebesguemeasure
export integral

@reexport using GridArrays
import GridArrays:
    subindices,
    resize,
    covering,
    →, rescale,
    isperiodic

using GridArrays: SubGrid, MaskedGrid, IndexSubGrid


## Exports

# Re-exports
export component, components, ncomponents, iscomposite,
    factors, factor, nfactors,
    prectype, numtype,
    ChebyshevNodes, ChebyshevExtremae,
    Point,
    leftendpoint, rightendpoint, range,
    ×,
    points, weights, discrete_weight,
    support,
    Measure, Weight, DiscreteWeight,
    dimension,
    ..,
    integral

# from bases/dictionary/indexing.jl
export LinearIndex, value

# from maps/partition.jl
export PiecewiseInterval, Partition,
    partition,
    split_interval

# from src/util/products.jl
export tensorproduct, ⊗
export ishomogeneous, basetype


# from GridArrays
export ProductGrid
export SubGrid, subindices, supergrid, issubindex,
    similar_subgrid

# from spaces/measure.jl
export innerproduct,
    FourierWeight, ChebyshevTWeight, ChebyshevWeight, ChebyshevUWeight,
    LegendreWeight, JacobiWeight, OPSNodesMeasure, discretemeasure,
    MappedWeight, ProductWeight,
    isnormalized,
    mappedmeasure, productmeasure, submeasure, weight, lebesguemeasure,
    supermeasure, applymeasure

# from spaces/spaces.jl
export GenericFunctionSpace, space, MeasureSpace, FourierSpace, ChebyshevTSpace,
    ChebyshevSpace, L2

# from bases/dictionary/dictionary.jl
export Dictionary, Dictionary1d, Dictionary2d, Dictionary3d,
    interpolation_grid, left, right, domain, codomain,
    domaintype, codomaintype, coefficienttype,
    measure, hasmeasure,
    eval_expansion, eval_element, eval_element_derivative, eval_gradient,
    resize,
    ordering,
    native_index, linear_index, multilinear_index, native_size, linear_size, native_coefficients,
    isbasis, isframe, isorthogonal, isbiorthogonal, isorthonormal,
    in_support,
    dimensions, approx_length, extensionsize,
    hastransform, hasextension, hasderivative, hasantiderivative, hasinterpolationgrid,
    linearize_coefficients, delinearize_coefficients, linearize_coefficients!,
    delinearize_coefficients!,
    moment, norm

# from bases/dictionary/span.jl
export Span

# from bases/modifiers/subdicts.jl
export Subdictionary, DenseSubdict, SparseSubdict,
    subdict, superindices

# from bases/modifiers/tensorproduct_dict.jl
export TensorProductDict, ProductIndex,
    recursive_native_index

# from bases/modifiers/derived_dict.jl
export DerivedDict, superdict

# from bases/modifiers/paramdict.jl
export param_dict, ParamDict, image

#from bases/dictionary/expansions.jl
export Expansion, TensorProductExpansion,
    expansion, coefficients, dictionary, roots,
    random_expansion, differentiate, antidifferentiate,
    ∂x, ∂y, ∂z, ∫∂x, ∫∂y, ∫∂z, ∫

# from operator/operator.jl
export AbstractOperator, DictionaryOperator,
    src, dest, src_space, dest_space,
    apply!, apply, apply_multiple, apply_inplace!,
    matrix, diag, isdiag, isinplace

# from operator/composite_operator.jl
export CompositeOperator, compose

# from operator/special_operators.jl
export IdentityOperator, ScalingOperator, scalar, DiagonalOperator,
    ArrayOperator, FunctionOperator,
    MultiplicationOperator,
    IndexRestriction, IndexExtension,
    HorizontalBandedOperator, VerticalBandedOperator, CirculantOperator, Circulant

# from operator/solvers.jl
export QR_solver, SVD_solver, regularized_SVD_solver, operator, LSQR_solver, LSMR_solver

# from operator/generic_operators.jl
export GenericIdentityOperator

# from computations/transform.jl
export transform, transform_dict, transform_to_grid, transform_from_grid

# from bases/dictionary/orthog.jl
export gramelement, gram, dual, mixedgram, gramdual

# from computations/evaluation.jl
export evaluation_matrix

# from computations/approximation.jl
export approximation, default_approximation, approximate,
    discrete_approximation, continuous_approximation, project,
    interpolation, interpolation_fun, default_interpolation,
    interpolation_matrix

# from operator/tensorproductoperator.jl
export TensorProductOperator

# from operator/block_operator.jl
export BlockOperator, block_row_operator, block_column_operator, composite_size

# from bases/modifiers/weighted_dict.jl
export WeightedDict, WeightedDict1d, WeightedDict2d, WeightedDict3d,
    weightfunction, weightfun_scaling_operator


# from bases/modifiers/composite_dict.jl
export tail

# from bases/modifiers/multiple_dict.jl
export MultiDict, multidict, ⊕

# from bases/modifiers/piecewise_dict.jl
export PiecewiseDict, dictionaries

# from bases/modifiers/operated_dict.jl
export OperatedDict,
    orthogonalize

# from bases/dictionary/discrete_sets.jl
export DiscreteDictionary, DiscreteVectorDictionary, DiscreteArrayDictionary,
    isdiscrete

# from bases/dictionary/gridbasis.jl
export GridBasis,
    grid, gridbasis, weights

# from sampling/sampling_operator.jl
export GridSampling, ProjectionSampling, AnalysisOperator, SynthesisOperator,
    sample

# from sampling/quadrature.jl
export clenshaw_curtis, fejer_first_rule, fejer_second_rule,
    trapezoidal_rule, rectangular_rule

# from sampling/normalization.jl
export sampling_normalization

# from bases/fourier/fourier.jl
export Fourier,
    FastFourierTransform, InverseFastFourierTransform,
    FastFourierTransformFFTW, InverseFastFourierTransformFFTW,
    frequency2idx, idx2frequency

# from bases/fourier/(co)sineseries.jl
export CosineSeries, SineSeries

# from bases/fourier/sinc.jl
export PeriodicSincFunctions

# from bases/fourier/trig.jl
export TrigSeries

# from bases/poly/chebyshev.jl
export ChebyshevT, ChebyshevU,
    FastChebyshevTransform, InverseFastChebyshevTransform,
    FastChebyshevTransformFFTW, InverseFastChebyshevTransformFFTW

# from bases/poly/orthopoly.jl and friends
export Legendre, Jacobi, Laguerre, Hermite, Ultraspherical,
    Monomials, GenericOPS,
    recurrence_eval, recurrence_eval_derivative, monic_recurrence_eval,
    monic_recurrence_coefficients,
    symmetric_jacobi_matrix, roots, gauss_rule, sorted_gauss_rule, first_moment,
    leading_order_coefficient,
    ops_roots


## Includes

include("util/common.jl")
include("util/arrays/specialarrays.jl")
include("util/arrays/outerproductarrays.jl")

include("maps/partition.jl")

include("spaces/measure.jl")
include("spaces/discretemeasure.jl")
include("spaces/spaces.jl")
include("sampling/quadweights.jl")

include("bases/dictionary/indexing.jl")
include("bases/dictionary/dictionary.jl")
include("bases/dictionary/dict_evaluation.jl")
include("bases/dictionary/dict_moments.jl")
include("bases/dictionary/promotion.jl")
include("bases/dictionary/span.jl")
include("bases/dictionary/orthog.jl")
include("bases/dictionary/duality.jl")
include("bases/dictionary/discrete_sets.jl")
include("bases/dictionary/gridbasis.jl")
include("bases/dictionary/basisfunction.jl")

include("operator/operator.jl")
include("operator/composite_operator.jl")

include("bases/modifiers/derived_dict.jl")
include("bases/modifiers/complexified_dict.jl")
include("bases/modifiers/tensorproduct_dict.jl")
include("bases/modifiers/mapped_dict.jl")
include("bases/modifiers/paramdict.jl")

include("operator/arrayoperator.jl")
include("operator/solvers.jl")
include("operator/special_operators.jl")
include("operator/tensorproductoperator.jl")
include("operator/block_operator.jl")
include("operator/arithmetics.jl")
include("operator/simplify.jl")
include("operator/orthogonality.jl")

include("operator/generic_operators.jl")


include("bases/dictionary/expansions.jl")


include("util/products.jl")

include("computations/transform.jl")
include("computations/evaluation.jl")
include("computations/approximation.jl")
include("computations/differentiation.jl")
include("computations/conversion.jl")
include("computations/product.jl")
include("computations/fun.jl")

################################################################
# Generic dictionaries
################################################################

include("bases/modifiers/subdicts.jl")
include("bases/modifiers/composite_dict.jl")
include("bases/modifiers/multiple_dict.jl")
include("bases/modifiers/piecewise_dict.jl")
include("bases/modifiers/operated_dict.jl")
include("bases/modifiers/weighted_dict.jl")
include("bases/modifiers/domain_dict.jl")
include("bases/modifiers/logic.jl")

include("bases/dictionary/custom_dictionary.jl")

################################################################
# Sampling
################################################################

include("sampling/synthesis.jl")
include("sampling/sampling_operator.jl")
include("sampling/quadrature.jl")
include("sampling/normalization.jl")
include("sampling/interaction.jl")


################################################################
# Trigonometric sets: Fourier, cosines and sines
################################################################

include("bases/fourier/fouriertransforms.jl")
include("bases/fourier/fourier.jl")
include("bases/fourier/cosineseries.jl")
include("bases/fourier/sineseries.jl")
include("bases/fourier/sinc.jl")
include("bases/fourier/trig.jl")


################################################################
# Polynomials: monomials and (classical) orthogonal polynomials
################################################################

include("bases/poly/polynomials.jl")
include("bases/poly/monomials.jl")
include("bases/poly/quadrature.jl")
include("bases/poly/ops/dlmf.jl")
include("bases/poly/ops/orthopoly.jl")
include("bases/poly/chebyshev/chebyshev.jl")
include("bases/poly/ops/legendre.jl")
include("bases/poly/ops/jacobi.jl")
include("bases/poly/ops/laguerre.jl")
include("bases/poly/ops/hermite.jl")
include("bases/poly/ops/generic_op.jl")
include("bases/poly/ops/specialOPS.jl")
include("bases/poly/ops/connections.jl")

include("util/plot.jl")

include("test/Test.jl")

include("examples/pwconstants.jl")

include("deprecated.jl")

end # module
