module BasisFunctions

using FastTransforms

using ArrayViews
using FixedSizeArrays
#using PyPlot
using Compat
using RecipesBase

import Base: +, *, /, ==, |, &, -, \, ^, .+, .*, .-, .\, ./, .^
import Base: ≈

import Base: promote, promote_rule, convert, promote_eltype

import Base: length, size, start, next, done, ind2sub, sub2ind, eachindex,
        range, collect, endof

import Base: cos, sin, exp, log

import Base: zeros, fill!

import Base: getindex, setindex!, eltype

import Base: isreal, iseven, isodd

import Base: ctranspose, transpose, inv, hcat, vcat, ndims

import Base: show, showcompact, call, convert, similar

import Base: dct, idct

import Base: indices, normalize

# import PyPlot: plot


## Exports

# from grid/grid.jl
export AbstractGrid, AbstractGrid1d, AbstractGrid2d, AbstractGrid3d, AbstractEquispacedGrid, EquispacedGrid, PeriodicEquispacedGrid,
        TensorProductGrid, AbstractIntervalGrid, eachelement, stepsize, ChebyshevGrid, ScatteredGrid
export dim, left, right, range, sample

# from operator/dimop.jl
export DimensionOperator, dimension_operator

# from sets/functionset.jl
export FunctionSet, FunctionSet1d, FunctionSet2d, FunctionSet3d
export numtype, grid, left, right, support, call, call!, call_expansion_with_set,
        call_expansion_with_set!, call_set, call_set!
export name
export complexify
export instantiate, promote_eltype, resize
export native_index, linear_index, native_size, linear_size
export is_basis, is_frame, is_orthogonal, is_biorthogonal
export True, False
export approx_length, extension_size
export has_transform, has_extension, has_derivative, has_antiderivative, has_grid
export linearize_coefficients, delinearize_coefficients
export moment

# from sets/functionsubset.jl
export FunctionSubSet, indices

# from sets/tensorproductset.jl
export TensorProductSet, tensorproduct, ⊗, element, elements, composite_length

# from sets/mappedsets.jl
export map, imap, map_linear, imap_linear, rescale

#from expansions.jl
export SetExpansion, TensorProductExpansion, coefficients, set, roots,
        random_expansion, differentiate, antidifferentiate,
        ∂x, ∂y, ∂z, ∫∂x, ∫∂y, ∫∂z, ∫, is_compatible

# from operator/operators.jl
export AbstractOperator, ctranspose, operator, src, dest, apply!,
        apply, apply_multiple
export matrix

# from operator/composite_operator.jl
export CompositeOperator, compose

# from operator/special_operators.jl
export IdentityOperator, ScalingOperator, DiagonalOperator, inv_diagonal,
        IdxnScalingOperator, CoefficientScalingOperator, MatrixOperator,
        MultiplicationOperator, WrappedOperator

# from generic_operator.jl
export extension_operator, restriction_operator, interpolation_operator,
    approximation_operator, transform_operator, transform_set, full_transform_operator,
    differentiation_operator, antidifferentiation_operator, derivative_set, antiderivative_set,
    approximate, evaluation_operator,
    Extension, Restriction, extend, Differentiation, AntiDifferentiation,
    extension_size, transform_pre_operator, transform_post_operator, interpolation_matrix,
    tensorproduct

# from operator/tensorproductoperator.jl
export TensorProductOperator

# from operator/block_operator.jl
export BlockOperator, block_row_operator, block_column_operator, composite_size

# from functional/functional.jl
export AbstractFunctional, EvaluationFunctional, row

# from grid/discretegridspace.jl
export DiscreteGridSpace, DiscreteGridSpace1d, DiscreteGridSpaceNd, left, right

# from util/functors.jl
export Cos, Sin, Exp, Log, PowerFunction, IdentityFunction

# from sets/normalized_set.jl
export NormalizedSet, normalize

# from sets/augmented_set.jl
export ⊕, set, fun, derivative, AugmentedSet

# from sets/multiple_set.jl
export MultiSet, multiset

# from sets/operated_set.jl
export OperatedSet

# from sets/euclidean.jl
export Cn, Rn

# from fourier/fourier.jl
export FourierBasis, FourierBasisEven, FourierBasisOdd, FourierBasisNd,
    FastFourierTransform, InverseFastFourierTransform,
    FastFourierTransformFFTW, InverseFastFourierTransformFFTW,
    frequency2idx, idx2frequency,
    fourier_basis_even, fourier_basis_odd

# from fourier/(co)sineseries.jl
export CosineSeries, SineSeries


# from poly/chebyshev.jl
export ChebyshevBasis, ChebyshevBasisSecondKind,
    FastChebyshevTransform, InverseFastChebyshevTransform,
    FastChebyshevTransformFFTW, InverseFastChebyshevTransformFFTW

# from util/plots.jl
# export plot, plot_expansion, plot_samples, plot_error
# from util/recipes.jl
export plotgrid, postprocess

# from poly/polynomials.jl and friends
export LegendreBasis, JacobiBasis, LaguerreBasis, HermiteBasis, MonomialBasis

# from bf_splines.jl
export SplineBasis, FullSplineBasis, PeriodicSplineBasis, NaturalSplineBasis, SplineDegree
export degree, interval


using Base.Cartesian


include("util/common.jl")
include("util/multiarray.jl")

include("grid/grid.jl")

include("util/slices.jl")

include("sets/functionset.jl")

include("operator/operator.jl")
include("operator/composite_operator.jl")

include("sets/tensorproductset.jl")
include("sets/mappedsets.jl")

include("sets/euclidean.jl")

include("operator/dimop.jl")

include("operator/special_operators.jl")
include("operator/tensorproductoperator.jl")
include("operator/block_operator.jl")



include("expansions.jl")

include("functional/functional.jl")

include("grid/discretegridspace.jl")

include("tensorproducts.jl")

include("util/functors.jl")

include("generic_operators.jl")

include("sets/subsets.jl")
include("sets/multiple_set.jl")
include("sets/operated_set.jl")
include("sets/augmented_set.jl")
include("sets/normalized_set.jl")


include("fourier/fouriertransforms.jl")
include("fourier/fourier.jl")
include("fourier/cosineseries.jl")
include("fourier/sineseries.jl")

include("bf_splines.jl")

include("bf_wavelets.jl")

include("poly/polynomials.jl")

include("poly/chebyshev.jl")
include("poly/legendre.jl")
include("poly/jacobi.jl")
include("poly/laguerre.jl")
include("poly/hermite.jl")

# include("util/plots.jl")
include("util/recipes.jl")

end # module
