module BasisFunctions

# We may import ApproxFun to use its implementation of FFT for BigFloat's
using ApproxFun

using ArrayViews
using FixedSizeArrays
using Debug

import Base: +, *, /, ==, |, &, -, \, ^

import Base: promote_rule, convert

import Base: length, size, start, next, done, ind2sub, sub2ind, eachindex, range, collect

import Base: cos, sin, exp, log

import Base: getindex, setindex!, eltype

import Base: isreal, iseven, isodd

import Base: ctranspose, transpose, inv, hcat, vcat

import Base: show, showcompact, call, convert, similar

import Base: dct, idct


## Exports

# from grid/grid.jl
export AbstractGrid, AbstractGrid1d, AbstractGrid2d, AbstractGrid3d, AbstractEquispacedGrid, EquispacedGrid, PeriodicEquispacedGrid,
        TensorProductGrid, AbstractIntervalGrid, eachelement, stepsize, ChebyshevGrid
export dim, left, right, range, sample

# from operator/dimop.jl
export DimensionOperator, dim_operator

# from sets/functionset.jl
export FunctionSet, AbstractFrame, AbstractBasis, AbstractBasis1d
export numtype, grid, left, right, support, call, call!, call_set, call_set!
export name
export transform_operator, differentiation_operator, approximation_operator
export complexify
export instantiate, promote_eltype, resize
export natural_index, logical_index, natural_size, logical_size
export is_basis, is_frame, is_orthogonal, is_biorthogonal, index_dim
export True, False
export approx_length, extension_size

# from sets/setfunction.jl
export SetFunction, index, functionset

# from sets/tensorproductset.jl
export TensorProductSet, tensorproduct, ⊗, sets, tp_length

# from sets/mappedsets.jl
export map, imap, map_linear, imap_linear, rescale

#from expansions.jl
export SetExpansion, TensorProductExpansion, coefficients, set, random_expansion, differentiate, antidifferentiate, ∂x, ∂y, ∂z, ∫∂x, ∫∂y, ∫∂z, ∫

# from operator/operators.jl
export AbstractOperator, CompositeOperator, OperatorTranspose, ctranspose, operator, src, dest,
    DenseOperator,  apply!
    export matrix, inv

# from operator/special_operators.jl
export IdentityOperator, ScalingOperator, DiagonalOperator, IdxnScalingOperator, CoefficientScalingOperator, MatrixOperator, WrappedOperator
# from generic_operator.jl
export extension_operator, restriction_operator, interpolation_operator, 
    approximation_operator, transform_operator, differentiation_operator,
    antidifferentiation_operator, approximate,
    evaluation_operator, normalization_operator,
    Extension, Restriction, extend, Differentiation, TransformOperator,
    extension_size, transform_normalization_operator, interpolation_matrix

# from operator/tensorproductoperator.jl
export TensorProductOperator

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

# from sets/concatenated_set.jl
export set1, set2, ConcatenatedSet

# from sets/operated_set.jl
export OperatedSet

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
export plot, plot_expansion, plot_samples, plot_error

# from poly/polynomials.jl and friends
export LegendreBasis, JacobiBasis, LaguerreBasis, HermiteBasis, MonomialBasis

# from bf_splines.jl
export SplineBasis, FullSplineBasis, PeriodicSplineBasis, NaturalSplineBasis, SplineDegree
export degree, interval


using Base.Cartesian


# Convenience definitions for the implementation of traits
typealias True Val{true}
typealias False Val{false}

(&){T1,T2}(::Type{Val{T1}}, ::Type{Val{T2}}) = Val{T1 & T2}
(|){T1,T2}(::Type{Val{T1}}, ::Type{Val{T2}}) = Val{T1 | T2}

(&){T1,T2}(::Val{T1}, ::Val{T2}) = Val{T1 & T2}()
(|){T1,T2}(::Val{T1}, ::Val{T2}) = Val{T1 | T2}()

"Return a complex type associated with the argument type."
complexify{T <: Real}(::Type{T}) = Complex{T}
complexify{T <: Real}(::Type{Complex{T}}) = Complex{T}
# In 0.5 we will be able to use Base.complex(T)
isreal{T <: Real}(::Type{T}) = True
isreal{T <: Real}(::Type{Complex{T}}) = False

# Starting with julia 0.4.3 we can just do float(T)
floatify{T <: AbstractFloat}(::Type{T}) = T
floatify(::Type{Int}) = Float64
floatify(::Type{BigInt}) = BigFloat
floatify{T}(::Type{Complex{T}}) = Complex{floatify(T)}
floatify{T}(::Type{Rational{T}}) = floatify(T)


include("grid/grid.jl")

include("util/slices.jl")

include("sets/functionset.jl")

include("sets/setfunction.jl")

include("sets/tensorproductset.jl")

include("sets/mappedsets.jl")

include("sets/euclidean.jl")

include("operator/operator.jl")

#include("dimop.jl")

include("operator/tensorproductoperator.jl")

include("expansions.jl")

include("functional/functional.jl")

include("grid/discretegridspace.jl")

include("generic_operators.jl")

include("util/functors.jl")

include("sets/concatenated_set.jl")
include("sets/operated_set.jl")
include("sets/augmented_set.jl")
include("sets/normalized_set.jl")

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

include("util/plots.jl")

end # module
